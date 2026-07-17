"""Phase 0 — leakage-free training that emits a versioned artifact bundle.

Produces the trustworthy, serializable hot-path model the real-time service loads.
Design decisions (mapped to the plan):

  A1  Threshold is chosen on a dedicated VALIDATION split carved from train, then
      frozen and applied unchanged to the untouched test set — no test-set leakage.
  A3  Scaling lives in a single fitted ColumnTransformer serialized with the model,
      so serving transforms features identically (no train/serve skew).
  A4  Headline metrics lead with PR-AUC (Average Precision) and MCC, not ROC-AUC.
  A6  A cost-optimal threshold is also computed and stored.
  A7  Cost-sensitive learning via scale_pos_weight instead of SMOTE — avoids the
      probability distortion SMOTE causes, which matters because we score live.
  A12 Global seeds set for reproducibility.
  A14 Everything saved as a versioned bundle via the registry.

Run:  python -m antifraud.training.build_artifact [--fraction 0.2] [--split random|temporal]
"""
from __future__ import annotations

import argparse
import random

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             matthews_corrcoef, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler
from xgboost import XGBClassifier

from antifraud.common.config import DATA_PATH, RANDOM_SEED, REVIEW_COST
from antifraud.common.cost_model import best_cost_threshold, evaluate_cost
from antifraud.common.schemas import FEATURE_ORDER
from antifraud.registry.artifact import new_version, save_bundle
# Reuse the project's existing F1-optimal threshold finder (classifiers/utils.py).
from classifiers.utils import find_best_threshold

V_COLS = [f"V{i}" for i in range(1, 29)]


def set_seeds(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)


def load_frame(fraction: float | None) -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df = df.drop_duplicates()
    if fraction and fraction < 1.0:
        # Keep all fraud; subsample legit — preserves positives for a fast run.
        fraud = df[df.Class == 1]
        legit = df[df.Class == 0].sample(frac=fraction, random_state=RANDOM_SEED)
        df = pd.concat([fraud, legit]).sort_index()
    return df


def split(df: pd.DataFrame, mode: str):
    """Return (train, val, test) frames. Random-stratified or time-ordered (A2)."""
    if mode == "temporal":
        df = df.sort_values("Time")
        n = len(df)
        train = df.iloc[: int(n * 0.7)]
        val = df.iloc[int(n * 0.7): int(n * 0.8)]
        test = df.iloc[int(n * 0.8):]
        return train, val, test
    # random stratified: 80/20 test, then carve 20% of train as validation
    train_full, test = train_test_split(
        df, test_size=0.2, random_state=RANDOM_SEED, stratify=df.Class)
    train, val = train_test_split(
        train_full, test_size=0.2, random_state=RANDOM_SEED, stratify=train_full.Class)
    return train, val, test


def xy(frame: pd.DataFrame):
    return frame[FEATURE_ORDER], frame["Class"].to_numpy()


def build_preprocessor() -> ColumnTransformer:
    # StandardScaler on the PCA components, RobustScaler on Amount (matches the
    # original preprocessing intent). Output column order == FEATURE_ORDER.
    return ColumnTransformer(
        transformers=[
            ("v", StandardScaler(), V_COLS),
            ("amount", RobustScaler(), ["Amount"]),
        ],
        remainder="drop",
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a versioned fraud-model artifact.")
    ap.add_argument("--fraction", type=float, default=1.0,
                    help="Fraction of legit rows to use (all fraud kept). 1.0 = full data.")
    ap.add_argument("--split", choices=["random", "temporal"], default="random")
    args = ap.parse_args()

    set_seeds()
    print(f"[build] loading {DATA_PATH} (fraction={args.fraction}, split={args.split})")
    df = load_frame(args.fraction)
    train, val, test = split(df, args.split)
    print(f"[build] rows  train={len(train)}  val={len(val)}  test={len(test)}")

    X_train, y_train = xy(train)
    X_val, y_val = xy(val)
    X_test, y_test = xy(test)

    pre = build_preprocessor()
    Xtr = pre.fit_transform(X_train)
    Xval = pre.transform(X_val)
    Xte = pre.transform(X_test)

    # A7: cost-sensitive weighting instead of SMOTE.
    pos = int((y_train == 1).sum())
    neg = int((y_train == 0).sum())
    spw = neg / max(pos, 1)
    print(f"[build] class balance  neg={neg}  pos={pos}  scale_pos_weight={spw:.1f}")

    clf = XGBClassifier(
        n_estimators=400, max_depth=6, learning_rate=0.1,
        subsample=0.9, colsample_bytree=0.8, scale_pos_weight=spw,
        eval_metric="aucpr", tree_method="hist",
        random_state=RANDOM_SEED, n_jobs=-1,
    )
    clf.fit(Xtr, y_train)

    # A1: choose threshold on VALIDATION, freeze it, evaluate on TEST.
    val_proba = clf.predict_proba(Xval)[:, 1]
    f1_threshold = float(find_best_threshold(y_val, val_proba))

    cost = best_cost_threshold(y_val, val_proba, X_val["Amount"].to_numpy(), REVIEW_COST)
    cost_threshold = cost.threshold
    print(f"[build] frozen thresholds  f1={f1_threshold:.4f}  cost={cost_threshold:.4f}")

    # Evaluate on the untouched test set at the frozen F1 threshold.
    test_proba = clf.predict_proba(Xte)[:, 1]
    y_pred = (test_proba >= f1_threshold).astype(int)
    test_cost = evaluate_cost(y_test, test_proba, X_test["Amount"].to_numpy(),
                              f1_threshold, REVIEW_COST)

    metrics = {
        "pr_auc": float(average_precision_score(y_test, test_proba)),   # A4 headline
        "mcc": float(matthews_corrcoef(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, test_proba)),            # secondary
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "test_total_cost": test_cost.total_cost,
        "test_amount_saved": test_cost.amount_saved,
    }
    print("[build] TEST metrics (frozen threshold, no leakage):")
    for k in ("pr_auc", "mcc", "precision", "recall", "f1", "roc_auc"):
        print(f"          {k:>10}: {metrics[k]:.4f}")
    print(f"          amount_saved: £{metrics['test_amount_saved']:,.0f}  "
          f"total_cost: £{metrics['test_total_cost']:,.0f}")

    version = new_version()
    metadata = {
        "model": "XGBoost",
        "feature_order": FEATURE_ORDER,
        "seed": RANDOM_SEED,
        "split": args.split,
        "fraction": args.fraction,
        "scale_pos_weight": spw,
        "review_cost": REVIEW_COST,
        "metrics": metrics,
        "n_train": len(train), "n_val": len(val), "n_test": len(test),
    }
    thresholds = {"f1": f1_threshold, "cost": cost_threshold, "default": 0.5}
    out = save_bundle(version, clf.get_booster(), pre, thresholds, metadata)
    print(f"[build] saved artifact bundle -> {out}")


if __name__ == "__main__":
    main()
