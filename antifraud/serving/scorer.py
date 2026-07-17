"""The hot-path scorer — loads the artifact once and scores transactions fast.

Speed design (plan B1-B3, B6):
  * Single gradient-boosted-tree model (XGBoost). CNN/ensemble stay offline.
  * Artifact (booster + fitted preprocessor + frozen threshold) loaded ONCE at
    construction into this singleton — never per request.
  * Native XGBoost prediction on a DMatrix (sub-millisecond for one row).
  * SHAP reason codes via booster's built-in `pred_contribs` — no extra `shap`
    dependency — and computed ONLY for flagged transactions to protect latency.
"""
from __future__ import annotations

import time
from typing import List, Sequence

import numpy as np
import pandas as pd
import xgboost as xgb

from antifraud.common.config import TOP_REASON_CODES
from antifraud.common.schemas import (FEATURE_ORDER, ReasonCode, ScoreResponse,
                                       Transaction)
from antifraud.registry.artifact import load_bundle


class Scorer:
    def __init__(self, version: str = "latest", threshold_key: str = "f1"):
        bundle = self.bundle = load_bundle(version)
        self.version = bundle.version
        self.booster = bundle.booster
        self.preprocessor = bundle.preprocessor
        self.feature_order = bundle.feature_order
        self.thresholds = bundle.thresholds
        self.threshold_key = threshold_key
        self.threshold = float(bundle.thresholds[threshold_key])

    def set_threshold(self, key_or_value) -> float:
        """Move the operating threshold live (dashboard console, plan C7)."""
        if isinstance(key_or_value, str):
            self.threshold_key = key_or_value
            self.threshold = float(self.thresholds[key_or_value])
        else:
            self.threshold_key = "custom"
            self.threshold = float(key_or_value)
        return self.threshold

    # ---- core prediction -------------------------------------------------
    def _to_dmatrix(self, rows: Sequence[Sequence[float]]) -> xgb.DMatrix:
        df = pd.DataFrame(list(rows), columns=FEATURE_ORDER)
        transformed = self.preprocessor.transform(df)
        return xgb.DMatrix(transformed, feature_names=self.feature_order)

    def _reason_codes(self, dmatrix: xgb.DMatrix, idx: int) -> List[ReasonCode]:
        # pred_contribs returns (n, n_features + 1); last col is the bias term.
        contribs = self.booster.predict(dmatrix, pred_contribs=True)[idx][:-1]
        order = np.argsort(np.abs(contribs))[::-1][:TOP_REASON_CODES]
        return [ReasonCode(feature=self.feature_order[i],
                           contribution=float(contribs[i])) for i in order]

    def score_many(self, txns: List[Transaction]) -> List[ScoreResponse]:
        start = time.perf_counter()
        rows = [t.to_vector() for t in txns]
        dmatrix = self._to_dmatrix(rows)
        probas = self.booster.predict(dmatrix)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        per = elapsed_ms / max(len(txns), 1)

        out: List[ScoreResponse] = []
        for i, (txn, p) in enumerate(zip(txns, probas)):
            p = float(p)
            is_fraud = p >= self.threshold
            out.append(ScoreResponse(
                probability=p,
                decision="fraud" if is_fraud else "legit",
                threshold=self.threshold,
                amount=txn.Amount,
                amount_at_risk=txn.Amount if is_fraud else 0.0,
                reason_codes=self._reason_codes(dmatrix, i) if is_fraud else [],
                latency_ms=round(per, 3),
                model_version=self.version,
            ))
        return out

    def score(self, txn: Transaction) -> ScoreResponse:
        return self.score_many([txn])[0]
