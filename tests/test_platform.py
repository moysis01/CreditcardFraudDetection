"""Verification suite for the real-time anti-fraud platform.

Covers the correctness guarantees the plan calls out (plan Verification):
  * no train/test leakage in the split,
  * frozen thresholds are valid probabilities,
  * artifact round-trip parity (fresh load reproduces the same score),
  * the live API contract (via FastAPI TestClient — no running server needed).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from antifraud.common.schemas import FEATURE_ORDER, Transaction
from antifraud.common.cost_model import best_cost_threshold, evaluate_cost


# ---- schema / feature contract -----------------------------------------
def test_feature_order_excludes_time_includes_amount():
    assert "Time" not in FEATURE_ORDER
    assert FEATURE_ORDER[-1] == "Amount"
    assert len(FEATURE_ORDER) == 29
    assert FEATURE_ORDER[:2] == ["V1", "V2"]


def test_transaction_to_vector_matches_feature_order():
    txn = Transaction(**{k: float(i) for i, k in enumerate(FEATURE_ORDER)})
    assert txn.to_vector() == [float(i) for i in range(len(FEATURE_ORDER))]


# ---- no-leakage split ---------------------------------------------------
def test_split_produces_disjoint_sets():
    from antifraud.training.build_artifact import split
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        **{c: rng.normal(size=500) for c in FEATURE_ORDER},
        "Time": np.arange(500),
        "Class": rng.integers(0, 2, size=500),
    })
    train, val, test = split(df, "random")
    idx_train, idx_val, idx_test = set(train.index), set(val.index), set(test.index)
    assert idx_train.isdisjoint(idx_val)
    assert idx_train.isdisjoint(idx_test)
    assert idx_val.isdisjoint(idx_test)
    assert len(idx_train) + len(idx_val) + len(idx_test) == 500


# ---- cost model ---------------------------------------------------------
def test_cost_threshold_prefers_catching_expensive_fraud():
    # one big fraud + one cheap legit; missing the fraud must cost more than a review
    y = np.array([1, 0])
    proba = np.array([0.4, 0.4])
    amounts = np.array([1000.0, 5.0])
    res = best_cost_threshold(y, proba, amounts, review_cost=3.0)
    # at a threshold below 0.4 we catch the fraud; total cost should be small
    assert res.total_cost <= evaluate_cost(y, proba, amounts, 0.99, 3.0).total_cost


# ---- artifact round-trip parity ----------------------------------------
def _bundle_available():
    from antifraud.registry.artifact import resolve_version
    try:
        resolve_version("latest")
        return True
    except FileNotFoundError:
        return False


needs_bundle = pytest.mark.skipif(
    not _bundle_available(),
    reason="no model bundle; run python -m antifraud.training.build_artifact")


@needs_bundle
def test_thresholds_are_valid_probabilities():
    from antifraud.registry.artifact import load_bundle
    b = load_bundle("latest")
    for key in ("f1", "cost", "default"):
        assert 0.0 <= b.thresholds[key] <= 1.0


@needs_bundle
def test_artifact_roundtrip_is_deterministic():
    from antifraud.serving.scorer import Scorer
    s1 = Scorer("latest")
    s2 = Scorer("latest")  # fresh independent load
    txn = Transaction(**{k: 0.0 for k in FEATURE_ORDER})
    r1, r2 = s1.score(txn), s2.score(txn)
    assert r1.probability == pytest.approx(r2.probability)
    assert r1.model_version == r2.model_version


# ---- live API contract (TestClient, no server) --------------------------
@needs_bundle
def test_api_score_and_health():
    from fastapi.testclient import TestClient
    from antifraud.serving.app import app
    with TestClient(app) as client:            # triggers startup -> loads model
        assert client.get("/healthz").json()["model_loaded"] is True
        payload = {k: 0.0 for k in FEATURE_ORDER}
        r = client.post("/score", json=payload).json()
        assert set(r) >= {"probability", "decision", "threshold", "latency_ms"}
        assert 0.0 <= r["probability"] <= 1.0
        assert r["decision"] in ("fraud", "legit")


@needs_bundle
def test_api_latency_under_budget():
    """Sanity check the speed goal: single-score server latency well under 100ms."""
    from fastapi.testclient import TestClient
    from antifraud.serving.app import app
    with TestClient(app) as client:
        payload = {k: 0.0 for k in FEATURE_ORDER}
        latencies = [client.post("/score", json=payload).json()["latency_ms"]
                     for _ in range(50)]
        assert max(latencies) < 100.0
