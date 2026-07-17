"""Pluggable decision engines (plan R-E).

The service can decide each transaction with either:
  * ThresholdEngine — the original XGBoost-probability-vs-threshold rule
    (approve / decline only), or
  * PolicyEngine   — the trained DQN policy (approve / decline / REVIEW), which
    builds the same 32-d state the RL env used and picks the argmax-Q action.

Both return a ScoreResponse so the API and dashboard treat them uniformly, and both
stay within the platform's <100 ms budget (the Q-network is a tiny MLP).
"""
from __future__ import annotations

import time
from collections import deque
from typing import Optional

import numpy as np
import pandas as pd

from antifraud.common.schemas import (DecisionAction, FEATURE_ORDER, ScoreResponse,
                                       Transaction)
from antifraud.serving.scorer import Scorer


def _finalize(action: DecisionAction, proba: float, txn: Transaction, engine: str,
              scorer: Scorer, reason_codes, latency_ms: float, threshold: float) -> ScoreResponse:
    flagged = action != DecisionAction.approve
    return ScoreResponse(
        probability=proba,
        decision="fraud" if flagged else "legit",
        action=action,
        engine=engine,
        threshold=threshold,
        amount=txn.Amount,
        amount_at_risk=txn.Amount if flagged else 0.0,
        reason_codes=reason_codes if flagged else [],
        latency_ms=round(latency_ms, 3),
        model_version=scorer.version,
    )


class ThresholdEngine:
    name = "threshold"

    def __init__(self, scorer: Scorer):
        self.scorer = scorer

    def decide(self, txn: Transaction) -> ScoreResponse:
        r = self.scorer.score(txn)   # already applies the frozen threshold
        action = DecisionAction.decline if r.decision == "fraud" else DecisionAction.approve
        return _finalize(action, r.probability, txn, self.name, self.scorer,
                         r.reason_codes, r.latency_ms, self.scorer.threshold)


class PolicyEngine:
    name = "dqn"
    _ACTIONS = [DecisionAction.approve, DecisionAction.decline, DecisionAction.review]

    def __init__(self, scorer: Scorer, agent, metadata: dict, lean_state: bool = False,
                 window: int = 500, review_budget_frac: float = 0.02):
        self.scorer = scorer
        self.agent = agent
        self.metadata = metadata
        self.lean_state = lean_state
        self.policy_version = metadata.get("version", "unknown")
        # Runtime approximations of the env's context features (no labels at serve time):
        # a rolling review budget and a rolling flag-rate proxy for the fraud rate.
        self.window = window
        self.capacity = max(1, int(review_budget_frac * window))
        self._recent_reviews = deque(maxlen=window)
        self._recent_flags = deque(maxlen=window)

    def _context(self) -> tuple[float, float]:
        budget_used = sum(self._recent_reviews)
        budget_frac = max(0.0, 1.0 - budget_used / self.capacity)
        fraud_rate = float(np.mean(self._recent_flags)) if self._recent_flags else 0.0
        return budget_frac, fraud_rate

    def decide(self, txn: Transaction) -> ScoreResponse:
        t0 = time.perf_counter()
        r = self.scorer.score(txn)   # proba + reason codes (reuses the model)
        scaled = self.scorer.preprocessor.transform(
            pd.DataFrame([txn.to_vector()], columns=FEATURE_ORDER))[0]
        budget_frac, fraud_rate = self._context()
        if self.lean_state:   # [score, scaled amount, budget, fraud rate]
            state = np.array([r.probability, scaled[-1], budget_frac, fraud_rate], np.float32)
        else:
            state = np.concatenate([scaled, [r.probability, budget_frac, fraud_rate]]).astype(np.float32)

        action = self._ACTIONS[self.agent.act_greedy(state)]
        # Guard: never review with an exhausted budget — fall back to decline.
        if action == DecisionAction.review and budget_frac <= 0.0:
            action = DecisionAction.decline

        self._recent_reviews.append(1 if action == DecisionAction.review else 0)
        self._recent_flags.append(1 if r.probability >= 0.5 else 0)

        latency_ms = (time.perf_counter() - t0) * 1000.0
        return _finalize(action, r.probability, txn, self.name, self.scorer,
                         r.reason_codes, latency_ms, self.scorer.threshold)
