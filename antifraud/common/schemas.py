"""Shared pydantic schemas — the single definition of the request/response
contract used by the serving API, the streaming simulator, and the tests.

The feature order here is authoritative: V1..V28 followed by Amount (Time is
dropped, exactly as the training pipeline does in preprocessing/preprocess.py).
"""
from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field

# Authoritative feature order for the model. Time is intentionally excluded.
FEATURE_ORDER: List[str] = [f"V{i}" for i in range(1, 29)] + ["Amount"]


class DecisionAction(str, Enum):
    """What the decision engine chose to do with a transaction.

    The threshold engine only ever emits approve/decline; the RL policy engine can
    also emit `review` (route to a human, spending review budget)."""
    approve = "approve"
    decline = "decline"
    review = "review"


class Transaction(BaseModel):
    """A single transaction to score. V1..V28 are the dataset's PCA components."""

    V1: float; V2: float; V3: float; V4: float; V5: float; V6: float; V7: float
    V8: float; V9: float; V10: float; V11: float; V12: float; V13: float; V14: float
    V15: float; V16: float; V17: float; V18: float; V19: float; V20: float; V21: float
    V22: float; V23: float; V24: float; V25: float; V26: float; V27: float; V28: float
    Amount: float = Field(..., ge=0, description="Transaction amount")

    def to_vector(self) -> List[float]:
        """Return features in the authoritative FEATURE_ORDER."""
        return [getattr(self, name) for name in FEATURE_ORDER]


class ReasonCode(BaseModel):
    """A single SHAP-style contribution explaining a flagged transaction."""

    feature: str
    contribution: float  # signed push toward (+) or away from (-) fraud


class ScoreResponse(BaseModel):
    probability: float = Field(..., description="Calibrated fraud probability [0,1]")
    decision: str = Field(..., description="'fraud' or 'legit' (back-compat)")
    action: DecisionAction = Field(default=DecisionAction.approve,
                                   description="approve / decline / review")
    engine: str = Field(default="threshold", description="Which decision engine decided")
    threshold: float = Field(..., description="Frozen operating threshold applied")
    amount: float
    amount_at_risk: float = Field(..., description="Amount flagged as fraud, else 0")
    reason_codes: List[ReasonCode] = Field(default_factory=list)
    latency_ms: float
    model_version: str


class BatchScoreRequest(BaseModel):
    transactions: List[Transaction]


class BatchScoreResponse(BaseModel):
    results: List[ScoreResponse]
    count: int
    total_latency_ms: float
    throughput_tps: float
