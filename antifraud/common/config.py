"""Central configuration and paths for the anti-fraud platform.

Kept dependency-free (pure stdlib) so it can be imported from anywhere —
training, serving, streaming, and tests — without side effects.
"""
from __future__ import annotations

import os
from pathlib import Path

# Repo root = two levels up from this file (antifraud/common/config.py).
REPO_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = Path(os.environ.get("AF_DATA_PATH", REPO_ROOT / "creditcard.csv"))
MODELS_DIR = Path(os.environ.get("AF_MODELS_DIR", REPO_ROOT / "models"))

# Which artifact version the serving layer loads. "latest" resolves to the
# most recently created bundle in MODELS_DIR (see registry.artifact.resolve_version).
SERVED_VERSION = os.environ.get("AF_MODEL_VERSION", "latest")

# Reproducibility
RANDOM_SEED = 25

# Cost model (see common/cost_model.py). A false negative costs the transaction
# amount (fraud goes through); a false positive costs a fixed manual-review fee.
REVIEW_COST = float(os.environ.get("AF_REVIEW_COST", 3.0))
# £ friction cost of wrongly declining a legit transaction (lost sale + annoyance).
FALSE_DECLINE_COST = float(os.environ.get("AF_FALSE_DECLINE_COST", 5.0))

# Number of SHAP reason codes returned for a flagged transaction.
TOP_REASON_CODES = int(os.environ.get("AF_TOP_REASON_CODES", 5))
