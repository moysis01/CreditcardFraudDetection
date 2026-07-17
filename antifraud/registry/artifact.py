"""Model registry — versioned artifact bundles (plan A14).

A bundle in models/<version>/ is the single source of truth shared by training
and serving:

    model.json          native XGBoost booster (fast to load, fast to predict)
    preprocessor.joblib fitted ColumnTransformer (scalers) — no train/serve skew
    threshold.json      frozen operating thresholds (f1 + cost) chosen on validation
    metadata.json       metrics, feature order, seed, git sha, timestamp

Serving validates and loads this bundle once at startup.
"""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import joblib

from antifraud.common.config import MODELS_DIR

MODEL_FILE = "model.json"
PREPROCESSOR_FILE = "preprocessor.joblib"
THRESHOLD_FILE = "threshold.json"
METADATA_FILE = "metadata.json"


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=MODELS_DIR.parent,
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def new_version() -> str:
    """Timestamp-based version id, sortable lexicographically."""
    return datetime.now(timezone.utc).strftime("v%Y%m%d-%H%M%S")


def save_bundle(version: str, booster, preprocessor, thresholds: Dict[str, float],
                metadata: Dict[str, Any]) -> Path:
    """Persist a complete artifact bundle and return its directory."""
    out = MODELS_DIR / version
    out.mkdir(parents=True, exist_ok=True)

    booster.save_model(str(out / MODEL_FILE))
    joblib.dump(preprocessor, out / PREPROCESSOR_FILE)
    (out / THRESHOLD_FILE).write_text(json.dumps(thresholds, indent=2))

    metadata = {
        **metadata,
        "version": version,
        "git_sha": _git_sha(),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    (out / METADATA_FILE).write_text(json.dumps(metadata, indent=2, default=str))
    return out


def resolve_version(version: str) -> str:
    """Resolve 'latest' to the newest bundle; otherwise validate it exists."""
    if version and version != "latest":
        if not (MODELS_DIR / version).is_dir():
            raise FileNotFoundError(f"Model version not found: {MODELS_DIR / version}")
        return version
    candidates = sorted(p.name for p in MODELS_DIR.glob("v*") if p.is_dir())
    if not candidates:
        raise FileNotFoundError(
            f"No model bundles in {MODELS_DIR}. Run: python -m antifraud.training.build_artifact"
        )
    return candidates[-1]


@dataclass
class LoadedBundle:
    version: str
    booster: Any
    preprocessor: Any
    thresholds: Dict[str, float]
    metadata: Dict[str, Any]
    feature_order: List[str]


def load_bundle(version: str = "latest") -> LoadedBundle:
    """Load a bundle for serving. Imports xgboost lazily so non-serving callers
    (e.g. the pure-schema tests) don't pay the import cost."""
    import xgboost as xgb

    resolved = resolve_version(version)
    d = MODELS_DIR / resolved

    booster = xgb.Booster()
    booster.load_model(str(d / MODEL_FILE))
    preprocessor = joblib.load(d / PREPROCESSOR_FILE)
    thresholds = json.loads((d / THRESHOLD_FILE).read_text())
    metadata = json.loads((d / METADATA_FILE).read_text())

    return LoadedBundle(
        version=resolved,
        booster=booster,
        preprocessor=preprocessor,
        thresholds=thresholds,
        metadata=metadata,
        feature_order=metadata["feature_order"],
    )
