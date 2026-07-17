"""Policy registry (plan R1) — versioned RL policy bundles, mirroring
antifraud.registry.artifact for the supervised model.

    models/policies/<version>/
        policy.pt       torch state_dict of the Q-network
        config.json     agent hyperparameters + state/action dims
        metadata.json   which base model it wraps, env costs, eval metrics, git sha
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Tuple

import torch

from antifraud.common.config import MODELS_DIR
from antifraud.registry.artifact import _git_sha
from antifraud.rl.agents.dqn import DQNAgent

POLICIES_DIR = MODELS_DIR / "policies"


def new_version(tag: str = "dqn") -> str:
    return datetime.now(timezone.utc).strftime(f"{tag}-%Y%m%d-%H%M%S")


def save_policy(version: str, agent: DQNAgent, config: Dict[str, Any],
                metadata: Dict[str, Any]) -> Path:
    out = POLICIES_DIR / version
    out.mkdir(parents=True, exist_ok=True)
    torch.save(agent.state_dict(), out / "policy.pt")
    (out / "config.json").write_text(json.dumps(config, indent=2))
    meta = {**metadata, "version": version, "git_sha": _git_sha(),
            "created_utc": datetime.now(timezone.utc).isoformat()}
    (out / "metadata.json").write_text(json.dumps(meta, indent=2, default=str))
    return out


def resolve_version(version: str) -> str:
    if version and version != "latest":
        if not (POLICIES_DIR / version).is_dir():
            raise FileNotFoundError(f"Policy not found: {POLICIES_DIR / version}")
        return version
    # Sort by modification time (creation order) — version tags differ by agent kind
    # (dqn/double/dueling), so lexical sort would not track recency. Only consider
    # complete bundles (those with config.json).
    cands = [p for p in POLICIES_DIR.glob("*") if p.is_dir() and (p / "config.json").exists()]
    if not cands:
        raise FileNotFoundError(
            f"No policies in {POLICIES_DIR}. Run: python -m antifraud.rl.train_agent")
    return max(cands, key=lambda p: p.stat().st_mtime).name


def load_policy(version: str = "latest") -> Tuple[DQNAgent, Dict[str, Any], Dict[str, Any]]:
    resolved = resolve_version(version)
    d = POLICIES_DIR / resolved
    config = json.loads((d / "config.json").read_text())
    metadata = json.loads((d / "metadata.json").read_text())
    agent = DQNAgent(config["state_dim"], config["n_actions"],
                     double=config.get("double", False),
                     dueling=config.get("dueling", False),
                     hidden=config.get("hidden", 128))
    agent.load_state_dict(torch.load(d / "policy.pt", map_location="cpu"))
    metadata["version"] = resolved
    return agent, config, metadata
