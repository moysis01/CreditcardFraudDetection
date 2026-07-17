"""Fraud-decision environments (plan Part R-A).

Offline RL: we cannot interact with a live bank, so we build a simulator from the
labelled dataset. Each step presents one transaction; the agent chooses an action;
the reward comes from the cost model (money saved/lost). A *shared review budget*
makes REVIEW actions have downstream consequences — that is what turns this from a
one-shot bandit problem into a genuine MDP and justifies Deep Q-Learning.

Two environments:
  FraudDecisionEnv  — flagship, 3 actions {APPROVE, DECLINE, REVIEW}, budget dynamics.
  ImbalancedClassEnv — Lin et al. (2020) baseline: 2 actions, ±reward, episode ends
                       on the first missed fraud.

The XGBoost model is put *in the loop*: its fraud probability is part of the state,
so the RL agent learns to act on top of the existing model rather than replace it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from antifraud.common.config import DATA_PATH, RANDOM_SEED, REVIEW_COST
from antifraud.common.schemas import FEATURE_ORDER
from antifraud.registry.artifact import load_bundle

# Actions
APPROVE, DECLINE, REVIEW = 0, 1, 2
ACTION_NAMES = {APPROVE: "approve", DECLINE: "decline", REVIEW: "review"}


@dataclass
class EnvCosts:
    review_cost: float = REVIEW_COST      # £ to send a transaction to human review
    false_decline_cost: float = 5.0       # £ friction of wrongly declining a legit txn
    approve_legit_reward: float = 0.0     # neutral: approving legit is the happy path
    review_budget_frac: float = 0.02      # fraction of a stream that may be reviewed
    over_review_penalty: float = 10.0     # penalty for REVIEW once the budget is spent


def _load_split(split: str, fraction: float) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Load one split's features/labels/amounts, using the SAME split logic as
    training (antifraud.training.build_artifact) so eval matches the model's test set."""
    from antifraud.training.build_artifact import load_frame, split as split_frame, xy
    df = load_frame(fraction)
    train, val, test = split_frame(df, "random")
    frame = {"train": train, "val": val, "test": test}[split]
    X, y = xy(frame)
    return X, y.astype(int), frame["Amount"].to_numpy(dtype=float)


class FraudDecisionEnv(gym.Env):
    """Flagship 3-action fraud environment with a shared review budget."""

    metadata = {"render_modes": []}

    def __init__(self, split: str = "train", fraction: float = 1.0,
                 costs: Optional[EnvCosts] = None, episode_length: Optional[int] = 4000,
                 oversample_fraud: bool = True, fraud_frac: float = 0.5,
                 lean_state: bool = False, seed: int = RANDOM_SEED,
                 bundle_version: str = "latest"):
        super().__init__()
        self.costs = costs or EnvCosts()
        self.episode_length = episode_length
        self.oversample_fraud = oversample_fraud
        self.fraud_frac = fraud_frac    # fraction of oversampled stream that is fraud
        # Lean state = [risk score, scaled amount, budget left, fraud rate]. Forces the
        # agent to decide from the model's calibrated score + business context, rather
        # than the 28 noisy PCA components (which the score already summarises).
        self.lean_state = lean_state
        self._rng = np.random.default_rng(seed)

        # Load data and put the XGBoost model in the loop (batch precompute — fast).
        X, self.y, self.amounts = _load_split(split, fraction)
        bundle = load_bundle(bundle_version)
        self.feature_order = bundle.feature_order
        self.X_scaled = bundle.preprocessor.transform(X[FEATURE_ORDER]).astype(np.float32)
        import xgboost as xgb
        dmatrix = xgb.DMatrix(self.X_scaled, feature_names=self.feature_order)
        self.proba = bundle.booster.predict(dmatrix).astype(np.float32)

        self.n = len(self.y)
        self.fraud_idx = np.where(self.y == 1)[0]
        self.legit_idx = np.where(self.y == 0)[0]

        # State = 29 scaled features + [xgb proba, budget frac left, rolling fraud rate],
        # OR the lean 4-d state [proba, scaled amount, budget frac, fraud rate].
        n_features = self.X_scaled.shape[1]
        obs_dim = 4 if self.lean_state else n_features + 3
        self.proba_index = 0 if self.lean_state else n_features  # where the score sits in obs
        self._amount_col = n_features - 1                        # Amount is the last feature
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)

        self._order: np.ndarray = np.array([], dtype=int)
        self._t = 0
        self._budget = 0
        self._recent = []  # rolling window of recent true labels

    # ---- episode plumbing ------------------------------------------------
    def _make_order(self) -> np.ndarray:
        length = self.episode_length or self.n
        if self.oversample_fraud and len(self.fraud_idx) > 0:
            # Oversample fraud so the agent sees enough positives, but keep the rate
            # closer to reality than 50/50 to avoid an over-declining bias.
            n_fraud = int(length * self.fraud_frac)
            frauds = self._rng.choice(self.fraud_idx, size=n_fraud, replace=True)
            legits = self._rng.choice(self.legit_idx, size=length - n_fraud, replace=True)
            order = np.concatenate([frauds, legits])
            self._rng.shuffle(order)
            return order
        # deterministic full pass (evaluation)
        return np.arange(self.n)

    def _obs(self) -> np.ndarray:
        i = self._order[self._t]
        budget_frac = self._budget / max(self._budget_start, 1)
        fraud_rate = float(np.mean(self._recent)) if self._recent else 0.0
        if self.lean_state:
            return np.array([self.proba[i], self.X_scaled[i][self._amount_col],
                             budget_frac, fraud_rate], dtype=np.float32)
        return np.concatenate([self.X_scaled[i], [self.proba[i], budget_frac, fraud_rate]]).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._order = self._make_order()
        self._t = 0
        self._budget_start = max(int(len(self._order) * self.costs.review_budget_frac), 1)
        self._budget = self._budget_start
        self._recent = []
        return self._obs(), {}

    # ---- the reward model ------------------------------------------------
    def _reward(self, action: int, label: int, amount: float) -> Tuple[float, str]:
        c = self.costs
        if action == APPROVE:
            return (-amount, "missed_fraud") if label == 1 else (c.approve_legit_reward, "approved_legit")
        if action == DECLINE:
            return (amount, "caught_fraud") if label == 1 else (-c.false_decline_cost, "false_decline")
        # REVIEW
        if self._budget <= 0:
            # Budget spent: forced approval + penalty — the downstream cost of over-reviewing.
            base = -amount if label == 1 else 0.0
            return base - c.over_review_penalty, "over_review"
        self._budget -= 1
        # Human review resolves correctly; we pay the review fee either way.
        return (amount - c.review_cost if label == 1 else -c.review_cost), "reviewed"

    def step(self, action: int):
        i = self._order[self._t]
        label, amount = int(self.y[i]), float(self.amounts[i])
        reward, outcome = self._reward(action, label, amount)

        self._recent.append(label)
        if len(self._recent) > 200:
            self._recent.pop(0)

        self._t += 1
        terminated = False
        truncated = self._t >= len(self._order)
        obs = self._obs() if not truncated else np.zeros(self.observation_space.shape, np.float32)
        info = {"label": label, "amount": amount, "outcome": outcome,
                "action": ACTION_NAMES[action], "proba": float(self.proba[i])}
        return obs, float(reward), terminated, truncated, info


class ImbalancedClassEnv(gym.Env):
    """Lin et al. (2020) imbalanced-classification-as-MDP baseline.

    2 actions {predict legit, predict fraud}; correct fraud rewarded more than
    correct legit; the episode terminates the moment a fraud is misclassified.
    """

    metadata = {"render_modes": []}

    def __init__(self, split: str = "train", fraction: float = 1.0, seed: int = RANDOM_SEED,
                 minority_reward: float = 1.0, majority_reward: float = 0.1,
                 episode_length: Optional[int] = 4000, bundle_version: str = "latest"):
        super().__init__()
        self._rng = np.random.default_rng(seed)
        X, self.y, self.amounts = _load_split(split, fraction)
        bundle = load_bundle(bundle_version)
        self.X_scaled = bundle.preprocessor.transform(X[FEATURE_ORDER]).astype(np.float32)
        self.minority_reward, self.majority_reward = minority_reward, majority_reward
        self.episode_length = episode_length
        self.n = len(self.y)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(self.X_scaled.shape[1],), dtype=np.float32)
        self.action_space = spaces.Discrete(2)
        self._order = np.array([], dtype=int)
        self._t = 0

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        length = self.episode_length or self.n
        self._order = self._rng.permutation(self.n)[:length]
        self._t = 0
        return self.X_scaled[self._order[0]], {}

    def step(self, action: int):
        i = self._order[self._t]
        label = int(self.y[i])
        correct = (action == label)
        if label == 1:  # minority
            reward = self.minority_reward if correct else -self.minority_reward
            terminated = not correct  # episode ends on a missed fraud
        else:
            reward = self.majority_reward if correct else -self.majority_reward
            terminated = False
        self._t += 1
        truncated = self._t >= len(self._order)
        done = terminated or truncated
        obs = self.X_scaled[self._order[self._t]] if not done else np.zeros(self.observation_space.shape, np.float32)
        return obs, float(reward), terminated, truncated, {"label": label, "correct": correct}
