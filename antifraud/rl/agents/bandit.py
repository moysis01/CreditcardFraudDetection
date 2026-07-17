"""LinUCB contextual-bandit baseline (plan R2).

A per-transaction approve/decline/review decision is largely a *one-step* choice,
which a contextual bandit is theoretically well-suited to. LinUCB (Li et al. 2010)
is the classic linear bandit: for each action it keeps a ridge-regression estimate
of expected reward and adds an upper-confidence bonus for exploration. Including it
is what makes the DQN comparison honest — if a bandit matches DQN, the sequential
(budget) structure isn't buying much; if DQN wins, it is.
"""
from __future__ import annotations

import numpy as np


class LinUCBAgent:
    def __init__(self, n_actions: int, dim: int, alpha: float = 1.0, seed: int = 25):
        self.n_actions, self.dim, self.alpha = n_actions, dim, alpha
        self.A = [np.eye(dim) for _ in range(n_actions)]      # d×d per action
        self.b = [np.zeros(dim) for _ in range(n_actions)]
        self._rng = np.random.default_rng(seed)

    def _ucb(self, x: np.ndarray, a: int) -> float:
        theta = np.linalg.solve(self.A[a], self.b[a])
        mean = float(theta @ x)
        bonus = self.alpha * float(np.sqrt(x @ np.linalg.solve(self.A[a], x)))
        return mean + bonus

    def act(self, state: np.ndarray, explore: bool = True) -> int:
        x = np.asarray(state, dtype=np.float64)
        scores = [self._ucb(x, a) for a in range(self.n_actions)]
        if not explore:  # greedy = mean only
            scores = [float(np.linalg.solve(self.A[a], self.b[a]) @ x) for a in range(self.n_actions)]
        return int(np.argmax(scores))

    def act_greedy(self, state: np.ndarray) -> int:
        return self.act(state, explore=False)

    def update(self, state: np.ndarray, action: int, reward: float) -> None:
        x = np.asarray(state, dtype=np.float64)
        self.A[action] += np.outer(x, x)
        self.b[action] += reward * x
