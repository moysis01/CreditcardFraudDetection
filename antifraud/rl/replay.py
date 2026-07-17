"""Experience replay with fraud-aware sampling (plan R1).

Extreme class imbalance means naive uniform replay rarely revisits the rare,
high-magnitude fraud transitions. We bias sampling toward transitions with large
|reward| (the fraud-relevant ones) via lightweight priority sampling.
"""
from __future__ import annotations

from collections import deque
from typing import Tuple

import numpy as np


class ReplayBuffer:
    def __init__(self, capacity: int = 100_000, alpha: float = 0.6, seed: int = 25):
        self.capacity = capacity
        self.alpha = alpha              # 0 = uniform, 1 = fully |reward|-proportional
        self._rng = np.random.default_rng(seed)
        self.s = deque(maxlen=capacity)
        self.a = deque(maxlen=capacity)
        self.r = deque(maxlen=capacity)
        self.ns = deque(maxlen=capacity)
        self.d = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self.s)

    def push(self, state, action, reward, next_state, done) -> None:
        self.s.append(np.asarray(state, np.float32))
        self.a.append(int(action))
        self.r.append(float(reward))
        self.ns.append(np.asarray(next_state, np.float32))
        self.d.append(bool(done))

    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        n = len(self.s)
        rewards = np.abs(np.asarray(self.r, dtype=np.float64)) + 1e-3
        probs = rewards ** self.alpha
        probs /= probs.sum()
        idx = self._rng.choice(n, size=batch_size, p=probs)
        return (
            np.stack([self.s[i] for i in idx]),
            np.asarray([self.a[i] for i in idx], dtype=np.int64),
            np.asarray([self.r[i] for i in idx], dtype=np.float32),
            np.stack([self.ns[i] for i in idx]),
            np.asarray([self.d[i] for i in idx], dtype=np.float32),
        )
