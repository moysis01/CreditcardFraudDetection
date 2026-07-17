"""Deep Q-Network agents (plan R1) — one class, three variants via flags.

  vanilla  : DQN (Mnih et al. 2015)
  double   : Double DQN (van Hasselt 2016) — decouple action selection/evaluation
  dueling  : Dueling DQN (Wang et al. 2016) — separate value/advantage streams

The Q-network is a small MLP, so greedy inference is sub-millisecond on CPU —
it keeps the serving hot path within the platform's <100 ms latency budget.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    def __init__(self, state_dim: int, n_actions: int, hidden: int = 128, dueling: bool = False):
        super().__init__()
        self.dueling = dueling
        self.body = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        if dueling:
            self.value = nn.Linear(hidden, 1)
            self.advantage = nn.Linear(hidden, n_actions)
        else:
            self.head = nn.Linear(hidden, n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.body(x)
        if self.dueling:
            v = self.value(h)
            a = self.advantage(h)
            return v + a - a.mean(dim=1, keepdim=True)
        return self.head(h)


class DQNAgent:
    def __init__(self, state_dim: int, n_actions: int, *, double: bool = False,
                 dueling: bool = False, lr: float = 1e-3, gamma: float = 0.95,
                 hidden: int = 128, seed: int = 25, device: str = "cpu"):
        torch.manual_seed(seed)
        self.state_dim, self.n_actions = state_dim, n_actions
        self.double, self.dueling, self.gamma = double, dueling, gamma
        self.device = torch.device(device)
        self.online = QNetwork(state_dim, n_actions, hidden, dueling).to(self.device)
        self.target = QNetwork(state_dim, n_actions, hidden, dueling).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        self.opt = torch.optim.Adam(self.online.parameters(), lr=lr)
        self._rng = np.random.default_rng(seed)

    # ---- action selection ------------------------------------------------
    @torch.no_grad()
    def act(self, state: np.ndarray, epsilon: float) -> int:
        if self._rng.random() < epsilon:
            return int(self._rng.integers(self.n_actions))
        return self.act_greedy(state)

    @torch.no_grad()
    def act_greedy(self, state: np.ndarray) -> int:
        t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        return int(self.online(t).argmax(dim=1).item())

    @torch.no_grad()
    def q_values(self, state: np.ndarray) -> np.ndarray:
        t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        return self.online(t).cpu().numpy()[0]

    # ---- learning --------------------------------------------------------
    def update(self, batch: Tuple[np.ndarray, ...]) -> float:
        s, a, r, ns, d = batch
        s = torch.as_tensor(s, dtype=torch.float32, device=self.device)
        a = torch.as_tensor(a, dtype=torch.int64, device=self.device).unsqueeze(1)
        r = torch.as_tensor(r, dtype=torch.float32, device=self.device).unsqueeze(1)
        ns = torch.as_tensor(ns, dtype=torch.float32, device=self.device)
        d = torch.as_tensor(d, dtype=torch.float32, device=self.device).unsqueeze(1)

        q = self.online(s).gather(1, a)
        with torch.no_grad():
            if self.double:
                next_a = self.online(ns).argmax(dim=1, keepdim=True)
                next_q = self.target(ns).gather(1, next_a)
            else:
                next_q = self.target(ns).max(dim=1, keepdim=True).values
            target = r + self.gamma * next_q * (1.0 - d)

        loss = F.smooth_l1_loss(q, target)   # Huber loss for stability
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)
        self.opt.step()
        return float(loss.item())

    def sync_target(self) -> None:
        self.target.load_state_dict(self.online.state_dict())

    def soft_update(self, tau: float = 0.01) -> None:
        """Polyak averaging: slowly track the online net for stable targets."""
        for tp, op in zip(self.target.parameters(), self.online.parameters()):
            tp.data.mul_(1.0 - tau).add_(tau * op.data)

    # ---- persistence -----------------------------------------------------
    def state_dict(self) -> dict:
        return self.online.state_dict()

    def load_state_dict(self, sd: dict) -> None:
        self.online.load_state_dict(sd)
        self.target.load_state_dict(sd)
        self.online.eval()
