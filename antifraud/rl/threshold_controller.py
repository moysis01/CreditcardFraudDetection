"""Adaptive-threshold RL controller (plan R-D, secondary experiment).

Instead of deciding per transaction, this agent controls the *operating threshold*
of the existing XGBoost model over time. It is the meta-controller framing: as the
fraud rate drifts, a single static threshold becomes stale — an RL agent that raises
/ holds / lowers the threshold in response to recent performance can track the drift.

  state  = [current_threshold, rolling_fraud_rate, rolling_precision, rolling_flag_rate]
  action = {lower, hold, raise} threshold by a fixed step
  reward = −(net cost of the batch processed at the current threshold)

We inject drift (fraud rate rises then falls across the stream) so there is
something to adapt to, and compare against the best fixed threshold.

Run:  python -m antifraud.rl.threshold_controller --episodes 30
"""
from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from antifraud.common.config import FALSE_DECLINE_COST, RANDOM_SEED
from antifraud.registry.artifact import load_bundle
from antifraud.common.schemas import FEATURE_ORDER
from antifraud.rl.agents.dqn import DQNAgent

STEP = 0.05  # threshold move per action


def _load_proba(split: str, fraction: float):
    from antifraud.training.build_artifact import load_frame, split as split_frame, xy
    df = load_frame(fraction)
    tr, va, te = split_frame(df, "random")
    frame = {"train": tr, "val": va, "test": te}[split]
    X, y = xy(frame)
    bundle = load_bundle("latest")
    import xgboost as xgb
    proba = bundle.booster.predict(
        xgb.DMatrix(bundle.preprocessor.transform(X[FEATURE_ORDER]),
                    feature_names=bundle.feature_order))
    return proba.astype(np.float32), y.astype(int), frame["Amount"].to_numpy(float)


def _drift_stream(proba, y, amounts, rng, segments=6, base_frac=0.4):
    """Build an index stream whose fraud proportion rises then falls (injected drift)."""
    fraud = np.where(y == 1)[0]
    legit = np.where(y == 0)[0]
    seg_len = 3000
    profile = np.concatenate([np.linspace(0.2, 3.0, segments // 2),
                              np.linspace(3.0, 0.2, segments - segments // 2)])
    stream = []
    for mult in profile:
        n_fraud = min(len(fraud), int(seg_len * 0.005 * mult))
        idx = np.concatenate([rng.choice(fraud, n_fraud, replace=True),
                              rng.choice(legit, seg_len - n_fraud, replace=True)])
        rng.shuffle(idx)
        stream.append(idx)
    return stream  # list of segments (each an index array)


def _batch_cost(proba, y, amounts, idx, threshold):
    pred = (proba[idx] >= threshold).astype(int)
    label = y[idx]
    fn = (label == 1) & (pred == 0)
    fp = (label == 0) & (pred == 1)
    tp = (label == 1) & (pred == 1)
    friction = fp.sum() * FALSE_DECLINE_COST
    cost = amounts[idx][fn].sum() + friction
    precision = tp.sum() / max((pred == 1).sum(), 1)
    fraud_rate = label.mean()
    flag_rate = pred.mean()
    return float(cost), float(precision), float(fraud_rate), float(flag_rate)


class ThresholdControlEnv:
    def __init__(self, proba, y, amounts, seed=RANDOM_SEED):
        self.proba, self.y, self.amounts = proba, y, amounts
        self.rng = np.random.default_rng(seed)

    def reset(self):
        self.stream = _drift_stream(self.proba, self.y, self.amounts, self.rng)
        self.seg = 0
        self.threshold = 0.5
        self._pr, self._fr, self._fl = 0.0, 0.0, 0.0
        return np.array([self.threshold, self._fr, self._pr, self._fl], np.float32)

    def step(self, action):
        if action == 0: self.threshold = max(0.01, self.threshold - STEP)
        elif action == 2: self.threshold = min(0.99, self.threshold + STEP)
        idx = self.stream[self.seg]
        cost, pr, fr, fl = _batch_cost(self.proba, self.y, self.amounts, idx, self.threshold)
        self._pr, self._fr, self._fl = pr, fr, fl
        self.seg += 1
        done = self.seg >= len(self.stream)
        reward = -cost / 100.0
        obs = np.array([self.threshold, fr, pr, fl], np.float32)
        return obs, reward, done, {"cost": cost, "threshold": self.threshold, "fraud_rate": fr}


def train_and_compare(episodes: int, fraction: float):
    proba, y, amounts = _load_proba("test", fraction)
    env = ThresholdControlEnv(proba, y, amounts)
    agent = DQNAgent(4, 3, double=True, dueling=True, lr=1e-3, gamma=0.9, seed=RANDOM_SEED)
    from antifraud.rl.replay import ReplayBuffer
    buf = ReplayBuffer(20000)
    eps = 1.0
    for ep in range(1, episodes + 1):
        obs = env.reset(); done = False; total = 0.0
        while not done:
            a = agent.act(obs, eps)
            nobs, r, done, _ = env.step(a)
            buf.push(obs, a, r, nobs, done); obs = nobs; total += r
            if len(buf) >= 200:
                agent.update(buf.sample(64))
        agent.sync_target()
        eps = max(0.05, eps * 0.92)
    # ---- evaluate: adaptive vs best static threshold on a fresh drift stream ----
    obs = env.reset(); done = False
    adaptive_cost = 0.0; thr_trace = []; fraud_trace = []
    while not done:
        a = agent.act_greedy(obs)
        obs, r, done, info = env.step(a)
        adaptive_cost += info["cost"]; thr_trace.append(info["threshold"]); fraud_trace.append(info["fraud_rate"])
    # best fixed threshold on the same stream
    stream = env.stream
    static = {}
    for thr in np.linspace(0.1, 0.9, 9):
        c = sum(_batch_cost(proba, y, amounts, idx, thr)[0] for idx in stream)
        static[round(float(thr), 2)] = c
    best_static_thr = min(static, key=static.get)
    best_static_cost = static[best_static_thr]

    print(f"\n[threshold-controller] adaptive total cost: £{adaptive_cost:,.0f}")
    print(f"[threshold-controller] best static ({best_static_thr}) cost: £{best_static_cost:,.0f}")
    verdict = "adaptive wins" if adaptive_cost < best_static_cost else "static wins"
    print(f"[threshold-controller] {verdict}")

    import os; os.makedirs("plots/rl", exist_ok=True)
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(thr_trace, color="#3987e5", label="RL threshold")
    ax1.set_ylabel("threshold", color="#3987e5"); ax1.set_ylim(0, 1)
    ax2 = ax1.twinx()
    ax2.plot(fraud_trace, color="#e05555", alpha=0.6, label="fraud rate (drift)")
    ax2.set_ylabel("fraud rate", color="#e05555")
    ax1.axhline(best_static_thr, ls="--", color="#898781", label=f"best static ({best_static_thr})")
    ax1.set_xlabel("stream segment"); ax1.set_title("RL adaptive threshold tracking injected fraud-rate drift")
    fig.legend(loc="upper right"); fig.tight_layout()
    fig.savefig("plots/rl/adaptive_threshold.png", dpi=120); plt.close()
    print("[threshold-controller] plot -> plots/rl/adaptive_threshold.png")
    return adaptive_cost, best_static_cost


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--fraction", type=float, default=1.0)
    args = ap.parse_args()
    train_and_compare(args.episodes, args.fraction)


if __name__ == "__main__":
    main()
