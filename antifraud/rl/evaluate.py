"""Evaluation harness + full comparison (plan R2).

Runs every decision method GREEDILY over a deterministic full pass of the test
split and reports money and detection metrics on one common cost basis:

  XGBoost+F1 threshold | XGBoost+cost threshold | LinUCB bandit |
  DQN | Double DQN | Dueling DQN

Outputs a table (stdout) and comparison plots to plots/rl/. Being honest: this is
where we see whether the sequential/budget structure makes DQN beat the bandit and
the static thresholds — or not.

Run:  python -m antifraud.rl.evaluate --episodes 40 --fraction 1.0
"""
from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from antifraud.common.config import RANDOM_SEED
from antifraud.registry.artifact import load_bundle
from antifraud.rl.agents.bandit import LinUCBAgent
from antifraud.rl.env import APPROVE, DECLINE, EnvCosts, FraudDecisionEnv
from antifraud.rl.train_agent import train

PLOTS_DIR = "plots/rl"


def make_test_env(fraction: float, lean_state: bool = False) -> FraudDecisionEnv:
    # deterministic full pass over the test split (no oversampling)
    return FraudDecisionEnv(split="test", fraction=fraction, costs=EnvCosts(),
                            episode_length=None, oversample_fraud=False,
                            lean_state=lean_state, seed=RANDOM_SEED)


def eval_policy(env: FraudDecisionEnv, policy_fn) -> dict:
    obs, _ = env.reset()
    c = env.costs
    done = False
    TP = FP = FN = TN = 0
    amount_saved = amount_lost = 0.0
    reviews = false_declines = 0
    while not done:
        a = policy_fn(obs)
        obs, _, term, trunc, info = env.step(a)
        done = term or trunc
        label, amount, outcome = info["label"], info["amount"], info["outcome"]
        flagged = a != APPROVE
        if label == 1 and flagged:
            TP += 1; amount_saved += amount
        elif label == 0 and flagged:
            FP += 1
        elif label == 1 and not flagged:
            FN += 1; amount_lost += amount
        else:
            TN += 1
        if outcome in ("reviewed", "over_review"):
            reviews += 1
        if outcome == "false_decline":
            false_declines += 1
    precision = TP / (TP + FP) if TP + FP else 0.0
    recall = TP / (TP + FN) if TP + FN else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    friction = false_declines * c.false_decline_cost + reviews * c.review_cost
    total_cost = amount_lost + friction
    return dict(precision=precision, recall=recall, f1=f1,
                amount_saved=amount_saved, money_lost_to_fraud=amount_lost,
                friction_cost=friction, total_cost=total_cost,
                reviews=reviews, false_declines=false_declines,
                TP=TP, FP=FP, FN=FN)


def proba_idx(env: FraudDecisionEnv) -> int:
    return env.proba_index  # where the XGBoost score sits in the observation (lean or full)


def train_linucb(fraction: float, episodes: int = 3) -> LinUCBAgent:
    env = FraudDecisionEnv(split="train", fraction=fraction, episode_length=4000,
                           oversample_fraud=True, seed=RANDOM_SEED)
    agent = LinUCBAgent(env.action_space.n, env.observation_space.shape[0], alpha=1.0)
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            a = agent.act(obs, explore=True)
            nobs, r, term, trunc, _ = env.step(a)
            agent.update(obs, a, r / 100.0)   # scale keeps the linear system well-conditioned
            obs = nobs
            done = term or trunc
    return agent


def run(fraction: float, episodes: int) -> dict:
    bundle = load_bundle("latest")
    thr = bundle.thresholds
    env = make_test_env(fraction)
    pidx = proba_idx(env)

    results, curves = {}, {}

    # --- static XGBoost threshold baselines -----------------------------
    results["XGBoost+F1"] = eval_policy(
        env, lambda o: DECLINE if o[pidx] >= thr["f1"] else APPROVE)
    results["XGBoost+cost"] = eval_policy(
        env, lambda o: DECLINE if o[pidx] >= thr["cost"] else APPROVE)

    # --- LinUCB bandit baseline -----------------------------------------
    print("[eval] training LinUCB bandit ...")
    bandit = train_linucb(fraction)
    results["LinUCB"] = eval_policy(env, bandit.act_greedy)

    # --- DQN family ------------------------------------------------------
    for kind, label in [("dqn", "DQN"), ("double", "Double DQN"), ("dueling", "Dueling DQN")]:
        print(f"[eval] training {label} ({episodes} episodes) ...")
        _, hist = train(kind, episodes=episodes, fraction=fraction,
                        episode_length=4000, verbose=False)
        curves[label] = [h["reward"] for h in hist]
        from antifraud.rl.policy_registry import load_policy
        agent, _, _ = load_policy("latest")
        results[label] = eval_policy(env, agent.act_greedy)

    _print_table(results)
    _plots(results, curves)
    return results


def _print_table(results: dict) -> None:
    cols = ["precision", "recall", "f1", "money_lost_to_fraud", "friction_cost",
            "total_cost", "reviews", "false_declines"]
    print("\n" + "=" * 108)
    print(f"{'method':<16}" + "".join(f"{c[:12]:>13}" for c in cols))
    print("-" * 108)
    for name, m in results.items():
        row = f"{name:<16}"
        for c in cols:
            v = m[c]
            row += f"{v:>13.3f}" if isinstance(v, float) and c in ("precision", "recall", "f1") \
                else f"{v:>13.0f}"
        print(row)
    best = min(results.items(), key=lambda kv: kv[1]["total_cost"])
    print("-" * 108)
    print(f"lowest total cost: {best[0]}  (£{best[1]['total_cost']:,.0f})")
    print("=" * 108 + "\n")


def _plots(results: dict, curves: dict) -> None:
    import os
    os.makedirs(PLOTS_DIR, exist_ok=True)
    names = list(results.keys())

    # 1) total cost (lower is better)
    plt.figure(figsize=(9, 5))
    costs = [results[n]["total_cost"] for n in names]
    colors = ["#898781" if n.startswith("XGBoost") else "#3987e5" if n == "LinUCB" else "#e05555"
              for n in names]
    plt.bar(names, costs, color=colors)
    plt.ylabel("Total cost £ (fraud lost + friction) — lower is better")
    plt.title("Decision policy comparison — net cost on test set")
    plt.xticks(rotation=20, ha="right"); plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/policy_total_cost.png", dpi=120); plt.close()

    # 2) precision / recall grouped
    plt.figure(figsize=(9, 5))
    x = np.arange(len(names)); w = 0.38
    plt.bar(x - w/2, [results[n]["precision"] for n in names], w, label="precision", color="#3987e5")
    plt.bar(x + w/2, [results[n]["recall"] for n in names], w, label="recall", color="#1baf7a")
    plt.xticks(x, names, rotation=20, ha="right"); plt.ylim(0, 1); plt.legend()
    plt.title("Precision vs recall by decision policy"); plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/policy_precision_recall.png", dpi=120); plt.close()

    # 3) DQN learning curves
    if curves:
        plt.figure(figsize=(9, 5))
        for label, hist in curves.items():
            plt.plot(range(1, len(hist) + 1), hist, label=label)
        plt.xlabel("episode"); plt.ylabel("episode reward")
        plt.title("DQN learning curves (training)"); plt.legend(); plt.tight_layout()
        plt.savefig(f"{PLOTS_DIR}/dqn_learning_curves.png", dpi=120); plt.close()
    print(f"[eval] plots saved to {PLOTS_DIR}/")


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate & compare fraud-decision policies.")
    ap.add_argument("--fraction", type=float, default=1.0)
    ap.add_argument("--episodes", type=int, default=40)
    args = ap.parse_args()
    run(args.fraction, args.episodes)


if __name__ == "__main__":
    main()
