"""Train ONE strong DQN policy and immediately benchmark it against the XGBoost
threshold baselines (plan R2 follow-up).

Used to close the gap found in the first comparison: the agent was over-declining.
This trains with the improved hyperparameters (low gamma, reward scaling, realistic
fraud fraction, soft target) and reports whether it now matches XGBoost on net cost.

Run:  python -m antifraud.rl.tune --agent dueling --episodes 80 --fraction 1.0
"""
from __future__ import annotations

import argparse

from antifraud.registry.artifact import load_bundle
from antifraud.rl.env import APPROVE, DECLINE
from antifraud.rl.evaluate import eval_policy, make_test_env, proba_idx
from antifraud.rl.policy_registry import load_policy
from antifraud.rl.train_agent import train


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", default="dueling")
    ap.add_argument("--episodes", type=int, default=80)
    ap.add_argument("--fraction", type=float, default=1.0)
    ap.add_argument("--episode-length", type=int, default=6000)
    ap.add_argument("--gamma", type=float, default=0.5)
    ap.add_argument("--fraud-frac", type=float, default=0.15)
    ap.add_argument("--reward-scale", type=float, default=0.01)
    ap.add_argument("--lean", action="store_true", default=True,
                    help="Use the lean [score, amount, budget, fraud_rate] state (default).")
    ap.add_argument("--full-state", dest="lean", action="store_false",
                    help="Use the full 32-d state instead of the lean one.")
    args = ap.parse_args()

    print(f"[tune] training {args.agent}: episodes={args.episodes} fraction={args.fraction} "
          f"gamma={args.gamma} fraud_frac={args.fraud_frac} lean_state={args.lean}")
    train(args.agent, episodes=args.episodes, fraction=args.fraction,
          episode_length=args.episode_length, gamma=args.gamma,
          fraud_frac=args.fraud_frac, reward_scale=args.reward_scale,
          lean_state=args.lean, verbose=True)

    # ---- benchmark the freshly trained policy vs the XGBoost thresholds ----
    thr = load_bundle("latest").thresholds
    env = make_test_env(args.fraction, lean_state=args.lean)
    pidx = proba_idx(env)
    agent, _, meta = load_policy("latest")

    results = {
        "XGBoost+F1":   eval_policy(env, lambda o: DECLINE if o[pidx] >= thr["f1"] else APPROVE),
        "XGBoost+cost": eval_policy(env, lambda o: DECLINE if o[pidx] >= thr["cost"] else APPROVE),
        f"{args.agent} DQN": eval_policy(env, agent.act_greedy),
    }

    cols = ["precision", "recall", "f1", "money_lost_to_fraud", "friction_cost",
            "total_cost", "reviews", "false_declines"]
    print("\n" + "=" * 108)
    print(f"{'method':<16}" + "".join(f"{c[:12]:>13}" for c in cols))
    print("-" * 108)
    for name, m in results.items():
        row = f"{name:<16}"
        for c in cols:
            v = m[c]
            row += f"{v:>13.3f}" if c in ("precision", "recall", "f1") else f"{v:>13.0f}"
        print(row)
    best = min(results.items(), key=lambda kv: kv[1]["total_cost"])
    xgb_cost = min(results["XGBoost+F1"]["total_cost"], results["XGBoost+cost"]["total_cost"])
    dqn_cost = results[f"{args.agent} DQN"]["total_cost"]
    print("-" * 108)
    print(f"best: {best[0]} (£{best[1]['total_cost']:,.0f}) | "
          f"DQN £{dqn_cost:,.0f} vs XGBoost £{xgb_cost:,.0f} | "
          f"DQN is {'competitive' if dqn_cost <= xgb_cost * 1.5 else 'still behind'}")
    print("=" * 108)


if __name__ == "__main__":
    main()
