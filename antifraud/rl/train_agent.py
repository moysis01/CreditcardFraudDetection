"""Offline DQN training loop (plan R1).

Trains a DQN / Double DQN / Dueling DQN on FraudDecisionEnv built from the training
split, then saves a versioned policy artifact. Offline / off-policy: the agent learns
from the logged, labelled dataset — no live environment interaction (see plan
Out-of-scope).

Run:
  python -m antifraud.rl.train_agent --agent dueling --episodes 40 --fraction 1.0
"""
from __future__ import annotations

import argparse
import random
import time

import numpy as np
import torch

from antifraud.common.config import RANDOM_SEED
from antifraud.rl.agents.dqn import DQNAgent
from antifraud.rl.env import EnvCosts, FraudDecisionEnv
from antifraud.rl.policy_registry import new_version, save_policy
from antifraud.rl.replay import ReplayBuffer

AGENT_FLAGS = {
    "dqn": dict(double=False, dueling=False),
    "double": dict(double=True, dueling=False),
    "dueling": dict(double=True, dueling=True),  # dueling + double (common pairing)
}


def set_seeds(seed: int = RANDOM_SEED) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def train(agent_kind: str, episodes: int, fraction: float, episode_length: int,
          seed: int = RANDOM_SEED, verbose: bool = True, *,
          gamma: float = 0.5, reward_scale: float = 0.01, fraud_frac: float = 0.25,
          soft_tau: float = 0.01, lr: float = 5e-4, lean_state: bool = False,
          train_freq: int = 2):
    """Train a DQN policy.

    Defaults reflect the diagnosis of the first cut: the per-transaction decision is
    nearly one-step, so a LOW gamma makes Q approximate immediate cost (stable);
    REWARD_SCALE tames the large £ targets; a reduced FRAUD_FRAC keeps the training
    base-rate closer to reality so the agent doesn't learn to over-decline; and a
    Polyak SOFT_TAU target keeps learning stable.
    """
    set_seeds(seed)
    env = FraudDecisionEnv(split="train", fraction=fraction, costs=EnvCosts(),
                           episode_length=episode_length, oversample_fraud=True,
                           fraud_frac=fraud_frac, lean_state=lean_state, seed=seed)
    state_dim = env.observation_space.shape[0]
    flags = AGENT_FLAGS[agent_kind]
    agent = DQNAgent(state_dim, env.action_space.n, lr=lr, gamma=gamma, seed=seed, **flags)
    buffer = ReplayBuffer(capacity=200_000, seed=seed)

    eps, eps_min, eps_decay = 1.0, 0.02, 0.94
    batch_size, warmup = 256, 2000
    history = []
    t0 = time.time()

    for ep in range(1, episodes + 1):
        obs, _ = env.reset()
        done = False
        ep_reward, ep_loss, ep_steps = 0.0, 0.0, 0
        while not done:
            action = agent.act(obs, eps)
            next_obs, reward, term, trunc, _ = env.step(action)
            done = term or trunc
            buffer.push(obs, action, reward * reward_scale, next_obs, done)  # scaled for stability
            obs = next_obs
            ep_reward += reward
            ep_steps += 1
            if len(buffer) >= warmup and ep_steps % train_freq == 0:
                ep_loss += agent.update(buffer.sample(batch_size))
                agent.soft_update(soft_tau)
        eps = max(eps_min, eps * eps_decay)
        history.append({"episode": ep, "reward": ep_reward,
                        "loss": ep_loss / max(ep_steps, 1), "epsilon": eps})
        if verbose:
            print(f"[train:{agent_kind}] ep {ep:3d}  reward={ep_reward:12.1f}  "
                  f"avg_loss={ep_loss/max(ep_steps,1):.4f}  eps={eps:.3f}")

    dur = time.time() - t0
    config = dict(state_dim=int(state_dim), n_actions=int(env.action_space.n), hidden=128,
                  gamma=gamma, agent_kind=agent_kind, lean_state=lean_state, **flags)
    metadata = dict(agent_kind=agent_kind, episodes=episodes, fraction=fraction,
                    episode_length=episode_length, seed=seed, train_seconds=round(dur, 1),
                    gamma=gamma, reward_scale=reward_scale, fraud_frac=fraud_frac,
                    env_costs=vars(EnvCosts()), final_reward=history[-1]["reward"],
                    reward_history=[round(h["reward"], 1) for h in history])
    version = save_policy(new_version(agent_kind), agent, config, metadata)
    print(f"[train:{agent_kind}] saved policy -> {version}  ({dur:.1f}s)")
    return version, history


def main() -> None:
    ap = argparse.ArgumentParser(description="Train a DQN fraud-decision policy.")
    ap.add_argument("--agent", choices=list(AGENT_FLAGS), default="dueling")
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--fraction", type=float, default=1.0)
    ap.add_argument("--episode-length", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = ap.parse_args()
    train(args.agent, args.episodes, args.fraction, args.episode_length, args.seed)


if __name__ == "__main__":
    main()
