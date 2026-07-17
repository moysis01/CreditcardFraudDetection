"""Verification suite for the RL decision layer (plan R6).

Covers: env reward correctness for every (action × label), budget dynamics,
determinism, replay buffer, policy round-trip parity, and the live /engine switch.
"""
from __future__ import annotations

import numpy as np
import pytest

from antifraud.rl.env import (APPROVE, DECLINE, REVIEW, EnvCosts, FraudDecisionEnv)


@pytest.fixture(scope="module")
def env():
    return FraudDecisionEnv(split="test", fraction=0.1, episode_length=500,
                            oversample_fraud=True, seed=1)


# ---- reward model -------------------------------------------------------
def test_reward_signs(env):
    env.reset()
    env._budget = 5
    c = env.costs
    assert env._reward(APPROVE, 1, 100.0) == (-100.0, "missed_fraud")
    assert env._reward(APPROVE, 0, 100.0) == (c.approve_legit_reward, "approved_legit")
    assert env._reward(DECLINE, 1, 100.0) == (100.0, "caught_fraud")
    assert env._reward(DECLINE, 0, 100.0) == (-c.false_decline_cost, "false_decline")


def test_review_spends_budget_and_penalises_when_empty(env):
    env.reset()
    env._budget = 1
    r, outcome = env._reward(REVIEW, 1, 100.0)          # uses the last budget unit
    assert outcome == "reviewed" and r == pytest.approx(100.0 - env.costs.review_cost)
    assert env._budget == 0
    r2, outcome2 = env._reward(REVIEW, 1, 100.0)        # budget now exhausted
    assert outcome2 == "over_review"
    assert r2 < r                                       # over-review is worse


def test_env_deterministic():
    e1 = FraudDecisionEnv(split="test", fraction=0.1, episode_length=50, seed=7)
    e2 = FraudDecisionEnv(split="test", fraction=0.1, episode_length=50, seed=7)
    assert np.allclose(e1.reset()[0], e2.reset()[0])


def test_state_dimension(env):
    obs, _ = env.reset()
    assert obs.shape == (env.observation_space.shape[0],)
    assert env.observation_space.shape[0] == env.X_scaled.shape[1] + 3


# ---- replay buffer ------------------------------------------------------
def test_replay_buffer_shapes_and_priority():
    from antifraud.rl.replay import ReplayBuffer
    buf = ReplayBuffer(capacity=1000, alpha=1.0, seed=0)
    for i in range(200):
        reward = 1000.0 if i == 0 else 0.0     # one high-|reward| transition
        buf.push(np.zeros(4), 1, reward, np.zeros(4), False)
    s, a, r, ns, d = buf.sample(64)
    assert s.shape == (64, 4) and a.shape == (64,) and r.shape == (64,)
    # priority sampling should draw the big-reward transition far more than uniform (0.5%)
    big = buf.sample(2000)[2]
    assert np.mean(big == 1000.0) > 0.05


# ---- policy round-trip (uses whatever policy was last trained) ----------
def _policy_available():
    from antifraud.rl.policy_registry import resolve_version
    try:
        resolve_version("latest"); return True
    except FileNotFoundError:
        return False


needs_policy = pytest.mark.skipif(not _policy_available(),
                                  reason="no trained policy; run antifraud.rl.train_agent")


@needs_policy
def test_policy_roundtrip_deterministic():
    from antifraud.rl.policy_registry import load_policy
    agent, cfg, _ = load_policy("latest")
    s = np.random.RandomState(0).randn(cfg["state_dim"]).astype("float32")
    assert agent.act_greedy(s) == agent.act_greedy(s)
    assert 0 <= agent.act_greedy(s) < cfg["n_actions"]


# ---- live engine switch (TestClient, no server) -------------------------
def test_sequential_env_reward_is_negative_cost():
    """Reward must equal negative incremental cost (0 for correctly stopping fraud),
    the alignment that stops the agent over-blocking."""
    from antifraud.rl.sequential import (build_sessions, SequentialCardEnv,
                                         APPROVE, DECLINE, BLOCK)
    cards, proba, y, amounts, scaled, thr = build_sessions("test", 0.1, seed=3,
                                                           n_compromised=50, n_legit_cards=200)
    env = SequentialCardEnv(cards, proba, y, amounts, scaled, episode_cards=None, seed=3)
    env.reset()
    c = env.costs
    # find a fraud and a legit transaction and check their reward under each action
    seen_fraud = seen_legit = False
    for _ in range(5000):
        card = env._order[env.ci]; i = card[env.pi]
        label, amount = int(env.y[i]), float(env.amounts[i])
        obs, r, done, _ = env.step(DECLINE)
        if label == 1 and not seen_fraud:      # correctly declining fraud costs nothing
            assert r == 0.0; seen_fraud = True
        if label == 0 and not seen_legit:      # declining legit costs the friction fee
            assert r == pytest.approx(-c.false_decline_cost); seen_legit = True
        if done or (seen_fraud and seen_legit):
            break
    assert seen_fraud and seen_legit


@needs_policy
def test_engine_switch_and_review_action():
    from fastapi.testclient import TestClient
    from antifraud.serving.app import app
    from antifraud.common.schemas import FEATURE_ORDER
    with TestClient(app) as client:
        info = client.get("/model").json()
        if "dqn" not in info["engines"]:
            pytest.skip("dqn engine not loaded")
        assert client.post("/engine", json={"engine": "dqn"}).json()["active_engine"] == "dqn"
        payload = {k: 0.0 for k in FEATURE_ORDER}
        r = client.post("/score", json=payload).json()
        assert r["action"] in ("approve", "decline", "review")
        assert r["engine"] == "dqn"
        assert r["latency_ms"] < 100.0
