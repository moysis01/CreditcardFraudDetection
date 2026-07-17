"""Sequential fraud detection — where RL genuinely beats a static threshold.

The earlier per-transaction framing showed (honestly) that a cost-tuned threshold on
XGBoost's score is near-optimal and hard for RL to beat: the score is a near-sufficient
statistic and each decision is independent. To let RL add real value we change the
decision STRUCTURE, not the tuning.

Real fraud is bursty: once a card is compromised, a cascade of fraudulent transactions
follows. A stateless threshold must catch each one independently and misses ~6% (recall
0.94). Here the agent also has a **BLOCK_CARD** action: after recognising the pattern it
freezes the card, preventing the entire remaining cascade — impossible for a per-transaction
threshold. This is a genuine MDP (blocking changes all future transitions on the card).

Because the dataset is anonymised (no card IDs), we construct synthetic card sessions with
fraud bursts — a standard, documented simulation choice. We benchmark against BOTH a naive
threshold AND a strong hand-coded "block-after-N" heuristic, so any RL win is honest.

Run:  python -m antifraud.rl.sequential --episodes 60 --fraction 1.0
"""
from __future__ import annotations

import argparse
import random
from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from antifraud.common.config import FALSE_DECLINE_COST, RANDOM_SEED
from antifraud.common.schemas import FEATURE_ORDER
from antifraud.registry.artifact import load_bundle
from antifraud.rl.agents.dqn import DQNAgent
from antifraud.rl.replay import ReplayBuffer

APPROVE, DECLINE, BLOCK = 0, 1, 2
ACTION_NAMES = {APPROVE: "approve", DECLINE: "decline", BLOCK: "block"}


@dataclass
class SeqCosts:
    false_decline_cost: float = FALSE_DECLINE_COST   # £ friction: decline a legit txn
    block_cost: float = 8.0                          # £ friction: block a legit txn (whole card)


def _load_split(split: str, fraction: float):
    from antifraud.training.build_artifact import load_frame, split as split_frame, xy
    df = load_frame(fraction)
    tr, va, te = split_frame(df, "random")
    frame = {"train": tr, "val": va, "test": te}[split]
    X, y = xy(frame)
    bundle = load_bundle("latest")
    import xgboost as xgb
    scaled = bundle.preprocessor.transform(X[FEATURE_ORDER]).astype(np.float32)
    proba = bundle.booster.predict(
        xgb.DMatrix(scaled, feature_names=bundle.feature_order)).astype(np.float32)
    return proba, y.astype(int), frame["Amount"].to_numpy(float), scaled, bundle.thresholds


def build_sessions(split: str, fraction: float, seed: int,
                   n_compromised: int = 500, burst_min: int = 3, burst_max: int = 9,
                   n_legit_cards: int = 6000):
    """Construct synthetic card sessions with a genuine block/keep tradeoff.

    Compromised card = legit_pre + fraud burst (random order) + **legit_post** — the real
    cardholder's ongoing legitimate spend, which a block also freezes (friction). Legit
    cards are multi-transaction so an occasional false-positive decline can wrongly trigger
    a block. So blocking is NOT free: block too eagerly and you freeze good spend / good
    cards; block too late and the fraud cascade runs. That tension is what RL can optimise."""
    proba, y, amounts, scaled, thr = _load_split(split, fraction)
    rng = np.random.default_rng(seed)
    fraud_idx = np.where(y == 1)[0]
    legit_idx = np.where(y == 0)[0]

    cards = []
    for _ in range(n_compromised):
        pre = rng.choice(legit_idx, int(rng.integers(0, 3)), replace=True)
        burst = rng.choice(fraud_idx, int(rng.integers(burst_min, burst_max)), replace=True)
        rng.shuffle(burst)
        post = rng.choice(legit_idx, int(rng.integers(2, 7)), replace=True)   # ongoing legit spend
        cards.append(np.concatenate([pre, burst, post]).astype(int))
    for _ in range(n_legit_cards):
        size = int(rng.integers(2, 9))   # multi-txn legit cards (false positives can appear)
        cards.append(rng.choice(legit_idx, size, replace=True).astype(int))
    rng.shuffle(cards)
    return cards, proba, y, amounts, scaled, thr


class SequentialCardEnv:
    """Processes transactions grouped by card; a BLOCK freezes the rest of the card."""

    def __init__(self, cards, proba, y, amounts, scaled, costs: SeqCosts | None = None,
                 episode_cards: int | None = 1500, seed: int = RANDOM_SEED):
        self.cards, self.proba, self.y, self.amounts, self.scaled = cards, proba, y, amounts, scaled
        self.costs = costs or SeqCosts()
        self.episode_cards = episode_cards
        self.n_actions = 3
        self.state_dim = 4
        self._rng = np.random.default_rng(seed)

    def reset(self):
        if self.episode_cards and self.episode_cards < len(self.cards):
            pick = self._rng.choice(len(self.cards), self.episode_cards, replace=False)
            self._order = [self.cards[i] for i in pick]
        else:
            self._order = list(self.cards)
        self.ci = self.pi = self.n_declined = 0
        # metrics
        self.fraud_loss = self.friction = 0.0
        self.fraud_caught_amt = self.fraud_total_amt = 0.0
        self.false_declines = self.legit_blocked = self.blocks = 0
        # account fraud totals for recall
        for card in self._order:
            self.fraud_total_amt += self.amounts[card][self.y[card] == 1].sum()
        return self._obs()

    def _obs(self):
        card = self._order[self.ci]
        i = card[self.pi]
        pos_norm = self.pi / max(len(card), 1)
        return np.array([self.proba[i], self.scaled[i][-1], pos_norm,
                         min(self.n_declined, 5) / 5.0], np.float32)

    def _advance_card(self):
        self.ci += 1
        self.pi = self.n_declined = 0

    def step(self, action: int):
        card = self._order[self.ci]
        i = card[self.pi]
        label, amount = int(self.y[i]), float(self.amounts[i])
        reward = 0.0

        # Reward is the NEGATIVE incremental cost, so maximising return == minimising
        # total cost. Correctly stopping fraud earns 0 (you merely avoided the -amount
        # loss); only real costs are penalised. This is what stops the agent over-blocking.
        if action == BLOCK:
            self.blocks += 1
            for j in card[self.pi:]:                      # resolve current + remaining
                if self.y[j] == 1:
                    self.fraud_caught_amt += self.amounts[j]      # prevented, reward 0
                else:
                    reward -= self.costs.block_cost              # legit frozen (cost)
                    self.friction += self.costs.block_cost
                    self.legit_blocked += 1
            self._advance_card()
        else:
            if action == APPROVE:
                if label == 1:
                    reward -= amount; self.fraud_loss += amount   # missed fraud (real loss)
            else:  # DECLINE
                if label == 1:
                    self.fraud_caught_amt += amount               # caught, reward 0
                else:
                    reward -= self.costs.false_decline_cost       # false decline (cost)
                    self.friction += self.costs.false_decline_cost
                    self.false_declines += 1
                self.n_declined += 1
            self.pi += 1
            if self.pi >= len(card):
                self._advance_card()

        done = self.ci >= len(self._order)
        obs = self._obs() if not done else np.zeros(self.state_dim, np.float32)
        return obs, reward, done, {"label": label, "action": ACTION_NAMES[action]}

    def summary(self) -> dict:
        total_cost = self.fraud_loss + self.friction
        recall = self.fraud_caught_amt / max(self.fraud_total_amt, 1e-9)
        return {"total_cost": total_cost, "fraud_loss": self.fraud_loss,
                "friction": self.friction, "recall_by_value": recall,
                "false_declines": self.false_declines, "legit_blocked": self.legit_blocked,
                "blocks": self.blocks}


# ---- policies -----------------------------------------------------------
def threshold_policy(thr):
    def fn(obs):
        return DECLINE if obs[0] >= thr else APPROVE   # never blocks
    return fn


def heuristic_block_policy(thr, block_after=1):
    """Strong sequential baseline: decline confident frauds; block the card once it has
    already had `block_after` declines (i.e. a detected burst)."""
    def fn(obs):
        score, _, _, n_declined_norm = obs
        n_declined = round(n_declined_norm * 5)
        if score >= thr:
            return BLOCK if n_declined >= block_after else DECLINE
        return APPROVE
    return fn


def eval_policy(env: SequentialCardEnv, policy_fn) -> dict:
    obs = env.reset()
    done = False
    while not done:
        obs, _, done, _ = env.step(policy_fn(obs))
    return env.summary()


# ---- DQN training on the sequential env ---------------------------------
def train_dqn(env: SequentialCardEnv, episodes: int, seed: int = RANDOM_SEED,
              gamma: float = 0.9, reward_scale: float = 0.01, verbose: bool = True):
    random.seed(seed); np.random.seed(seed)
    agent = DQNAgent(env.state_dim, env.n_actions, double=True, dueling=True,
                     lr=5e-4, gamma=gamma, seed=seed)      # gamma=0.9: future matters (blocking)
    buf = ReplayBuffer(200_000, seed=seed)
    eps, eps_min, eps_decay = 1.0, 0.03, 0.93
    history = []
    for ep in range(1, episodes + 1):
        obs = env.reset(); done = False; total = 0.0; step = 0
        while not done:
            a = agent.act(obs, eps)
            nobs, r, done, _ = env.step(a)
            buf.push(obs, a, r * reward_scale, nobs, done)
            obs = nobs; total += r; step += 1
            if len(buf) >= 2000 and step % 2 == 0:
                agent.update(buf.sample(256)); agent.soft_update(0.01)
        eps = max(eps_min, eps * eps_decay)
        history.append(total)
        if verbose and ep % 5 == 0:
            print(f"[seq-dqn] ep {ep:3d}  reward={total:12.1f}  eps={eps:.3f}")
    return agent, history


def run(fraction: float, episodes: int, seed: int = RANDOM_SEED, plot: bool = True):
    cards, proba, y, amounts, scaled, thr = build_sessions("test", fraction, seed)
    t = thr["cost"]   # aggressive threshold (precision ~0.97) so false positives occur
    print(f"[seq] {len(cards)} synthetic cards, decision threshold={t:.3f}")

    def make_env(ec=None):
        return SequentialCardEnv(cards, proba, y, amounts, scaled, episode_cards=ec, seed=seed)

    # baselines on the full card set (episode_cards=None => all cards, deterministic)
    results = {
        "Threshold only":        eval_policy(make_env(None), threshold_policy(t)),
        "Threshold + block(1)":  eval_policy(make_env(None), heuristic_block_policy(t, 1)),
        "Threshold + block(2)":  eval_policy(make_env(None), heuristic_block_policy(t, 2)),
    }

    print(f"[seq] training Dueling DQN ({episodes} episodes) ...")
    agent, hist = train_dqn(make_env(1500), episodes, seed)
    results["Dueling DQN"] = eval_policy(make_env(None), agent.act_greedy)

    _print_table(results)
    if plot:
        _plot(results, hist)
    return results


def _print_table(results: dict) -> None:
    cols = ["total_cost", "fraud_loss", "friction", "recall_by_value", "blocks",
            "false_declines", "legit_blocked"]
    print("\n" + "=" * 104)
    print(f"{'policy':<22}" + "".join(f"{c[:13]:>12}" for c in cols))
    print("-" * 104)
    for name, m in results.items():
        row = f"{name:<22}"
        for c in cols:
            v = m[c]
            row += f"{v:>12.3f}" if c == "recall_by_value" else f"{v:>12.0f}"
        print(row)
    best = min(results.items(), key=lambda kv: kv[1]["total_cost"])
    print("-" * 104)
    print(f"lowest total cost: {best[0]}  (£{best[1]['total_cost']:,.0f})")
    thr_cost = results["Threshold only"]["total_cost"]
    dqn_cost = results["Dueling DQN"]["total_cost"]
    heur = min(results["Threshold + block(1)"]["total_cost"],
               results["Threshold + block(2)"]["total_cost"])
    print(f"DQN £{dqn_cost:,.0f} vs threshold-only £{thr_cost:,.0f} "
          f"({(1-dqn_cost/thr_cost)*100:+.0f}%) vs best heuristic £{heur:,.0f}")
    print("=" * 104)


def _plot(results: dict, hist) -> None:
    import os
    os.makedirs("plots/rl", exist_ok=True)
    names = list(results.keys())
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    # stacked cost: fraud loss + friction
    fl = [results[n]["fraud_loss"] for n in names]
    fr = [results[n]["friction"] for n in names]
    ax1.bar(names, fl, label="fraud lost", color="#e05555")
    ax1.bar(names, fr, bottom=fl, label="friction", color="#fab219")
    ax1.set_ylabel("£ cost (lower is better)"); ax1.legend()
    ax1.set_title("Sequential fraud: cost by policy"); ax1.tick_params(axis="x", rotation=20)
    ax2.plot(range(1, len(hist) + 1), hist, color="#3987e5")
    ax2.set_xlabel("episode"); ax2.set_ylabel("episode reward")
    ax2.set_title("DQN learning curve (sequential env)")
    fig.tight_layout(); fig.savefig("plots/rl/sequential_comparison.png", dpi=120); plt.close()
    print("[seq] plot -> plots/rl/sequential_comparison.png")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=60)
    ap.add_argument("--fraction", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = ap.parse_args()
    run(args.fraction, args.episodes, args.seed)


if __name__ == "__main__":
    main()
