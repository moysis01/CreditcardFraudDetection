# Real-Time Anti-Fraud Platform

A low-latency fraud-detection system built on top of the research pipeline in this
repo. It turns the offline batch experiment into: an offline training step that
produces a **versioned, leakage-free model artifact**, a **sub-100 ms FastAPI
scoring service**, and a **live web dashboard**.

See the design in [`../IMPROVEMENTS.md`](../IMPROVEMENTS.md) and the full build plan.

```
antifraud/
  common/     pydantic schemas, config, cost model
  registry/   versioned artifact bundles (the train↔serve contract)
  training/   leakage-free training -> models/<version>/
  serving/    FastAPI service + hot-path scorer + telemetry + dashboard
  streaming/  transaction replay simulator (the demo feed)
  rl/         Deep Q-Learning decision layer (env, DQN family, evaluation)
```

## Reinforcement-learning decision layer (`antifraud/rl/`)

On top of the XGBoost score, a **Deep Q-Network** learns a per-transaction policy —
**APPROVE / DECLINE / REVIEW** — that minimises net £ loss, going beyond a single
fixed threshold. A shared *review budget* makes REVIEW actions have downstream
consequences, so this is a genuine MDP (not just a bandit). See the plan for the
honest framing.

```sh
# Train a policy (dueling = double + dueling DQN). Needs the XGBoost artifact first.
python -m antifraud.rl.train_agent --agent dueling --episodes 40 --fraction 1.0

# Full comparison: XGBoost thresholds vs LinUCB bandit vs DQN / Double / Dueling
python -m antifraud.rl.evaluate --episodes 40 --fraction 1.0      # table + plots/rl/

# Secondary experiment: RL controller that adapts the threshold under drift
python -m antifraud.rl.threshold_controller --episodes 30

# Sequential fraud: where RL genuinely beats a threshold (BLOCK-card action)
python -m antifraud.rl.sequential --episodes 60 --fraction 1.0
```

### Where RL actually wins (the honest result)

- **Per-transaction decision** (`env.py`): a cost-tuned threshold on XGBoost's score is
  near-optimal; a tuned DQN reaches recall parity but does **not** reliably beat it, because
  the score is a near-sufficient statistic. This is the honest, correct finding.
- **Sequential decision** (`sequential.py`): when fraud is bursty and the agent can act on a
  card's *history*, RL wins decisively — **~76% lower cost than a threshold and ~78% lower than
  hand-coded block heuristics**, catching 99.9% of fraud. It learns to escalate scrutiny on a
  card after detecting fraud rather than bluntly freezing the whole card. RL adds value exactly
  where the decision has structure a static threshold cannot express.

The service auto-loads the latest trained policy as a second **decision engine**;
the dashboard's engine toggle switches `XGBoost + threshold` ↔ `DQN policy` live,
and the REVIEW lane + per-engine £ comparison show the difference.

## Quick start

```sh
source .venv/bin/activate            # deps: pip install -r ../requirements.txt

# 1. Build a model artifact (full data; add --fraction 0.15 for a fast run)
python -m antifraud.training.build_artifact

# 2. Start the scoring service + dashboard
python -m uvicorn antifraud.serving.app:app --host 127.0.0.1 --port 8000

# 3. In another terminal, stream transactions into it
python -m antifraud.streaming.replay --rate 30 --fraud-boost 40

# 4. Open the dashboard
open http://127.0.0.1:8000/
```

## What it demonstrates

- **Speed:** single-transaction scoring in a few milliseconds (p99 well under the
  100 ms budget); ~1,700 txn/s batch throughput. The model artifact is loaded once
  at startup; only `Amount` is scaled at request time.
- **Correctness:** the decision threshold is chosen on a **validation** split and
  frozen before the test set is touched (no leakage — the flaw noted in
  `IMPROVEMENTS.md §1.1`). Headline metric is PR-AUC + MCC, not ROC-AUC.
- **Explainability:** every flagged transaction returns top-N **SHAP reason codes**
  (via XGBoost `pred_contribs`, no extra dependency).
- **Business framing:** a cost model (FN = transaction amount, FP = review fee)
  gives a cost-optimal threshold and a "money protected" KPI.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| POST | `/score` | score one transaction |
| POST | `/score/batch` | score many (throughput) |
| POST | `/ingest` | score + record label + broadcast (used by the simulator) |
| POST | `/threshold` | move the operating threshold live (`f1` / `cost` / `default`) |
| GET | `/model` | served version + metadata |
| GET | `/metrics` | live latency percentiles + KPIs |
| WS | `/stream` | live scored-transaction feed |

## Tests

```sh
python -m pytest tests/ -q
```

Covers no-leakage split, valid frozen thresholds, artifact round-trip parity, the
API contract, and a latency-under-budget sanity check.
