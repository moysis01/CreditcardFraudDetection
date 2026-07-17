"""Resumable multi-seed robustness sweep for the sequential RL result.

Each invocation runs ONE not-yet-done seed (so a self-paced /loop can drive it, and
it is crash-resumable), appends the row to results/rl_sweep.jsonl, and regenerates a
dissertation-ready results/rl_sweep.md with per-seed rows and mean ± std. When every
target seed is done it prints `SWEEP COMPLETE` and makes no further changes.

Why: the single sequential run (DQN £6,347 vs threshold £38,976) is not publishable on
its own — RL is high-variance. This turns it into mean ± std over many seeds.

Run one seed:   python -m antifraud.rl.sweep
Smoke test:     python -m antifraud.rl.sweep --episodes 2 --target 1 --out-dir /tmp/sweeptest
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from antifraud.rl.sequential import run


def _load(jsonl: Path):
    if not jsonl.exists():
        return []
    return [json.loads(l) for l in jsonl.read_text().splitlines() if l.strip()]


def _fmt(x):
    return f"£{x:,.0f}"


def write_md(rows, md: Path, fraction: float, episodes: int, n_target: int) -> None:
    rows = sorted(rows, key=lambda r: r["seed"])
    lines = [
        "# Sequential RL Robustness Sweep",
        "",
        f"Sequential card-session env · Dueling Double DQN · fraction={fraction} · "
        f"{episodes} episodes/seed · {len(rows)}/{n_target} seeds done.",
        "",
        "**Metric:** total cost £ (fraud lost + friction) on the full card set — lower is better. "
        "DQN also reports fraud recall by value.",
        "",
        "| Seed | Threshold only | Block(1) | Block(2) | **Dueling DQN** | DQN recall | DQN vs threshold |",
        "|-----:|---------------:|---------:|---------:|----------------:|-----------:|-----------------:|",
    ]
    for r in rows:
        red = (1 - r["dqn"] / r["threshold"]) * 100 if r["threshold"] else 0
        lines.append(
            f"| {r['seed']} | {_fmt(r['threshold'])} | {_fmt(r['block1'])} | {_fmt(r['block2'])} "
            f"| **{_fmt(r['dqn'])}** | {r['dqn_recall']:.3f} | {red:+.0f}% |")

    if len(rows) >= 2:
        def ms(key):
            vals = [r[key] for r in rows]
            return statistics.mean(vals), statistics.stdev(vals)
        mt, st = ms("threshold"); mb1, sb1 = ms("block1"); mb2, sb2 = ms("block2")
        md_, sd = ms("dqn")
        reds = [(1 - r["dqn"] / r["threshold"]) * 100 for r in rows if r["threshold"]]
        mr, sr = statistics.mean(reds), statistics.stdev(reds)
        recalls = [r["dqn_recall"] for r in rows]
        lines.append(
            f"| **mean±std** | £{mt:,.0f}±{st:,.0f} | £{mb1:,.0f}±{sb1:,.0f} | "
            f"£{mb2:,.0f}±{sb2:,.0f} | **£{md_:,.0f}±{sd:,.0f}** | "
            f"{statistics.mean(recalls):.3f} | **{mr:+.0f}%±{sr:.0f}** |")
        lines += ["",
                  f"**Headline:** over {len(rows)} seeds the Dueling DQN cuts total cost by "
                  f"**{mr:.0f}% ± {sr:.0f}%** vs the static threshold "
                  f"(£{md_:,.0f}±{sd:,.0f} vs £{mt:,.0f}±{st:,.0f}), at "
                  f"{statistics.mean(recalls):.1%} fraud recall by value."]
    lines.append("")
    md.write_text("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=35)
    ap.add_argument("--fraction", type=float, default=0.3)
    ap.add_argument("--target", type=int, default=8, help="Number of seeds (1..target).")
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    jsonl, md = out / "rl_sweep.jsonl", out / "rl_sweep.md"
    seeds = list(range(1, args.target + 1))

    done = _load(jsonl)
    done_seeds = {r["seed"] for r in done}
    remaining = [s for s in seeds if s not in done_seeds]

    if not remaining:
        write_md(done, md, args.fraction, args.episodes, args.target)
        print(f"SWEEP COMPLETE: {len(done)}/{args.target} seeds -> {md}")
        return

    seed = remaining[0]
    print(f"[sweep] seed {seed}  ({len(done) + 1}/{args.target}) ...")
    res = run(args.fraction, args.episodes, seed, plot=False)
    row = {
        "seed": seed,
        "threshold": res["Threshold only"]["total_cost"],
        "block1": res["Threshold + block(1)"]["total_cost"],
        "block2": res["Threshold + block(2)"]["total_cost"],
        "dqn": res["Dueling DQN"]["total_cost"],
        "dqn_recall": res["Dueling DQN"]["recall_by_value"],
    }
    with jsonl.open("a") as f:
        f.write(json.dumps(row) + "\n")
    done.append(row)
    write_md(done, md, args.fraction, args.episodes, args.target)
    red = (1 - row["dqn"] / row["threshold"]) * 100
    print(f"[sweep] seed {seed} done: DQN £{row['dqn']:,.0f} vs threshold "
          f"£{row['threshold']:,.0f} ({red:+.0f}%). {len(done)}/{args.target} complete.")
    if len(done) >= args.target:
        print(f"SWEEP COMPLETE: {len(done)}/{args.target} seeds -> {md}")


if __name__ == "__main__":
    main()
