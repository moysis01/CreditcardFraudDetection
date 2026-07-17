"""Transaction replay simulator (plan B7) — the demo's heartbeat.

There is no live bank feed, so we replay creditcard.csv into the running service's
/ingest endpoint at a configurable rate. Each POST carries the true label so the
dashboard can show detection correctness and money protected.

Because genuine fraud is ~0.17% of rows, --fraud-boost oversamples fraud into the
stream so the live demo actually shows fraud being caught (set to 1 for realistic
base rate).

Run (service must be up):
  python -m antifraud.streaming.replay --rate 25 --fraud-boost 40
"""
from __future__ import annotations

import argparse
import asyncio
import random

import httpx
import pandas as pd

from antifraud.common.config import DATA_PATH, RANDOM_SEED
from antifraud.common.schemas import FEATURE_ORDER


def build_stream(df: pd.DataFrame, fraud_boost: int) -> list[int]:
    """Return a shuffled list of row indices, with fraud rows repeated
    `fraud_boost` times so the demo is lively."""
    fraud_idx = df.index[df.Class == 1].tolist()
    legit_idx = df.index[df.Class == 0].tolist()
    order = legit_idx + fraud_idx * max(fraud_boost, 1)
    random.Random(RANDOM_SEED).shuffle(order)
    return order


async def run(rate: float, url: str, limit: int | None, fraud_boost: int) -> None:
    df = pd.read_csv(DATA_PATH).drop_duplicates().reset_index(drop=True)
    order = build_stream(df, fraud_boost)
    if limit:
        order = order[:limit]

    interval = 1.0 / rate if rate > 0 else 0.0
    sent = frauds = 0
    print(f"[replay] streaming {len(order)} txns to {url}/ingest at ~{rate}/s "
          f"(fraud_boost={fraud_boost}) — Ctrl-C to stop")

    async with httpx.AsyncClient(timeout=5.0) as client:
        for idx in order:
            row = df.iloc[idx]
            payload = {
                "transaction": {k: float(row[k]) for k in FEATURE_ORDER},
                "true_label": int(row["Class"]),
            }
            try:
                r = await client.post(f"{url}/ingest", json=payload)
                if r.status_code == 200 and r.json()["decision"] == "fraud":
                    frauds += 1
            except Exception as e:
                print(f"[replay] error: {e}")
                await asyncio.sleep(1.0)
                continue
            sent += 1
            if sent % 200 == 0:
                print(f"[replay] sent={sent}  flagged_fraud={frauds}")
            if interval:
                await asyncio.sleep(interval)
    print(f"[replay] done. sent={sent} flagged_fraud={frauds}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Replay transactions into the scoring service.")
    ap.add_argument("--rate", type=float, default=25.0, help="Transactions per second.")
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--limit", type=int, default=None, help="Max transactions to send.")
    ap.add_argument("--fraud-boost", type=int, default=40,
                    help="Repeat each fraud row N times so the demo shows fraud. 1 = realistic.")
    args = ap.parse_args()
    try:
        asyncio.run(run(args.rate, args.url, args.limit, args.fraud_boost))
    except KeyboardInterrupt:
        print("\n[replay] stopped.")


if __name__ == "__main__":
    main()
