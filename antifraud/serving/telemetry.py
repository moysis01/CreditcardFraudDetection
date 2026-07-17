"""Live telemetry (plan B5 + R-E) — latency percentiles, throughput, and now
per-engine money KPIs so the threshold engine and the DQN policy can be compared
live on the dashboard.

Thread-safe and allocation-light: a bounded deque of recent latencies plus running
counters, kept per decision engine.
"""
from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from typing import Deque, Dict, Optional

from antifraud.common.config import FALSE_DECLINE_COST, REVIEW_COST


class _EngineStats:
    __slots__ = ("scored", "flagged", "reviews", "false_declines",
                 "amount_saved", "amount_lost", "amount_at_risk")

    def __init__(self):
        self.scored = self.flagged = self.reviews = self.false_declines = 0
        self.amount_saved = self.amount_lost = self.amount_at_risk = 0.0

    def as_dict(self) -> Dict:
        friction = self.false_declines * FALSE_DECLINE_COST + self.reviews * REVIEW_COST
        return {
            "scored": self.scored, "flagged": self.flagged, "reviews": self.reviews,
            "false_declines": self.false_declines,
            "amount_saved": round(self.amount_saved, 2),
            "amount_lost": round(self.amount_lost, 2),
            "amount_at_risk": round(self.amount_at_risk, 2),
            "friction_cost": round(friction, 2),
            "total_cost": round(self.amount_lost + friction, 2),
        }


class Telemetry:
    def __init__(self, window: int = 5000):
        self._lock = threading.Lock()
        self._latencies: Deque[float] = deque(maxlen=window)
        self.started = time.time()
        self.total_scored = 0
        self.by_engine: Dict[str, _EngineStats] = defaultdict(_EngineStats)

    def record(self, latency_ms: float, action: str, amount: float, engine: str,
               true_label: Optional[int] = None) -> None:
        flagged = action != "approve"
        with self._lock:
            self._latencies.append(latency_ms)
            self.total_scored += 1
            s = self.by_engine[engine]
            s.scored += 1
            if action == "review":
                s.reviews += 1
            if flagged:
                s.flagged += 1
                s.amount_at_risk += amount
                if true_label == 1:
                    s.amount_saved += amount        # correctly stopped fraud
                elif true_label == 0 and action == "decline":
                    s.false_declines += 1           # wrongly blocked a legit sale
            elif true_label == 1:
                s.amount_lost += amount             # approved a fraud (missed)

    @staticmethod
    def _pct(sorted_vals, q: float) -> float:
        if not sorted_vals:
            return 0.0
        return sorted_vals[min(len(sorted_vals) - 1, int(q * len(sorted_vals)))]

    def snapshot(self, active_engine: str = "") -> Dict:
        with self._lock:
            vals = sorted(self._latencies)
            total = self.total_scored
            engines = {name: s.as_dict() for name, s in self.by_engine.items()}
        uptime = max(time.time() - self.started, 1e-6)
        return {
            "total_scored": total,
            "active_engine": active_engine,
            "engines": engines,
            "latency_ms": {
                "p50": round(self._pct(vals, 0.50), 3),
                "p95": round(self._pct(vals, 0.95), 3),
                "p99": round(self._pct(vals, 0.99), 3),
                "max": round(vals[-1], 3) if vals else 0.0,
            },
            "throughput_tps": round(total / uptime, 1),
            "uptime_s": round(uptime, 1),
        }
