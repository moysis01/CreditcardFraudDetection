"""FastAPI real-time scoring service (plan Part B).

Endpoints
  GET  /                dashboard (static single-page app)
  POST /score           score one transaction               -> ScoreResponse
  POST /score/batch     score many (throughput benchmark)   -> BatchScoreResponse
  POST /ingest          score + record label + broadcast     (used by the simulator)
  GET  /healthz         liveness
  GET  /model           served model version + metadata
  GET  /metrics         live latency percentiles + KPIs (Telemetry snapshot)
  POST /threshold       move the operating threshold live (dashboard console)
  WS   /stream          pushes each scored transaction to connected dashboards

The Scorer artifact is loaded ONCE at startup (plan B2). Scoring is sub-millisecond,
so it runs inline in the async handlers.
"""
from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Optional, Set

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from antifraud.common.config import SERVED_VERSION
from antifraud.common.schemas import (BatchScoreRequest, BatchScoreResponse,
                                       ScoreResponse, Transaction)
from antifraud.serving.scorer import Scorer
from antifraud.serving.decision_engine import PolicyEngine, ThresholdEngine
from antifraud.serving.telemetry import Telemetry

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="Real-Time Anti-Fraud Platform", version="1.0")

# --- singletons, populated at startup ------------------------------------
scorer: Optional[Scorer] = None
engines: dict = {}                 # name -> DecisionEngine
active_engine_name: str = "threshold"
telemetry = Telemetry()


class _WSManager:
    def __init__(self) -> None:
        self.active: Set[WebSocket] = set()

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self.active.add(ws)

    def disconnect(self, ws: WebSocket) -> None:
        self.active.discard(ws)

    async def broadcast(self, message: dict) -> None:
        dead = []
        for ws in list(self.active):
            try:
                await ws.send_json(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)


ws_manager = _WSManager()


@app.on_event("startup")
def _load_model() -> None:
    global scorer, engines, active_engine_name
    t0 = time.perf_counter()
    scorer = Scorer(version=SERVED_VERSION)
    engines = {"threshold": ThresholdEngine(scorer)}
    # Load the DQN policy engine if a trained policy exists (optional).
    try:
        from antifraud.rl.policy_registry import load_policy
        agent, cfg, meta = load_policy("latest")
        engines["dqn"] = PolicyEngine(scorer, agent, meta,
                                      lean_state=cfg.get("lean_state", False))
        print(f"[serving] loaded DQN policy {meta.get('version')} "
              f"({meta.get('agent_kind')}, lean={cfg.get('lean_state', False)})")
    except Exception as e:
        print(f"[serving] no DQN policy loaded ({e}); threshold engine only")
    load_ms = (time.perf_counter() - t0) * 1000.0
    print(f"[serving] loaded model {scorer.version} in {load_ms:.1f} ms "
          f"(engines: {list(engines)})")


async def _process(txn: Transaction, true_label: Optional[int]) -> ScoreResponse:
    """Decide via the active engine, record telemetry, and broadcast to dashboards."""
    resp = engines[active_engine_name].decide(txn)
    telemetry.record(resp.latency_ms, resp.action.value, txn.Amount,
                     active_engine_name, true_label)
    await ws_manager.broadcast({
        "type": "txn",
        "probability": round(resp.probability, 4),
        "decision": resp.decision,
        "action": resp.action.value,
        "engine": resp.engine,
        "amount": txn.Amount,
        "amount_at_risk": resp.amount_at_risk,
        "latency_ms": resp.latency_ms,
        "true_label": true_label,
        "reason_codes": [rc.model_dump() for rc in resp.reason_codes],
        "ts": time.time(),
    })
    return resp


@app.post("/score", response_model=ScoreResponse)
async def score(txn: Transaction):
    return await _process(txn, true_label=None)


class IngestRequest(BaseModel):
    transaction: Transaction
    true_label: Optional[int] = None


@app.post("/ingest", response_model=ScoreResponse)
async def ingest(req: IngestRequest):
    return await _process(req.transaction, true_label=req.true_label)


@app.post("/score/batch", response_model=BatchScoreResponse)
async def score_batch(req: BatchScoreRequest):
    t0 = time.perf_counter()
    results = scorer.score_many(req.transactions)
    total_ms = (time.perf_counter() - t0) * 1000.0
    for txn, r in zip(req.transactions, results):
        action = "decline" if r.decision == "fraud" else "approve"
        telemetry.record(r.latency_ms, action, txn.Amount, "threshold", None)
    return BatchScoreResponse(
        results=results, count=len(results), total_latency_ms=round(total_ms, 3),
        throughput_tps=round(len(results) / max(total_ms / 1000.0, 1e-9), 1),
    )


@app.get("/healthz")
async def healthz():
    return {"status": "ok", "model_loaded": scorer is not None}


@app.get("/model")
async def model_info():
    dqn = engines.get("dqn")
    return {"version": scorer.version, "threshold": scorer.threshold,
            "threshold_key": scorer.threshold_key, "thresholds": scorer.thresholds,
            "engines": list(engines), "active_engine": active_engine_name,
            "policy_version": dqn.policy_version if dqn else None,
            "policy_kind": dqn.metadata.get("agent_kind") if dqn else None,
            "metadata": scorer.bundle.metadata}


@app.get("/metrics")
async def metrics():
    return telemetry.snapshot(active_engine_name)


class EngineRequest(BaseModel):
    engine: str        # "threshold" | "dqn"


@app.post("/engine")
async def set_engine(req: EngineRequest):
    global active_engine_name
    if req.engine not in engines:
        return JSONResponse(status_code=400,
                            content={"error": f"unknown engine '{req.engine}'",
                                     "available": list(engines)})
    active_engine_name = req.engine
    return {"active_engine": active_engine_name, "available": list(engines)}


class ThresholdRequest(BaseModel):
    key: Optional[str] = None      # "f1" | "cost" | "default"
    value: Optional[float] = None  # or a raw threshold in [0,1]


@app.post("/threshold")
async def set_threshold(req: ThresholdRequest):
    if req.value is not None:
        t = scorer.set_threshold(req.value)
    elif req.key is not None:
        t = scorer.set_threshold(req.key)
    else:
        return JSONResponse(status_code=400, content={"error": "provide key or value"})
    return {"threshold": t, "threshold_key": scorer.threshold_key}


@app.websocket("/stream")
async def stream(ws: WebSocket):
    await ws_manager.connect(ws)
    try:
        # Send an initial metrics snapshot, then keep the socket open.
        await ws.send_json({"type": "hello", "metrics": telemetry.snapshot(active_engine_name),
                            "model_version": scorer.version, "engines": list(engines)})
        while True:
            await asyncio.sleep(1.0)
            await ws.send_json({"type": "metrics", "metrics": telemetry.snapshot(active_engine_name)})
    except WebSocketDisconnect:
        ws_manager.disconnect(ws)
    except Exception:
        ws_manager.disconnect(ws)


@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")
