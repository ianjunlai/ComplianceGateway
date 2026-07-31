# -*- coding: utf-8 -*-
"""Stand-in for the sync inference API, for rehearsing the load test.

E3 measures the *gateway's* behaviour under concurrency — 202 versus 200 versus
503 versus 504, the semaphore, the polling loop, the JMeter extractors. None of
that depends on how long the SLM takes, but all of it is unverifiable on
hardware where one real inference costs minutes: this project's own laptop
degraded from 160 s to over 26 min per request once the sync API and Ollama
were competing for memory.

This serves the same contract as `inference-service/sync_api.py` with a
configurable delay instead of a pipeline, so the plumbing can be debugged in
seconds and the real backend is only needed when actual latency figures are
being collected. Nothing in the gateway changes: it already calls
`gateway.inference.sync-url`, so pointing that at this process is the whole
switch.

It also records the peak number of requests in flight, which is what
distinguishes the throttled condition from the unbounded one — a claim that
would otherwise rest on inference timings alone.

    python loadtest/stub_inference.py --delay 3
    python loadtest/stub_inference.py --delay 250   # long enough to trip the
                                                    # gateway's 240 s acquire timeout
"""
import argparse
import threading
import time
from datetime import datetime, timezone

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="ComplianceGateway stub inference")

DELAY_SECONDS = 3.0

_lock = threading.Lock()
_in_flight = 0
_peak_in_flight = 0
_served = 0


class SyncInferRequest(BaseModel):
    request_id: str
    source_system: str
    audit_query: str


@app.post("/infer")
def infer(req: SyncInferRequest):
    global _in_flight, _peak_in_flight, _served
    with _lock:
        _in_flight += 1
        _peak_in_flight = max(_peak_in_flight, _in_flight)
        concurrent_now = _in_flight
    started = time.perf_counter()
    print(f"  -> {req.request_id[:8]} start   (in flight: {concurrent_now})", flush=True)
    try:
        time.sleep(DELAY_SECONDS)
    finally:
        with _lock:
            _in_flight -= 1
            _served += 1
    elapsed_ms = int((time.perf_counter() - started) * 1000)
    print(f"  <- {req.request_id[:8]} done in {elapsed_ms}ms "
          f"(served {_served}, peak concurrency {_peak_in_flight})", flush=True)

    # Same shape as common.schemas.AuditResultEvent, so the gateway and any
    # client parse it exactly as they would a real result.
    return {
        "request_id": req.request_id,
        "source_system": req.source_system,
        "decision": "DENY",
        "reasoning": f"[stub] fixed {DELAY_SECONDS:g}s response, no inference performed",
        "retrieved_chunk_ids": ["gdpr-art-46", "gdpr-art-45-1"],
        "strategy": "stub",
        "stage_timings_ms": {
            "queue_wait_ms": 0, "ner_ms": 0, "retrieval_ms": 0,
            "generation_ms": elapsed_ms, "total_ms": elapsed_ms,
        },
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "error": None,
    }


@app.get("/stats")
def stats():
    """Peak concurrency is the evidence for whether the throttle held."""
    return {"served": _served, "in_flight": _in_flight, "peak_in_flight": _peak_in_flight,
            "delay_seconds": DELAY_SECONDS}


@app.post("/stats/reset")
def reset_stats():
    global _peak_in_flight, _served
    with _lock:
        _peak_in_flight = _in_flight
        _served = 0
    return {"reset": True}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--delay", type=float, default=3.0,
                        help="seconds each request blocks, standing in for inference")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    DELAY_SECONDS = args.delay
    print(f"stub inference on :{args.port}, {DELAY_SECONDS:g}s per request", flush=True)
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")
