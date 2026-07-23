"""Synchronous inference API — used ONLY by the gateway's synchronous
endpoints (/audit/sync and /audit/sync-throttled).

Wraps exactly the same run_pipeline() as the Kafka consumer, so the only
difference between integration modes is how clients wait for results.

Run: uvicorn sync_api:app --port 8000
"""
from datetime import datetime, timezone

from fastapi import FastAPI
from pydantic import BaseModel

from common.schemas import AuditRequestEvent
from pipeline.pipeline import run_pipeline

app = FastAPI(title="ComplianceGateway sync inference")


class SyncInferRequest(BaseModel):
    request_id: str
    source_system: str
    audit_query: str


@app.post("/infer")
def infer(req: SyncInferRequest):
    event = AuditRequestEvent(
        request_id=req.request_id,
        source_system=req.source_system,
        timestamp=datetime.now(timezone.utc).isoformat(),
        audit_query=req.audit_query,
    )
    result = run_pipeline(event, queue_wait_ms=0)
    return result.model_dump()
