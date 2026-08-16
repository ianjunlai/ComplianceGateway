"""Synchronous inference API, called only by the gateway's synchronous endpoints."""
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
