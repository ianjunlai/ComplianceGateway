"""Event contracts shared with the Java gateway."""
from typing import Literal, Optional

from pydantic import BaseModel, Field


class AuditRequestEvent(BaseModel):
    request_id: str
    source_system: str
    timestamp: str  # ISO-8601 UTC, set by the gateway at ingress
    audit_query: str


class StageTimings(BaseModel):
    """Per-stage instrumentation."""
    queue_wait_ms: int = 0
    ner_ms: int = 0
    retrieval_ms: int = 0
    generation_ms: int = 0
    total_ms: int = 0


class ComplianceDecision(BaseModel):
    """Constrained-decoding target schema."""
    decision: Literal["APPROVE", "DENY", "UNKNOWN"]
    reasoning: str


class AuditResultEvent(BaseModel):
    request_id: str
    source_system: str
    decision: str  # APPROVE | DENY | UNKNOWN | ERROR
    reasoning: str = ""
    retrieved_chunk_ids: list[str] = Field(default_factory=list)
    # The chunks the model actually read: the top-K prefix plus any provisions
    # attached because a retrieved clause cites them.
    context_chunk_ids: list[str] = Field(default_factory=list)
    strategy: str
    stage_timings_ms: dict[str, int] = Field(default_factory=dict)
    completed_at: str
    error: Optional[str] = None
