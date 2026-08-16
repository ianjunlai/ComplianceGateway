"""Pipeline orchestrator: NER, retrieval, then constrained generation, each stage
timed separately."""
from datetime import datetime, timezone

import config
from common.schemas import AuditRequestEvent, AuditResultEvent
from common.timing import StageTimer
from pipeline import ner
from pipeline.attachment import attach_citations
from pipeline.base import RetrievalStrategy
from pipeline.generation import generate_decision
from pipeline.jurisdiction import jurisdictions_for
from pipeline.strategies import build_strategy

_strategy: RetrievalStrategy | None = None


def get_strategy() -> RetrievalStrategy:
    global _strategy
    if _strategy is None:
        _strategy = build_strategy(config.ACTIVE_STRATEGY)
    return _strategy


def run_pipeline(event: AuditRequestEvent, queue_wait_ms: int = 0) -> AuditResultEvent:
    timer = StageTimer()
    strategy = get_strategy()

    with timer.stage("ner"):
        seeds = ner.extract_seed_entities(event.audit_query)

    # Derived once from who sent the request, and applied to both retrieval and
    # attachment so a provision of another member state cannot enter by either
    # route.
    scope = jurisdictions_for(event.source_system)

    with timer.stage("retrieval"):
        # One ranked list of RETRIEVAL_K chunks serves both Recall@5 and @10
        context = strategy.retrieve(event.audit_query, seeds,
                                    top_k=config.RETRIEVAL_K,
                                    allowed_jurisdictions=scope)

    # The SLM sees a fixed top-GENERATION_CONTEXT_K prefix, so the generation
    # condition does not vary with the retrieval-evaluation K.
    gen_context = attach_citations(
        context.truncated(config.GENERATION_CONTEXT_K),
        allowed_jurisdictions=scope,
    )

    with timer.stage("generation"):
        decision = generate_decision(event.audit_query, gen_context)

    timings = timer.as_millis()
    timings["queue_wait_ms"] = queue_wait_ms

    return AuditResultEvent(
        request_id=event.request_id,
        source_system=event.source_system,
        decision=decision.decision,
        reasoning=decision.reasoning,
        retrieved_chunk_ids=context.chunk_ids,
        # What the model actually read, attachments included.
        context_chunk_ids=gen_context.chunk_ids,
        strategy=strategy.name,
        stage_timings_ms=timings,
        completed_at=datetime.now(timezone.utc).isoformat(),
    )
