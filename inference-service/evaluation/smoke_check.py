# -*- coding: utf-8 -*-
"""Pre-flight for the long run: exercise every code path once, cheaply."""
import sys
import time
from pathlib import Path

_SERVICE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_SERVICE))

import config                                                    # noqa: E402
from common.schemas import AuditRequestEvent                     # noqa: E402
from pipeline.attachment import attach_citations                 # noqa: E402
from pipeline.jurisdiction import jurisdictions_for              # noqa: E402
from pipeline.strategies import build_strategy                   # noqa: E402

FAIL: list[str] = []
WARN: list[str] = []

# Written the way a request would arrive: no institution named, no clause
# numbers. The requester is carried by source_system.
PROBES = [
    ("cambridge", "May student academic transcripts be transferred to a partner "
                  "university outside the UK using standard contractual clauses?"),
    ("goettingen", "May staff health data be processed to assess fitness for work "
                   "after a long absence?"),
    ("tcd", "May a former student object to the continued processing of their "
            "records held for a research study?"),
]


def check(ok: bool, msg: str, warn_only: bool = False) -> bool:
    print(("  ok    " if ok else ("  WARN  " if warn_only else "  FAIL  ")) + msg)
    if not ok:
        (WARN if warn_only else FAIL).append(msg)
    return ok


def main() -> None:
    print("configuration")
    print(f"  SLM              {config.SLM_MODEL}")
    print(f"  num_ctx          {config.SLM_NUM_CTX}")
    print(f"  embeddings       {config.EMBEDDING_MODEL}")
    print(f"  retrieval K      {config.RETRIEVAL_K}, "
          f"generation K {config.GENERATION_CONTEXT_K}")
    print(f"  attachment       on={config.ATTACH_CITATIONS} "
          f"per_chunk={config.ATTACH_PER_CHUNK} cap={config.ATTACH_CONTEXT_CAP}")
    print(f"  neo4j            {config.NEO4J_URI}")

    # --- graph ------------------------------------------------------------
    print("\ngraph")
    from pipeline.graph import get_driver
    with get_driver().session() as s:
        n_chunk = s.run("MATCH (c:Chunk) RETURN count(c) AS n").single()["n"]
        n_juris = s.run("MATCH (c:Chunk) WHERE c.jurisdiction IS NOT NULL "
                        "RETURN count(c) AS n").single()["n"]
        n_cite = s.run("MATCH ()-[r:CITES|IMPLEMENTS]->() "
                       "RETURN count(r) AS n").single()["n"]
    check(n_chunk > 0, f"{n_chunk} chunks")
    # Without jurisdiction on the nodes every scoped query matches nothing and
    # every strategy returns an empty context, for all 100 questions.
    check(n_juris == n_chunk, f"{n_juris}/{n_chunk} carry a jurisdiction")
    check(n_cite > 0, f"{n_cite} citation edges (attachment has something to follow)")

    # --- retrieval under scope --------------------------------------------
    print("\nretrieval, scoped to the requesting institution")
    scope = jurisdictions_for("cambridge")
    check(scope == ["EU", "UK"], f"cambridge maps to {scope}")
    ctxs = {}
    for name in ("vector_rag", "hybrid", "light_rag", "hippo_rag"):
        try:
            t0 = time.perf_counter()
            ctx = build_strategy(name).retrieve(PROBES[0][1], ["personal data", "transfer"],
                                                config.RETRIEVAL_K,
                                                allowed_jurisdictions=scope)
            ms = (time.perf_counter() - t0) * 1000
        except Exception as e:                                   # noqa: BLE001
            check(False, f"{name}: {type(e).__name__}: {str(e)[:70]}")
            continue
        ctxs[name] = ctx
        check(len(ctx.chunks) > 0, f"{name}: {len(ctx.chunks)} chunks in {ms:.0f} ms")

    # Everything returned must be in scope. A leak here means one strategy is
    # answering from a wider corpus than the others.
    if ctxs:
        with get_driver().session() as s:
            jur = {r["c"]: r["j"] for r in s.run(
                "MATCH (c:Chunk) RETURN c.chunk_id AS c, c.jurisdiction AS j")}
        for name, ctx in ctxs.items():
            bad = [c for c in ctx.chunk_ids if jur.get(c) not in scope]
            check(not bad, f"{name}: nothing out of scope"
                           + (f" (leaked {bad[:3]})" if bad else ""))

    # --- attachment --------------------------------------------------------
    print("\ncitation attachment")
    fired = 0
    for name, ctx in ctxs.items():
        base = ctx.truncated(config.GENERATION_CONTEXT_K)
        grown = attach_citations(base, scope)
        added = len(grown.chunks) - len(base.chunks)
        fired += added > 0
        labelled = sum(1 for c in grown.chunks if c.attached_to)
        print(f"    {name:<12}{len(base.chunks)} -> {len(grown.chunks)} chunks "
              f"({added} attached, {labelled} labelled)")
        check(len(grown.chunks) <= config.ATTACH_CONTEXT_CAP,
              f"{name}: context within the cap")
    check(fired > 0, "attachment fires on at least one strategy")

    # --- the model ---------------------------------------------------------
    print("\nlocal model")
    from pipeline.generation import generate_decision
    from pipeline.base import RetrievedChunk, RetrievedContext

    t0 = time.perf_counter()
    grounded = generate_decision(
        PROBES[0][1],
        RetrievedContext(chunks=[
            RetrievedChunk("uk-dpa-s17C",
                           "A transfer of personal data to a third country is "
                           "permitted where it is based on standard data "
                           "protection clauses specified by the Secretary of State."),
            RetrievedChunk("gdpr-art-46",
                           "A controller may transfer personal data to a third "
                           "country only if it has provided appropriate safeguards.",
                           attached_to="uk-dpa-s17C")]))
    ms = (time.perf_counter() - t0) * 1000
    check(grounded.decision in ("APPROVE", "DENY"),
          f"decides on grounded context: {grounded.decision} in {ms:.0f} ms")
    check("[" in grounded.reasoning,
          "cites clause ids in the reasoning", warn_only=True)
    print(f"      {grounded.reasoning[:110]}")

    # The 25 unanswerable questions rest entirely on this.
    empty = generate_decision(
        "How many years must attendance records be retained?", RetrievedContext())
    check(empty.decision == "UNKNOWN",
          f"abstains on empty context: {empty.decision}")
    print(f"      {empty.reasoning[:110]}")

    # --- end to end --------------------------------------------------------
    print("\nfull pipeline")
    from pipeline.pipeline import run_pipeline
    config.ACTIVE_STRATEGY = "vector_rag"
    for src, q in PROBES:
        try:
            r = run_pipeline(AuditRequestEvent(
                request_id=f"smoke-{src}", source_system=src,
                timestamp="2026-01-01T00:00:00Z", audit_query=q))
        except Exception as e:                                   # noqa: BLE001
            check(False, f"{src}: {type(e).__name__}: {str(e)[:70]}")
            continue
        grew = len(r.context_chunk_ids) - min(config.GENERATION_CONTEXT_K,
                                              len(r.retrieved_chunk_ids))
        check(bool(r.retrieved_chunk_ids) and r.decision != "ERROR",
              f"{src}: {r.decision}, retrieved {len(r.retrieved_chunk_ids)}, "
              f"context {len(r.context_chunk_ids)} (+{grew}), "
              f"{r.stage_timings_ms.get('total_ms')} ms")

    print()
    if FAIL:
        print(f"{len(FAIL)} FAILURE(S) — do not start the run:")
        for m in FAIL:
            print(f"  - {m}")
        sys.exit(1)
    if WARN:
        print(f"{len(WARN)} warning(s), not blocking:")
        for m in WARN:
            print(f"  - {m}")
    print("ready")


if __name__ == "__main__":
    main()
