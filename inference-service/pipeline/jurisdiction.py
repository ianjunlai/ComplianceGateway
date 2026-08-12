# -*- coding: utf-8 -*-
"""Which bodies of law bind the institution that sent a request.

A request from a UK university is governed by EU law and UK law. Irish and
German provisions are not merely less relevant to it, they are inapplicable:
no reading of them can answer the question. Retrieving them wastes context and
introduces near-duplicate distractors, because several member states implement
the same GDPR article in similar language.

The gateway knows who sent the request from the envelope, so the jurisdiction
comes from `source_system` rather than from parsing the question text. That is
both more realistic and more robust than reading an institution name out of
prose.

An unrecognised source returns None, which every caller treats as "no filter".
Warm-up and ad-hoc requests therefore behave as before, and a new institution
that has not been mapped fails open rather than silently retrieving nothing.
"""
from functools import lru_cache

# The four institutions in the corpus. EU law is added to every scope because
# the Regulation binds all of them.
_INSTITUTION_STATE = {
    "tcd": "IE",
    "ul": "IE",
    "cambridge": "UK",
    "goettingen": "DE",
}


def jurisdictions_for(source_system: str | None) -> list[str] | None:
    """Jurisdiction codes a request from this system may be answered from."""
    state = _INSTITUTION_STATE.get((source_system or "").strip().lower())
    return ["EU", state] if state else None


def state_of(source_system: str | None) -> str | None:
    """The single national jurisdiction, or None if the system is unknown."""
    return _INSTITUTION_STATE.get((source_system or "").strip().lower())


@lru_cache(maxsize=1)
def chunk_jurisdictions() -> dict[str, str]:
    """chunk_id -> jurisdiction, loaded once from the graph.

    HippoRAG scores passages with sparse matrix arithmetic and never touches
    the database at query time, so it cannot filter in Cypher like the others.
    It masks out-of-scope columns instead, and needs this map to do it. Read
    from Neo4j rather than from an artifact file so it cannot drift away from
    the graph the run is actually querying.
    """
    from pipeline.graph import get_driver

    with get_driver().session() as session:
        return {r["chunk_id"]: r["jurisdiction"] for r in session.run(
            "MATCH (c:Chunk) RETURN c.chunk_id AS chunk_id, "
            "c.jurisdiction AS jurisdiction")}
