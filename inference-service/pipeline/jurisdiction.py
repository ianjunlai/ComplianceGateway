# -*- coding: utf-8 -*-
"""Which bodies of law bind the institution that sent a request, read from the
request envelope rather than from the question text."""
from functools import lru_cache

# EU law is added to every scope: the Regulation binds all of them.
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
    """chunk_id -> jurisdiction, loaded once from the graph."""
    from pipeline.graph import get_driver

    with get_driver().session() as session:
        return {r["chunk_id"]: r["jurisdiction"] for r in session.run(
            "MATCH (c:Chunk) RETURN c.chunk_id AS chunk_id, "
            "c.jurisdiction AS jurisdiction")}
