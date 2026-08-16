# -*- coding: utf-8 -*-
"""Attach the provisions a retrieved clause cites, instead of ranking them: a cited article is abstract where the query is operational, so it ranks badly."""
import config
from pipeline.base import RetrievedChunk, RetrievedContext
from pipeline.graph import get_driver

# Both directions: the query may land on either end of the citation.
_ATTACH_QUERY = """
UNWIND $chunk_ids AS cid
MATCH (c:Chunk {chunk_id: cid})-[r:CITES|IMPLEMENTS]-(linked:Chunk)
WHERE NOT linked.chunk_id IN $chunk_ids
  AND ($allowed IS NULL OR linked.jurisdiction IN $allowed)
RETURN cid AS cites_from, linked.chunk_id AS chunk_id, linked.text AS text,
       r.resolution AS resolution, type(r) AS edge_type
"""

# Lower sorts first, so the most precisely resolved citation survives the cap.
_RESOLUTION_ORDER = {"exact": 0, "whole": 1, "spread": 2}


def select_attachments(rows: list[dict], order: list[str], per_chunk: int,
                       total_cap: int) -> dict[str, list[dict]]:
    """Choose which cited provisions to attach, grouped by the citing
    clause."""
    by_source: dict[str, list[dict]] = {cid: [] for cid in order}
    for r in rows:
        if r["cites_from"] in by_source:
            by_source[r["cites_from"]].append(r)

    taken: set[str] = set(order)
    budget = total_cap - len(order)
    chosen: dict[str, list[dict]] = {cid: [] for cid in order}
    for cid in order:
        if budget <= 0:
            break
        candidates = sorted(
            by_source[cid],
            key=lambda r: (_RESOLUTION_ORDER.get(r.get("resolution"), 3),
                           r["chunk_id"]),
        )
        for r in candidates:
            if len(chosen[cid]) >= per_chunk or budget <= 0:
                break
            # Cited by two clauses: attach once, to the higher-ranked one.
            if r["chunk_id"] in taken:
                continue
            chosen[cid].append(r)
            taken.add(r["chunk_id"])
            budget -= 1
    return chosen


def attach_citations(context: RetrievedContext,
                     allowed_jurisdictions: list[str] | None = None
                     ) -> RetrievedContext:
    """Copy of `context` with cited provisions inserted after the citing
    clause. The input ranking is preserved."""
    if not config.ATTACH_CITATIONS or not context.chunks:
        return context

    order = [c.chunk_id for c in context.chunks]
    with get_driver().session() as session:
        rows = [dict(r) for r in session.run(
            _ATTACH_QUERY, chunk_ids=order, allowed=allowed_jurisdictions)]
    if not rows:
        return context

    chosen = select_attachments(rows, order, config.ATTACH_PER_CHUNK,
                                config.ATTACH_CONTEXT_CAP)

    merged: list[RetrievedChunk] = []
    for c in context.chunks:
        merged.append(c)
        for r in chosen.get(c.chunk_id, []):
            merged.append(RetrievedChunk(
                chunk_id=r["chunk_id"],
                text=r["text"] or "",
                # Zero, not a similarity: it never competed on one.
                score=0.0,
                attached_to=c.chunk_id,
            ))
    return RetrievedContext(chunks=merged,
                            graph_nodes=context.graph_nodes,
                            graph_edges=context.graph_edges)
