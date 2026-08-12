# -*- coding: utf-8 -*-
"""Attach the provisions a retrieved clause cites, without ranking them.

A provision is often incomplete on its own. A national section gives effect to
a GDPR article; another section qualifies it with "subject to section N". An
answer that never sees the cited provision can be wrong even when the right
clause was retrieved.

Ranking cannot deliver those provisions. The retrieval strategies score every
candidate by similarity to the query, and the cited provision is precisely the
one that is *not* similar: a GDPR article states an abstract rule while the
request is written in operational language. Measured on the first round, when
the cited GDPR article was retrieved at all it sat at median rank 6, one place
below a five-chunk generation window. Ranking demoted exactly what the citation
had supplied.

So citations are attached rather than ranked. A chunk that a retrieved clause
cites enters the context because of the edge, not because it competes well on
similarity, and it is placed immediately after the clause that cites it so the
two read together.

Two caps keep this bounded. At most ATTACH_PER_CHUNK provisions per retrieved
clause, and at most ATTACH_CONTEXT_CAP chunks in the final context. Where a
clause cites more than the per-chunk cap allows, the ones whose citation
resolved most precisely are kept: an `exact` edge points at the paragraph that
was actually cited, `whole` at the article when it has no sub-chunks, and
`spread` is a fallback that points at every paragraph of the cited article and
is therefore the least reliable.
"""
import config
from pipeline.base import RetrievedChunk, RetrievedContext
from pipeline.graph import get_driver

# Both directions. A national provision cites a GDPR article, but the reverse
# lookup matters too: when the query lands on the GDPR article first, the
# national provision implementing it is what the answer is missing.
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
    """Choose which cited provisions to attach, grouped by the clause citing them.

    Pure function over query rows so the capping rules can be tested without a
    database. `order` is the retrieved chunk ids in rank order; earlier clauses
    get first claim on the shared total budget, which keeps the attachment for
    the best-ranked clause from being crowded out by a lower-ranked one.
    """
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
            # Global dedup: a provision cited by two retrieved clauses is
            # attached once, to the higher-ranked one.
            if r["chunk_id"] in taken:
                continue
            chosen[cid].append(r)
            taken.add(r["chunk_id"])
            budget -= 1
    return chosen


def attach_citations(context: RetrievedContext,
                     allowed_jurisdictions: list[str] | None = None
                     ) -> RetrievedContext:
    """Return a copy of `context` with cited provisions inserted after the
    clauses that cite them. The input ranking is preserved."""
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
                # Zero, not a similarity: an attached provision did not compete
                # on similarity and a score here would imply it had.
                score=0.0,
                attached_to=c.chunk_id,
            ))
    return RetrievedContext(chunks=merged,
                            graph_nodes=context.graph_nodes,
                            graph_edges=context.graph_edges)
