# -*- coding: utf-8 -*-
"""Vector entry, then one hop along graph edges — five variants of edge provenance.

These exist to answer a question the three published paradigms cannot: does it
matter WHERE a graph's edges come from? Measured on this corpus, one hop over
LLM-extracted entity relations from five vector entry points admits 410 of 414
chunks, which is no selection at all, while a hop along citations the drafters
actually wrote admits a small fraction.

Testing that needs the entry mechanism held constant. Comparing `hybrid`
against a citation walk would vary two things at once — hybrid enters through
NER seeds and entity linking, and that entry is known to be lossy here (a 1B
NER model, and switching to the paper's argmax linking cost 12 points of R@5).
So all five variants enter by vector search and differ only in which edges
they follow:

    vec_relates     entity co-mention edges, one hop through the entity layer
    vec_cites       every citation, both kinds
    vec_implements  cross-tier citations only (national -> GDPR)
    vec_intra       same-instrument citations only
    vec_both        entity edges and citations together

Ranking is query similarity over the admitted union in every case, so an
expanded chunk competes for its place rather than being appended.

A sixth, vec_juris, varies the ENTRY instead of the edges: same citation
expansion as vec_cites, but the entry set is restricted to EU law plus the
state the requesting university sits in. Report it separately from the five
above and do not call the result a property of the graph -- a jurisdiction
filter is metadata selection, reproducible on a plain vector store with a
`WHERE jurisdiction IN [...]` clause and no graph at all. What it measures is
whether the three-tier corpus's cross-jurisdiction distractors (24 of 49
implemented GDPR articles are implemented by more than one state) are what
dense retrieval is losing to. That is worth knowing, and it is a different
claim from "the graph helped".
"""
import config
from pipeline.base import RetrievalStrategy, RetrievedChunk, RetrievedContext
from pipeline.embeddings import embed_one
from pipeline.graph import get_driver, index_score_to_cosine

# Entry: nearest chunks to the question. Over-fetched for the same reason
# vector_rag over-fetches -- Neo4j's vector index is approximate, and a small-k
# query explores less of the HNSW graph than a large-k one.
_ENTRY = """
CALL db.index.vector.queryNodes($index, $k, $qvec) YIELD node
RETURN node.chunk_id AS chunk_id
"""

# Jurisdiction-filtered entry. Neo4j 5.24 community has no pre-filter on the
# vector index, so this over-fetches and filters after -- the same shape as
# vector_rag's over-fetch, with a larger multiplier because the filter discards
# most of what comes back.
_ENTRY_JURIS = """
CALL db.index.vector.queryNodes($index, $k, $qvec) YIELD node
WITH node WHERE node.jurisdiction IN $allowed
RETURN node.chunk_id AS chunk_id
LIMIT $entry_k
"""

# Which university sits in which state. The question names the institution in
# prose, so the jurisdiction has to be read off the text -- that is what a real
# gateway would do with the requesting system's identity, and it is the only
# signal available through the RetrievalStrategy interface.
_INSTITUTION_JURISDICTION = {
    "trinity college dublin": "IE",
    "university of limerick": "IE",
    "university of cambridge": "UK",
    "göttingen": "DE",
    "goettingen": "DE",
}

# One hop out of the entry set, by edge type, then rank everything admitted.
# %(expand)s is interpolated from a fixed table below, never from user input.
_EXPAND = """
UNWIND $entry AS cid
MATCH (c:Chunk {chunk_id: cid})
%(expand)s
WITH collect(DISTINCT c) + collect(DISTINCT linked) AS pool
UNWIND pool AS n
WITH DISTINCT n WHERE n IS NOT NULL
RETURN n.chunk_id AS chunk_id, n.text AS text,
       vector.similarity.cosine(n.embedding, $qvec) AS score
ORDER BY score DESC
LIMIT $limit
"""

# An entity-mediated hop: chunk -> entity it mentions -> related entity -> the
# chunks that mention it. One hop of RELATES, matching what the citation
# variant gets, so the two differ in edge provenance and not in depth.
_VIA_RELATES = """
OPTIONAL MATCH (c)<-[:MENTIONED_IN]-(:Entity)-[:RELATES]-(:Entity)-[:MENTIONED_IN]->(linked:Chunk)
"""
# The two citation types behave differently and are separable, which is the
# reason load_implements keeps them apart. IMPLEMENTS crosses a tier; CITES
# stays inside one instrument. On cross-tier questions the same-instrument
# edges pull in neighbouring provisions that score well on similarity without
# being gold, so they displace the cross-tier target rather than adding to it:
# on the three-tier corpus, mixing them in costs R@10 0.367 -> 0.300 and
# cross-tier completeness 0.333 -> 0.187.
_VIA_CITES = """
OPTIONAL MATCH (c)-[:CITES|IMPLEMENTS]-(linked:Chunk)
"""
_VIA_IMPLEMENTS = """
OPTIONAL MATCH (c)-[:IMPLEMENTS]-(linked:Chunk)
"""
_VIA_INTRA = """
OPTIONAL MATCH (c)-[:CITES]-(linked:Chunk)
"""
_VIA_BOTH = """
OPTIONAL MATCH (c)-[:CITES|IMPLEMENTS]-(direct:Chunk)
OPTIONAL MATCH (c)<-[:MENTIONED_IN]-(:Entity)-[:RELATES]-(:Entity)-[:MENTIONED_IN]->(viaent:Chunk)
WITH c, collect(DISTINCT direct) + collect(DISTINCT viaent) AS both
UNWIND (CASE WHEN size(both) = 0 THEN [null] ELSE both END) AS linked
"""

_EXPANSIONS = {
    "vec_relates": _VIA_RELATES,        # LLM-inferred entity co-mention
    "vec_cites": _VIA_CITES,            # every citation, both kinds
    "vec_implements": _VIA_IMPLEMENTS,  # cross-tier citations only
    "vec_intra": _VIA_INTRA,            # same-instrument citations only
    "vec_both": _VIA_BOTH,              # entity edges and citations together
    "vec_juris": _VIA_CITES,            # citations, but entry filtered by jurisdiction
}

# Entry over-fetch for the filtered variant. Three of four national provisions
# are the wrong state's, so a plain top-5 would often contain no admissible
# chunk at all.
_JURIS_OVERFETCH = 20


class VectorExpandStrategy(RetrievalStrategy):
    """One class, five registered names; the edge type is the only difference."""

    def __init__(self, name: str) -> None:
        if name not in _EXPANSIONS:
            raise ValueError(f"Unknown expansion variant: {name}")
        self.name = name
        self._expand = _EXPANSIONS[name]

    def retrieve(self, query: str, seed_entities: list[str], top_k: int) -> RetrievedContext:
        # seed_entities is accepted for interface parity and deliberately
        # unused: entry is by vector search, which is the control that lets the
        # three variants be compared on edge provenance alone.
        qvec = embed_one(query)
        entry_k = config.VECTOR_EXPAND_ENTRY_K
        with get_driver().session() as session:
            if self.name == "vec_juris":
                # EU law binds every institution, so the regional tier is always
                # admissible; only the national and institutional tiers are
                # narrowed to the state the requesting university sits in.
                allowed = ["EU"]
                lowered = query.lower()
                for needle, code in _INSTITUTION_JURISDICTION.items():
                    if needle in lowered:
                        allowed.append(code)
                        break
                entry = [r["chunk_id"] for r in session.run(
                    _ENTRY_JURIS, index=config.INDEX_CHUNKS,
                    k=entry_k * _JURIS_OVERFETCH, qvec=qvec,
                    allowed=allowed, entry_k=entry_k)]
            else:
                entry = [r["chunk_id"] for r in session.run(
                    _ENTRY, index=config.INDEX_CHUNKS, k=entry_k, qvec=qvec)]
            if not entry:
                return RetrievedContext()
            records = session.run(
                _EXPAND % {"expand": self._expand},
                entry=entry, qvec=qvec, limit=top_k)
            chunks = [
                RetrievedChunk(chunk_id=r["chunk_id"], text=r["text"],
                               score=index_score_to_cosine(r["score"]))
                for r in records
            ]
        return RetrievedContext(chunks=chunks[:top_k])
