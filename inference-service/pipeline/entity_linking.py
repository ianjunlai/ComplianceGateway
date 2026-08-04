"""Entity linking: query mentions -> graph node IDs.

Exact string match between SLM-extracted mentions and graph node names fails
almost always ("personal data transfer" vs node "Data Transfer"), so linking
is embedding-based: cosine similarity against the entity vector index.

One node per mention, unconditionally, which is what HippoRAG specifies:
"query nodes are defined as Rq = {r1, ..., rn} such that ri = ek where
k = argmax_j cosine_similarity(M(ci), M(ej))". Two things follow from that
being an argmax rather than a top-k above a threshold, and both were measured
to matter here:

  * Seeds share the PPR restart mass equally, so returning three nodes per
    mention instead of one dilutes every genuine seed. On the 2Wiki benchmark
    the old behaviour produced 5.6 seeds per query against the paper's 3.0.
  * A threshold silently drops a query entity whose best match is merely
    imperfect. Measured at tau=0.75, 13% of query entities linked to nothing
    at all -- and for a two-hop question, losing one anchor breaks the chain
    rather than weakening it.
"""
import config
from pipeline.embeddings import embed
from pipeline.graph import get_driver, index_score_to_cosine

# One round trip for all mentions. Each row is one mention's best node, so a
# mention that happens to share its argmax with another contributes once.
_LINK_QUERY = """
UNWIND range(0, size($vectors) - 1) AS i
CALL db.index.vector.queryNodes($index, $k, $vectors[i]) YIELD node, score
RETURN DISTINCT node.node_id AS node_id, score
ORDER BY score DESC
"""


def link_entities(mentions: list[str]) -> list[str]:
    """Return the best-matching graph node(s) for each mention.

    config.ENTITY_LINK_TOP_K = 1 is the paper's argmax and the default. Larger
    values also apply ENTITY_LINK_THRESHOLD, reproducing the earlier behaviour
    so the ablation can be run rather than argued about.
    """
    if not mentions:
        return []
    k = config.ENTITY_LINK_TOP_K
    with get_driver().session() as session:
        records = list(session.run(
            _LINK_QUERY, vectors=embed(mentions), index=config.INDEX_ENTITIES, k=k))
    if k <= 1:
        return [r["node_id"] for r in records]
    return [r["node_id"] for r in records
            if index_score_to_cosine(r["score"]) >= config.ENTITY_LINK_THRESHOLD]
