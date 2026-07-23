"""Entity linking: query mentions -> graph node IDs.

Exact string match between SLM-extracted mentions and graph node names fails
almost always ("personal data transfer" vs node "Data Transfer"), so linking
is embedding-based: cosine similarity against the entity vector index with
disclosed threshold tau = config.ENTITY_LINK_THRESHOLD.
"""
import config
from pipeline.embeddings import embed
from pipeline.graph import get_driver, index_score_to_cosine

# All mentions are matched in one round trip; scores are deduplicated per node.
_LINK_QUERY = """
UNWIND $vectors AS vec
CALL db.index.vector.queryNodes($index, $limit, vec) YIELD node, score
WITH node.node_id AS node_id, max(score) AS score
RETURN node_id, score
ORDER BY score DESC
"""


def link_entities(mentions: list[str], limit_per_mention: int = 3) -> list[str]:
    """Return graph node IDs whose embedding similarity to any mention >= tau."""
    if not mentions:
        return []
    with get_driver().session() as session:
        records = session.run(
            _LINK_QUERY,
            vectors=embed(mentions),
            index=config.INDEX_ENTITIES,
            limit=limit_per_mention,
        )
        return [
            r["node_id"]
            for r in records
            if index_score_to_cosine(r["score"]) >= config.ENTITY_LINK_THRESHOLD
        ]
