import config
from pipeline.embeddings import embed
from pipeline.graph import get_driver, index_score_to_cosine

_LINK_QUERY = """
UNWIND range(0, size($vectors) - 1) AS i
CALL db.index.vector.queryNodes($index, $k, $vectors[i]) YIELD node, score
RETURN DISTINCT node.node_id AS node_id, score
ORDER BY score DESC
"""


def link_entities(mentions: list[str]) -> list[str]:
    """Return the best-matching graph node(s) for each mention."""
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
