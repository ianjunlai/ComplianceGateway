"""Shared Neo4j access.

One driver (and therefore one connection pool) for every strategy: building a
driver per call would leak connection-setup time into the measured retrieval
stage.
"""
from functools import lru_cache

from neo4j import Driver, GraphDatabase

import config


@lru_cache(maxsize=1)
def get_driver() -> Driver:
    return GraphDatabase.driver(
        config.NEO4J_URI, auth=(config.NEO4J_USER, config.NEO4J_PASSWORD)
    )


def index_score_to_cosine(score: float) -> float:
    """Convert a Neo4j vector-index score back to plain cosine similarity.

    Neo4j reports cosine similarity rescaled to [0, 1] as (1 + cos) / 2.
    Thresholds in config are expressed as plain cosine similarity, so they must
    be compared against the converted value — comparing against the raw index
    score would silently halve the effective threshold.
    """
    return 2.0 * score - 1.0
