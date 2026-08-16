"""Shared Neo4j driver."""
from functools import lru_cache

from neo4j import Driver, GraphDatabase

import config


@lru_cache(maxsize=1)
def get_driver() -> Driver:
    return GraphDatabase.driver(
        config.NEO4J_URI, auth=(config.NEO4J_USER, config.NEO4J_PASSWORD)
    )


def index_score_to_cosine(score: float) -> float:
    """Convert a Neo4j vector-index score back to plain cosine similarity."""
    return 2.0 * score - 1.0
