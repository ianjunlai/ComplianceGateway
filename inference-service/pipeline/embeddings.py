"""Local embedding model.

bge-large-en-v1.5 via sentence-transformers; loaded once per process.
Online audit queries are embedded locally and never leave the machine.
"""
from functools import lru_cache

import config


@lru_cache(maxsize=1)
def get_model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(config.EMBEDDING_MODEL)


def embed(texts: list[str]) -> list[list[float]]:
    model = get_model()
    # bge models: no query prefix needed for symmetric similarity at this scale
    return model.encode(texts, normalize_embeddings=True).tolist()


def embed_one(text: str) -> list[float]:
    return embed([text])[0]
