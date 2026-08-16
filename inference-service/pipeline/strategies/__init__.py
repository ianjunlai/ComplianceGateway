"""Factory for the retrieval strategies.

Five conditions are evaluated: zero_shot as the no-retrieval baseline,
vector_rag as dense retrieval, and hybrid, light_rag and hippo_rag reproducing
the three published GraphRAG paradigms.
"""
from pipeline.base import RetrievalStrategy


def build_strategy(name: str) -> RetrievalStrategy:
    if name == "zero_shot":
        from pipeline.strategies.zero_shot import ZeroShotStrategy

        return ZeroShotStrategy()
    if name == "vector_rag":
        from pipeline.strategies.vector_rag import VectorRagStrategy

        return VectorRagStrategy()
    if name == "hybrid":
        from pipeline.strategies.hybrid_graph import HybridGraphStrategy

        return HybridGraphStrategy()
    if name == "light_rag":
        from pipeline.strategies.light_rag import LightRagStrategy

        return LightRagStrategy()
    if name == "hippo_rag":
        from pipeline.strategies.hippo_rag import HippoRagStrategy

        return HippoRagStrategy()
    raise ValueError(f"Unknown strategy: {name}")
