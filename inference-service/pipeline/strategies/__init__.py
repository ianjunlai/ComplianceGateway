"""Factory for the retrieval strategies.

Five reproduce published paradigms (zero_shot, vector_rag, hybrid, light_rag,
hippo_rag); five more, the vec_* family, hold the entry mechanism constant at
vector search and vary only which edges the one-hop expansion follows, which is
what isolates edge provenance from entry quality.
"""
from pipeline.base import RetrievalStrategy

VECTOR_EXPAND_VARIANTS = ("vec_relates", "vec_cites", "vec_implements",
                          "vec_intra", "vec_both", "vec_juris")


def build_strategy(name: str) -> RetrievalStrategy:
    if name in VECTOR_EXPAND_VARIANTS:
        from pipeline.strategies.vector_expand import VectorExpandStrategy

        return VectorExpandStrategy(name)
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
