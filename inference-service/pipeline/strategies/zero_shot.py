"""Zero-Shot condition.

No retrieval at all: the SLM answers from parametric knowledge only.
Faithfulness for this condition is judged against GOLD chunks, not retrieved
context.
"""
from pipeline.base import RetrievalStrategy, RetrievedContext


class ZeroShotStrategy(RetrievalStrategy):
    name = "zero_shot"

    def retrieve(self, query: str, seed_entities: list[str], top_k: int) -> RetrievedContext:
        return RetrievedContext()  # deliberately empty
