from pipeline.base import RetrievalStrategy, RetrievedContext


class ZeroShotStrategy(RetrievalStrategy):
    name = "zero_shot"

    def retrieve(self, query: str, seed_entities: list[str], top_k: int,
                 allowed_jurisdictions: list[str] | None = None) -> RetrievedContext:
        return RetrievedContext()  # deliberately empty
