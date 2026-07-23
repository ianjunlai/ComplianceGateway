"""Strategy-pattern base for the five retrieval paradigms.

One pipeline, interchangeable retrieval strategies:
    zero_shot | vector_rag | hybrid | light_rag | hippo_rag
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    score: float = 0.0


@dataclass
class RetrievedContext:
    """Common retrieval output. chunk IDs are the normalized retrieval unit
    — HippoRAG maps PPR node scores back to provenance chunks."""
    chunks: list[RetrievedChunk] = field(default_factory=list)
    graph_nodes: list[str] = field(default_factory=list)
    graph_edges: list[str] = field(default_factory=list)

    @property
    def chunk_ids(self) -> list[str]:
        return [c.chunk_id for c in self.chunks]

    def truncated(self, k: int) -> "RetrievedContext":
        """Copy limited to the top-k chunks (graph context unchanged)."""
        return RetrievedContext(
            chunks=self.chunks[:k],
            graph_nodes=self.graph_nodes,
            graph_edges=self.graph_edges,
        )

    def to_prompt_block(self) -> str:
        """Serialize subgraph + chunks into the structured context block
        that precedes the audit query in the generation prompt."""
        parts: list[str] = []
        if self.graph_nodes:
            parts.append("## Related legal entities\n" + "\n".join(f"- {n}" for n in self.graph_nodes))
        if self.graph_edges:
            parts.append("## Relationships\n" + "\n".join(f"- {e}" for e in self.graph_edges))
        if self.chunks:
            clauses = "\n\n".join(f"[{c.chunk_id}]\n{c.text}" for c in self.chunks)
            parts.append("## Legal clauses\n" + clauses)
        return "\n\n".join(parts) if parts else "(no retrieved context)"


class RetrievalStrategy(ABC):
    """One of the five interchangeable retrieval strategies."""

    name: str = "base"

    @abstractmethod
    def retrieve(self, query: str, seed_entities: list[str], top_k: int) -> RetrievedContext:
        """Return the legal context for the audit query.

        seed_entities: entity mentions extracted from the query by the local
        SLM (zero-shot NER); linking to graph nodes is strategy-specific.
        """
        raise NotImplementedError
