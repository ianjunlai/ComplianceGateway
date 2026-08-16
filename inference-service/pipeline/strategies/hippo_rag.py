import json
import os

import numpy as np
from scipy import sparse

import config
from pipeline.base import RetrievalStrategy, RetrievedChunk, RetrievedContext
from pipeline.entity_linking import link_entities
from pipeline.jurisdiction import chunk_jurisdictions


class HippoRagStrategy(RetrievalStrategy):
    name = "hippo_rag"

    def __init__(self) -> None:
        art = config.ARTIFACTS_DIR
        self.adj: sparse.csr_matrix = sparse.load_npz(os.path.join(art, "hippo_adjacency.npz"))
        self.passage_matrix: sparse.csr_matrix = sparse.load_npz(
            os.path.join(art, "hippo_passage_matrix.npz"))
        with open(os.path.join(art, "hippo_chunk_index.json"), encoding="utf-8") as f:
            chunk_index: dict[str, int] = json.load(f)
        with open(os.path.join(art, "hippo_nodes.json"), encoding="utf-8") as f:
            self.node_index: dict[str, int] = json.load(f)
        with open(os.path.join(art, "hippo_node_chunks.json"), encoding="utf-8") as f:
            self.node_chunks: dict[str, list[str]] = json.load(f)
        with open(os.path.join(art, "chunk_texts.json"), encoding="utf-8") as f:
            self.chunk_texts: dict[str, str] = json.load(f)
        try:
            # node_id -> human-readable name; internal IDs would be pure noise
            # in the generation prompt
            with open(os.path.join(art, "hippo_node_names.json"), encoding="utf-8") as f:
                self.node_names: dict[str, str] = json.load(f)
        except FileNotFoundError:
            self.node_names = {}
        self.index_node = {v: k for k, v in self.node_index.items()}
        self.index_chunk = {v: k for k, v in chunk_index.items()}
        self._masks: dict[tuple[str, ...], np.ndarray] = {}

    def _scope_mask(self, allowed: tuple[str, ...]) -> np.ndarray:
        """Column mask over passages, cached per jurisdiction scope."""
        cached = self._masks.get(allowed)
        if cached is None:
            juris = chunk_jurisdictions()
            cached = np.array(
                [juris.get(self.index_chunk.get(i)) in allowed
                 for i in range(self.passage_matrix.shape[1])],
                dtype=bool)
            self._masks[allowed] = cached
        return cached

    def retrieve(self, query: str, seed_entities: list[str], top_k: int,
                 allowed_jurisdictions: list[str] | None = None) -> RetrievedContext:
        seed_ids = [nid for nid in link_entities(seed_entities) if nid in self.node_index]
        if not seed_ids:
            return RetrievedContext()

        node_scores = self._personalized_pagerank(seed_ids)

        # Project node scores onto passages through the mention-count matrix
        passage_scores = self.passage_matrix.T @ node_scores
        if allowed_jurisdictions is not None:
            # Zero rather than remove, so column indices still line up with
            # index_chunk.
            passage_scores = np.where(
                self._scope_mask(tuple(allowed_jurisdictions)), passage_scores, 0.0)
        top_cols = np.argsort(-passage_scores)[:top_k]
        chunks = [
            RetrievedChunk(
                chunk_id=self.index_chunk[int(j)],
                text=self.chunk_texts.get(self.index_chunk[int(j)], ""),
                score=float(passage_scores[j]),
            )
            for j in top_cols
            if passage_scores[j] > 0
        ]

        # Highest-activation entities, for the generation prompt's graph context
        top_nodes = np.argsort(-node_scores)[:top_k]
        node_names = [
            self.node_names.get(self.index_node[int(i)], self.index_node[int(i)])
            for i in top_nodes
            if node_scores[i] > 0
        ]
        return RetrievedContext(chunks=chunks, graph_nodes=node_names)

    def _node_specificity(self, node_id: str) -> float:
        """1 / (number of passages the entity occurs in); IDF-like weight."""
        occurrences = len(self.node_chunks.get(node_id, []))
        return 1.0 / occurrences if occurrences else 1.0

    def _personalized_pagerank(self, seed_ids: list[str], iters: int = 50, tol: float = 1e-8) -> np.ndarray:
        n = self.adj.shape[0]
        personalization = np.zeros(n)
        for nid in seed_ids:
            personalization[self.node_index[nid]] += self._node_specificity(nid)
        total = personalization.sum()
        if total > 0:
            personalization /= total  # keep it a probability distribution

        dangling = np.asarray(self.adj.sum(axis=1)).ravel() == 0

        scores = personalization.copy()
        # PPR_ALPHA is the probability of following an edge; 1 - PPR_ALPHA is the probability of restarting at a seed node.
        alpha = config.PPR_ALPHA
        for _ in range(iters):
            new = alpha * (self.adj.T @ scores) + (1 - alpha) * personalization
            leaked = alpha * float(scores[dangling].sum())
            if leaked:
                new += leaked * personalization
            if np.abs(new - scores).sum() < tol:
                scores = new
                break
            scores = new
        return scores
