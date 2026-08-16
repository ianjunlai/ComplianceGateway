"""Synonymy edges between near-identical entity names, HippoRAG's E'. Only HippoRAG walks them."""
import logging

import numpy as np

import config

log = logging.getLogger("synonyms")

# Pairwise cosine over every entity at once is |N|^2 floats; at 8k entities that is 256 MB, and it grows quadratically.
_BLOCK = 1024


def build_synonym_edges(vectors: np.ndarray, threshold: float | None = None
                        ) -> list[tuple[int, int, float]]:
    """Index pairs (i, j) with i < j judged synonymous."""
    n = len(vectors)
    if n < 2:
        return []
    mat = np.asarray(vectors, dtype=np.float32)

    # One pass to collect candidate similarities, a second to emit the edges.
    floor = 0.5
    candidates = []
    for start in range(0, n, _BLOCK):
        stop = min(start + _BLOCK, n)
        sims = mat[start:stop] @ mat.T
        for r in range(stop - start):
            row = sims[r, start + r + 1:]       # upper triangle only
            candidates.append(row[row >= floor])
    pool = np.concatenate(candidates) if candidates else np.zeros(0, dtype=np.float32)

    if threshold is not None:
        tau, how = threshold, "pinned"
    elif config.SYNONYM_THRESHOLD is not None:
        tau, how = config.SYNONYM_THRESHOLD, "pinned"
    else:
        target = int(round(config.SYNONYM_EDGES_PER_ENTITY * n))
        if target <= 0 or len(pool) <= target:
            tau, how = floor, "density target unreachable, floor used"
        else:
            pool.sort()
            tau, how = float(pool[len(pool) - target]), "density-matched"

    edges: list[tuple[int, int, float]] = []
    for start in range(0, n, _BLOCK):
        stop = min(start + _BLOCK, n)
        sims = mat[start:stop] @ mat.T
        for r in range(stop - start):
            i = start + r
            row = sims[r, i + 1:]
            for offset in np.where(row >= tau)[0]:
                edges.append((i, i + 1 + int(offset), float(row[offset])))

    log.info("Synonymy: %d edges over %d entities (%.2f per entity) at tau=%.4f (%s)",
             len(edges), n, len(edges) / n, tau, how)
    return edges
