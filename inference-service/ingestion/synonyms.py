"""Synonymy edges between near-identical entity names.

HippoRAG expresses "these two mentions are probably the same thing" as an extra
edge rather than as a merge: nodes stay distinct, but probability can flow
between them during PPR. The paper calls these E' and adds one whenever the
cosine similarity of two entity representations exceeds tau, tuned to 0.8.

They are not a detail. On 2WikiMultiHopQA the paper reports 7,867 relation
phrases across 50,671 triples against 82,526 synonymy edges: the graph PPR
actually walks is mostly synonymy. They are what connects a passage that says
"Ken Annakin" to one that says "Kenneth Cooper Annakin" without the two
becoming one node -- which is what makes the difference between a bridge the
walk can cross and a distinction the corpus needed to keep.

Kept as a separate relationship type so only HippoRAG sees them: Hybrid's
traversal and LightRAG's expansion are defined over extracted relations, and
silently widening their neighbourhoods would change what those methods are.
"""
import logging

import numpy as np

import config

log = logging.getLogger("synonyms")

# Pairwise cosine over every entity at once is |N|^2 floats; at 8k entities
# that is 256 MB, and it grows quadratically. Blocking keeps peak memory flat
# and costs nothing measurable.
_BLOCK = 1024


def build_synonym_edges(vectors: np.ndarray, threshold: float | None = None
                        ) -> list[tuple[int, int, float]]:
    """Index pairs (i, j) with i < j judged synonymous.

    Vectors are assumed L2-normalised, as sentence-transformers returns them,
    so the dot product is the cosine.

    The threshold is derived from a target edge density rather than fixed,
    because a cosine cutoff transfers across neither encoder nor corpus. The
    paper's 0.8 was tuned with ColBERTv2; reaching its published density of
    1.93 edges per entity needs 0.726 on bge-large over 2WikiMultihopQA and
    0.786 over the GDPR corpus -- legal vocabulary is far more self-similar
    than encyclopaedic proper nouns, so the same cutoff means something
    different in each. Density is the quantity that carries over, so density is
    the parameter; set SYNONYM_THRESHOLD to pin the cutoff instead.
    """
    n = len(vectors)
    if n < 2:
        return []
    mat = np.asarray(vectors, dtype=np.float32)

    # One pass to collect candidate similarities, a second to emit the edges.
    # Cheaper than it looks: the floor discards almost everything, and the
    # alternative is guessing a cutoff.
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
