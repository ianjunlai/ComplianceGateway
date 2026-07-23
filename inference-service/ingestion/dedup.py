"""Entity deduplication across chunks and documents.

Semantic-similarity merge: entities whose embedding cosine similarity exceeds
config.DEDUP_THRESHOLD collapse into one canonical node. This is where
cross-document alignment happens — "data controller" in GDPR and in a
university policy become a single graph node.
"""
from dataclasses import dataclass, field

import numpy as np

import config
from pipeline.embeddings import embed


@dataclass
class CanonicalEntity:
    node_id: str
    name: str            # canonical (first-seen) name
    type: str
    aliases: set[str] = field(default_factory=set)
    chunk_ids: set[str] = field(default_factory=set)
    # chunk_id -> number of mentions in that chunk. Feeds the node-passage
    # count matrix used to score passages from node scores.
    chunk_counts: dict[str, int] = field(default_factory=dict)


def deduplicate_entities(
    raw_entities: list[dict],  # [{"name","type","chunk_id"}]
) -> tuple[list[CanonicalEntity], dict[str, str], list[dict]]:
    """Greedy merge by embedding similarity.

    Returns (canonical entities, alias name -> node_id map for relation
    rewiring, merge log). The merge log records every cross-name merge for
    MANDATORY human audit: legally distinct but semantically close terms
    ("data controller" vs "data processor") can exceed the threshold, and
    merging them would corrupt the compliance graph.
    O(n^2) similarity is fine at this corpus scale (~a few hundred entities).
    """
    if not raw_entities:
        return [], {}, []

    names = [e["name"] for e in raw_entities]
    vectors = np.array(embed(names))

    canonicals: list[CanonicalEntity] = []
    canonical_vecs: list[np.ndarray] = []
    name_to_node: dict[str, str] = {}
    merge_log: list[dict] = []

    for i, ent in enumerate(raw_entities):
        vec = vectors[i]
        merged = False
        if canonical_vecs:
            sims = np.array(canonical_vecs) @ vec  # embeddings are normalized
            best = int(np.argmax(sims))
            if sims[best] >= config.DEDUP_THRESHOLD:
                canon = canonicals[best]
                if ent["name"] != canon.name:
                    # cross-name merge: the risky kind — record for human audit
                    merge_log.append({
                        "alias": ent["name"],
                        "alias_type": ent.get("type", "CONCEPT"),
                        "canonical": canon.name,
                        "canonical_type": canon.type,
                        "similarity": round(float(sims[best]), 4),
                    })
                canon.aliases.add(ent["name"])
                canon.chunk_ids.add(ent["chunk_id"])
                canon.chunk_counts[ent["chunk_id"]] = canon.chunk_counts.get(ent["chunk_id"], 0) + 1
                name_to_node[ent["name"]] = canon.node_id
                merged = True
        if not merged:
            node_id = f"ent-{len(canonicals):04d}"
            canon = CanonicalEntity(
                node_id=node_id,
                name=ent["name"],
                type=ent.get("type", "CONCEPT"),
                aliases={ent["name"]},
                chunk_ids={ent["chunk_id"]},
                chunk_counts={ent["chunk_id"]: 1},
            )
            canonicals.append(canon)
            canonical_vecs.append(vec)
            name_to_node[ent["name"]] = node_id

    return canonicals, name_to_node, merge_log
