"""Entity deduplication across chunks and documents.

Semantic-similarity merge: entities whose embedding cosine similarity exceeds
config.DEDUP_THRESHOLD collapse into one canonical node. This is where
cross-document alignment happens — "data controller" in GDPR and in a
university policy become a single graph node.

Legal citations ("Article 6(1)(a)") are the one category exempted from
similarity-based merging: they are identifiers, not free-text concepts, and
near-identical embeddings do not imply the same legal meaning -- Article
6(1)(a) (consent) and 6(1)(b) (contract) are different lawful bases for
processing, but a generic sentence embedding puts them a hair's breadth apart
since almost the entire string is shared. Citation-like names are only merged
on an exact (whitespace/case-normalized) match.
"""
import re
from dataclasses import dataclass, field

import numpy as np

import config
from pipeline.embeddings import embed

_CITATION_RE = re.compile(r"^(article|art\.?)\s+\d", re.IGNORECASE)


def _is_citation(name: str) -> bool:
    return bool(_CITATION_RE.match(name.strip()))


def _normalize_citation(name: str) -> str:
    return re.sub(r"\s+", " ", name.strip().lower())


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
    """Greedy merge by embedding similarity (citations: exact match only).

    Returns (canonical entities, alias name -> node_id map for relation
    rewiring, merge log). The merge log has one row per distinct (alias,
    canonical) merge DECISION, sorted rarest-first, for MANDATORY human audit:
    legally distinct but semantically close terms ("data controller" vs
    "data processor") can exceed the threshold, and merging them would
    corrupt the compliance graph. A term merging the same way in many chunks
    only adds an "occurrences" count, not a repeated row — otherwise routine,
    high-frequency merges would bury the rare, risky ones.
    O(n^2) similarity is fine at this corpus scale (~a few hundred entities).
    """
    if not raw_entities:
        return [], {}, []

    names = [e["name"] for e in raw_entities]
    vectors = np.array(embed(names))

    canonicals: list[CanonicalEntity] = []
    canonical_vecs: list[np.ndarray] = []
    name_to_node: dict[str, str] = {}
    merge_log_by_pair: dict[tuple[str, str], dict] = {}
    # normalized citation -> index into canonicals; exact-match fast path
    citation_index: dict[str, int] = {}

    def merge_into(canon: CanonicalEntity, ent: dict, similarity: float) -> None:
        if ent["name"] != canon.name:
            # cross-name merge: the risky kind — record for human audit
            key = (ent["name"], canon.name)
            row = merge_log_by_pair.get(key)
            if row is None:
                merge_log_by_pair[key] = {
                    "alias": ent["name"],
                    "alias_type": ent.get("type", "CONCEPT"),
                    "canonical": canon.name,
                    "canonical_type": canon.type,
                    "similarity": round(similarity, 4),
                    "occurrences": 1,
                }
            else:
                row["occurrences"] += 1
        canon.aliases.add(ent["name"])
        canon.chunk_ids.add(ent["chunk_id"])
        canon.chunk_counts[ent["chunk_id"]] = canon.chunk_counts.get(ent["chunk_id"], 0) + 1
        name_to_node[ent["name"]] = canon.node_id

    for i, ent in enumerate(raw_entities):
        vec = vectors[i]
        merged = False
        is_citation = _is_citation(ent["name"])

        if is_citation:
            existing = citation_index.get(_normalize_citation(ent["name"]))
            if existing is not None:
                merge_into(canonicals[existing], ent, similarity=1.0)
                merged = True
        elif canonical_vecs:
            sims = np.array(canonical_vecs) @ vec  # embeddings are normalized
            best = int(np.argmax(sims))
            if sims[best] >= config.DEDUP_THRESHOLD and not _is_citation(canonicals[best].name):
                merge_into(canonicals[best], ent, similarity=float(sims[best]))
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
            if is_citation:
                citation_index[_normalize_citation(ent["name"])] = len(canonicals) - 1

    # Rarest merges first: a one-off borderline merge is exactly what the
    # audit should surface before scrolling past dozens of routine ones.
    merge_log = sorted(merge_log_by_pair.values(), key=lambda r: r["occurrences"])
    return canonicals, name_to_node, merge_log
