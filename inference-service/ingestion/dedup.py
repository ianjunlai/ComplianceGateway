"""Entity deduplication across chunks and documents.

Merges entities whose names are the same after normalising case and whitespace,
and nothing else. This is what both source papers specify: LightRAG's Dedupe
function "identifies and merges identical entities and relations from different
segments", and HippoRAG never merges at all -- it keeps every distinct noun
phrase as its own node and expresses near-identity as an extra SYNONYM edge
above a similarity threshold, so that probability can flow between two mentions
without their identities being destroyed.

An earlier version merged on embedding cosine above 0.90, which is neither
paper's rule and turned out to be actively destructive on encyclopaedic text.
Measured on the 2WikiMultiHopQA corpus it collapsed, among 260 cross-name
merges:

    13 may 1840          -> 13 may 1846            (different dates)
    15 january 1892      -> 15 january 1882        (different dates)
    johann wilhelm bach  -> johann christoph bach  (different people)
    john milton glover   -> john montgomery glover (different people)

2Wiki asks when people were born and died and which of two things came first,
so fusing two dates or two brothers removes precisely the distinction the
question tests, and no retrieval step downstream can recover it. Cross-document
alignment of genuinely synonymous names is now the SYNONYM edge's job (see
ingestion.synonyms), which is reversible in a way that a merge is not.
"""
import re
from dataclasses import dataclass, field

_WHITESPACE_RE = re.compile(r"\s+")


def normalise_name(name: str) -> str:
    """The key entities are considered identical on: case and internal spacing
    only. Anything looser risks merging two distinct named things."""
    return _WHITESPACE_RE.sub(" ", name.strip().lower())


@dataclass
class CanonicalEntity:
    node_id: str
    name: str            # canonical (first-seen) surface form
    type: str
    aliases: set[str] = field(default_factory=set)
    chunk_ids: set[str] = field(default_factory=set)
    # chunk_id -> number of mentions in that chunk. Feeds the node-passage
    # count matrix used to score passages from node scores.
    chunk_counts: dict[str, int] = field(default_factory=dict)


def deduplicate_entities(
    raw_entities: list[dict],  # [{"name","type","chunk_id"}]
) -> tuple[list[CanonicalEntity], dict[str, str], list[dict]]:
    """Group raw extractions by normalised name.

    Returns (canonical entities, raw name -> node_id map for relation rewiring,
    merge log). The merge log records only the surface forms that differed
    before normalisation -- with exact matching there is no risky merge to
    audit, but the variants are still worth seeing, since a long tail of
    case-only duplicates says something about extraction consistency.
    """
    if not raw_entities:
        return [], {}, []

    canonicals: list[CanonicalEntity] = []
    by_key: dict[str, CanonicalEntity] = {}
    name_to_node: dict[str, str] = {}
    variants: dict[str, dict] = {}

    for ent in raw_entities:
        key = normalise_name(ent["name"])
        if not key:
            continue
        canon = by_key.get(key)
        if canon is None:
            canon = CanonicalEntity(
                node_id=f"ent-{len(canonicals):05d}",
                name=ent["name"],
                type=ent.get("type", "CONCEPT"),
            )
            by_key[key] = canon
            canonicals.append(canon)
        elif ent["name"] != canon.name:
            row = variants.get(key)
            if row is None:
                variants[key] = {"canonical": canon.name, "variant": ent["name"],
                                 "occurrences": 1}
            else:
                row["occurrences"] += 1

        canon.aliases.add(ent["name"])
        canon.chunk_ids.add(ent["chunk_id"])
        canon.chunk_counts[ent["chunk_id"]] = canon.chunk_counts.get(ent["chunk_id"], 0) + 1
        name_to_node[ent["name"]] = canon.node_id

    merge_log = sorted(variants.values(), key=lambda r: -r["occurrences"])
    return canonicals, name_to_node, merge_log
