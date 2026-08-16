"""Entity deduplication: merge only names identical after normalising case and whitespace. Near-identity is a SYNONYM edge instead."""
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
    chunk_counts: dict[str, int] = field(default_factory=dict)


def deduplicate_entities(
    raw_entities: list[dict],  # [{"name","type","chunk_id"}]
) -> tuple[list[CanonicalEntity], dict[str, str], list[dict]]:
    """Group raw extractions by normalised name."""
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
