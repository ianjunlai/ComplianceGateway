"""Entity and relation extraction via a cloud LLM, offline on public legal text."""
import logging

import config
from common.llm_clients import complete_json
from ingestion.chunking import Chunk
from ingestion.cost_tracker import CostTracker

log = logging.getLogger("extraction")

_EXTRACTION_PROMPT = """You are a legal knowledge-graph extractor.
From the legal clause below, extract:
1. entities: legal concepts, actors, data categories, safeguards, jurisdictions.
2. relations: explicit directed relationships between those entities.

Return STRICT JSON:
{
  "entities": [{"name": "...", "type": "ACTOR|DATA|ACTION|SAFEGUARD|JURISDICTION|CONCEPT"}],
  "relations": [{"source": "...", "target": "...", "type": "...", "description": "one sentence"}]
}

Rules:
- entity names: short canonical noun phrases, singular, lowercase.
- relation type: UPPER_SNAKE_CASE verb phrase (e.g. REQUIRES, PROHIBITS, APPLIES_TO).
- extract only what the text states explicitly; no outside knowledge.

Legal clause [{chunk_id}]:
{text}
"""

# Domain-neutral counterpart, used only by the public-benchmark validity check.
_GENERAL_EXTRACTION_PROMPT = """You are a knowledge-graph extractor.
From the passage below, extract:
1. entities: people, organisations, places, works, events, dates and other
   named things the passage refers to.
2. relations: explicit directed relationships between those entities.

Return STRICT JSON:
{
  "entities": [{"name": "...", "type": "PERSON|ORGANISATION|LOCATION|WORK|EVENT|DATE|CONCEPT"}],
  "relations": [{"source": "...", "target": "...", "type": "...", "description": "one sentence"}]
}

Rules:
- entity names: short canonical noun phrases, singular, lowercase.
- relation type: UPPER_SNAKE_CASE verb phrase (e.g. DIRECTED_BY, BORN_IN, MEMBER_OF).
- extract only what the text states explicitly; no outside knowledge.

Passage [{chunk_id}]:
{text}
"""

_PROMPTS = {"legal": _EXTRACTION_PROMPT, "general": _GENERAL_EXTRACTION_PROMPT}


def _prompt_template() -> str:
    try:
        return _PROMPTS[config.EXTRACTION_PROFILE]
    except KeyError:
        raise SystemExit(
            f"EXTRACTION_PROFILE={config.EXTRACTION_PROFILE!r} is not one of "
            f"{sorted(_PROMPTS)} — a typo here would silently build the graph "
            f"with the wrong extractor"
        )


def extract_graph_elements(chunk: Chunk, tracker: CostTracker, max_attempts: int = 3) -> dict:
    """Returns {"entities": [...], "relations": [...], "usage": {...}} for one
    chunk, with malformed entries dropped rather than left to crash far
    downstream (in dedup.py, which indexes straight into
    "name"/"source"/"target").

    Token usage is returned alongside the data so a caller that caches this
    result can replay the cost figures too -- indexing cost is a reported
    metric, and it must not disappear just because extraction was cached.
    """
    prompt = _prompt_template().replace("{chunk_id}", chunk.chunk_id).replace("{text}", chunk.text)
    # The output budget has to scale with the input: entity and relation counts
    # track passage length, and a flat cap silently becomes a length filter.
    max_output = max(8000, 8 * chunk.approx_tokens)
    with tracker.llm_call("extraction"):
        data, usage = complete_json(
            config.EXTRACTION_PROVIDER, config.EXTRACTION_MODEL, prompt,
            max_attempts=max_attempts, max_tokens=max_output,
        )
    tracker.add_tokens("extraction", **usage)

    entities, relations, dropped = normalise_elements(data)
    if dropped:
        log.warning("%s: dropped %d malformed entity/relation object(s)", chunk.chunk_id, dropped)
    return {"entities": entities, "relations": relations, "usage": usage}


def _text(value) -> str:
    """JSON scalars the schema said would be strings but weren't."""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float)):
        return str(value)
    return ""


def normalise_elements(data: dict) -> tuple[list[dict], list[dict], int]:
    """Coerce one extraction result into the shape the rest of the pipeline
    assumes: string-valued fields, entities with a name, relations with both
    endpoints.

    Applied to cached results as well as fresh ones. Extraction is not re-run
    for a cached chunk, so a fix that only ran on the API path would never
    reach the data that actually gets indexed — and re-extracting to pick it up
    is not an option once a quota is spent.
    """
    entities = []
    for e in data.get("entities", []):
        # A bare scalar where an object was asked for is the model shortening
        # {"name": "x", "type": ...} to "x".
        name = _text(e) if not isinstance(e, dict) else _text(e.get("name"))
        if not name:
            continue
        etype = _text(e.get("type")) if isinstance(e, dict) else ""
        entities.append({"name": name, "type": etype or "CONCEPT"})

    relations = []
    for r in data.get("relations", []):
        if not isinstance(r, dict):
            continue
        source, target = _text(r.get("source")), _text(r.get("target"))
        if not (source and target):
            continue      # no salvage: without both endpoints there is no edge
        relations.append({
            "source": source, "target": target,
            "type": _text(r.get("type")) or "RELATES",
            "description": _text(r.get("description")),
        })

    dropped = ((len(data.get("entities", [])) - len(entities))
               + (len(data.get("relations", [])) - len(relations)))
    return entities, relations, dropped
