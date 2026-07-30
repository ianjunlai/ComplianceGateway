"""Graph entity/relation extraction via cloud LLM.

Runs OFFLINE on public legal text only, so a cloud extraction model is
permitted; online audit queries never touch this module.

Every call is metered by cost_tracker (shared indexing cost).
"""
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


def extract_graph_elements(chunk: Chunk, tracker: CostTracker, max_attempts: int = 3) -> dict:
    """Returns {"entities": [...], "relations": [...], "usage": {...}} for one
    chunk, with malformed entries dropped rather than left to crash far
    downstream (in dedup.py, which indexes straight into
    "name"/"source"/"target").

    Token usage is returned alongside the data so a caller that caches this
    result can replay the cost figures too -- indexing cost is a reported
    metric, and it must not disappear just because extraction was cached.
    """
    prompt = _EXTRACTION_PROMPT.replace("{chunk_id}", chunk.chunk_id).replace("{text}", chunk.text)
    with tracker.llm_call("extraction"):
        # Generous relative to a single clause's entity/relation count: an
        # enumerative article (e.g. GDPR Art. 13's list of required
        # disclosures) can produce more output than the 2000-token default,
        # and different providers vary in JSON verbosity for the same content.
        data, usage = complete_json(
            config.EXTRACTION_PROVIDER, config.EXTRACTION_MODEL, prompt,
            max_attempts=max_attempts, max_tokens=4000,
        )
    tracker.add_tokens("extraction", **usage)

    entities = [e for e in data.get("entities", []) if e.get("name")]
    relations = [r for r in data.get("relations", []) if r.get("source") and r.get("target")]
    dropped = (len(data.get("entities", [])) - len(entities)) + (len(data.get("relations", [])) - len(relations))
    if dropped:
        log.warning("%s: dropped %d malformed entity/relation object(s)", chunk.chunk_id, dropped)
    return {"entities": entities, "relations": relations, "usage": usage}
