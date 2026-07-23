"""Graph entity/relation extraction via cloud LLM.

Runs OFFLINE on public legal text only, so a cloud extraction model is
permitted; online audit queries never touch this module.

Every call is metered by cost_tracker (shared indexing cost).
"""
import json
import time

from openai import OpenAI

import config
from ingestion.chunking import Chunk
from ingestion.cost_tracker import CostTracker

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
    """Returns {"entities": [...], "relations": [...]} for one chunk.

    Retries with exponential backoff: a transient API failure (rate limit,
    network) must not abort a multi-minute corpus build.
    """
    client = OpenAI()
    prompt = _EXTRACTION_PROMPT.replace("{chunk_id}", chunk.chunk_id).replace("{text}", chunk.text)
    for attempt in range(1, max_attempts + 1):
        try:
            with tracker.llm_call("extraction"):
                response = client.chat.completions.create(
                    model=config.EXTRACTION_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                    temperature=0,
                )
            break
        except Exception:
            if attempt == max_attempts:
                raise
            time.sleep(2 ** attempt)
    tracker.add_tokens(
        "extraction",
        prompt_tokens=response.usage.prompt_tokens,
        completion_tokens=response.usage.completion_tokens,
    )
    data = json.loads(response.choices[0].message.content)
    data.setdefault("entities", [])
    data.setdefault("relations", [])
    return data
