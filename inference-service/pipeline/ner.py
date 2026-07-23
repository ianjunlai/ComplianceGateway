"""Query entity extraction: zero-shot NER by the local SLM.

Extracted mentions become the seed entities passed to the retrieval strategy.
"""
import json
from functools import lru_cache

import ollama

import config


@lru_cache(maxsize=1)
def _client() -> ollama.Client:
    # Singleton: per-call client construction would leak into ner_ms
    return ollama.Client(host=config.OLLAMA_HOST)

_NER_SCHEMA = {
    "type": "object",
    "properties": {
        "entities": {
            "type": "array",
            "items": {"type": "string"},
        }
    },
    "required": ["entities"],
}

_NER_PROMPT = """You are an entity extractor for legal compliance auditing.
Extract the key legal entities from the audit request below: actors
(e.g. data controller, student, university), data categories (e.g. grades,
transcripts, health data), actions (e.g. transfer, storage, disclosure) and
jurisdictions/destinations (e.g. third country, US university).

Return JSON only.

Audit request:
{query}
"""


def extract_seed_entities(query: str) -> list[str]:
    response = _client().chat(
        model=config.SLM_MODEL,
        messages=[{"role": "user", "content": _NER_PROMPT.format(query=query)}],
        format=_NER_SCHEMA,
        options={"temperature": 0},
    )
    try:
        entities = json.loads(response["message"]["content"])["entities"]
        return [e.strip() for e in entities if e and e.strip()]
    except (KeyError, ValueError, TypeError):
        return []
