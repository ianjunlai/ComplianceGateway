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

# Domain-neutral counterpart for the public-benchmark validity check. Seeds are
# where every graph strategy enters the graph, so a legal-primed extractor
# reading "Who is the mother of the director of Polish-Russian War?" returns
# seeds that link to nothing and the traversal never starts. Kept beside the
# legal prompt rather than templated from it: the two ask for different things,
# and merging them would blunt both.
_GENERAL_NER_PROMPT = """You are an entity extractor.
Extract the named entities the question is about: people, organisations,
places, works (films, books, albums), and events.

Return JSON only.

Question:
{query}
"""

_PROMPTS = {"legal": _NER_PROMPT, "general": _GENERAL_NER_PROMPT}


def _prompt_template() -> str:
    try:
        return _PROMPTS[config.EXTRACTION_PROFILE]
    except KeyError:
        raise SystemExit(
            f"EXTRACTION_PROFILE={config.EXTRACTION_PROFILE!r} is not one of "
            f"{sorted(_PROMPTS)}"
        )


def extract_seed_entities(query: str) -> list[str]:
    response = _client().chat(
        model=config.SLM_MODEL,
        messages=[{"role": "user", "content": _prompt_template().format(query=query)}],
        format=_NER_SCHEMA,
        options={"temperature": 0},
    )
    try:
        entities = json.loads(response["message"]["content"])["entities"]
        return [e.strip() for e in entities if e and e.strip()]
    except (KeyError, ValueError, TypeError):
        return []
