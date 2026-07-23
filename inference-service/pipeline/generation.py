"""Constrained generation via Ollama JSON-Schema structured outputs.

The 8B SLM is structurally forced to emit {"decision": ..., "reasoning": ...};
temperature 0 for reproducibility.
"""
import json
from functools import lru_cache

import ollama

import config
from common.schemas import ComplianceDecision
from pipeline.base import RetrievedContext


@lru_cache(maxsize=1)
def _client() -> ollama.Client:
    # Singleton: per-call client construction would leak into generation_ms
    return ollama.Client(host=config.OLLAMA_HOST)

_AUDIT_PROMPT = """You are a GDPR compliance auditor for a federation of universities.
Decide whether the requested data operation is compliant, based STRICTLY on the
legal context provided below. Do not use outside knowledge.

Rules:
- APPROVE only if the context explicitly permits the operation.
- DENY if the context prohibits it or required safeguards are missing.
- UNKNOWN if the provided context is insufficient to decide. Do NOT guess.
- In `reasoning`, cite the clause IDs in square brackets, e.g. [gdpr-art-46-2].

# Legal context
{context}

# Audit request
{query}
"""


def generate_decision(query: str, context: RetrievedContext) -> ComplianceDecision:
    prompt = _AUDIT_PROMPT.format(context=context.to_prompt_block(), query=query)
    response = _client().chat(
        model=config.SLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        format=ComplianceDecision.model_json_schema(),
        options={"temperature": 0},
    )
    raw = response["message"]["content"]
    try:
        return ComplianceDecision.model_validate_json(raw)
    except ValueError:
        # Constrained decoding should prevent this; guard anyway (abstain, don't guess)
        return ComplianceDecision(
            decision="UNKNOWN",
            reasoning=f"Malformed model output: {raw[:200]}",
        )
