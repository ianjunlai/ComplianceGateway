"""Constrained generation via Ollama JSON-Schema structured outputs: the SLM is
forced to emit {"decision": ..., "reasoning": ...}, at temperature 0."""
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

# Zero-shot needs its own prompt, not the one above with an empty context
# block.
_ZERO_SHOT_PROMPT = """You are a GDPR compliance auditor for a federation of universities.
Decide whether the requested data operation is compliant, using your own
knowledge of the GDPR and of national data protection law. No legal text is
supplied with this request.

Rules:
- APPROVE if the operation is permitted under the law as you understand it.
- DENY if it is prohibited, or if a required safeguard would be missing.
- UNKNOWN only where the answer turns on a detail the legislation itself leaves
  open, for example one set by secondary legislation.
- In `reasoning`, name the provisions you are relying on.

# Audit request
{query}
"""


def generate_decision(query: str, context: RetrievedContext) -> ComplianceDecision:
    # Keyed on the strategy, not on whether the context is empty: retrieving
    # nothing is a retrieval result and is judged against the retrieval prompt.
    if config.ACTIVE_STRATEGY == "zero_shot":
        prompt = _ZERO_SHOT_PROMPT.format(query=query)
    else:
        prompt = _AUDIT_PROMPT.format(context=context.to_prompt_block(), query=query)
    response = _client().chat(
        model=config.SLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        format=ComplianceDecision.model_json_schema(),
        # Must be explicit: Ollama's default window truncates silently from the
        # end, which is exactly where the attached provisions sit.
        options={"temperature": 0, "num_ctx": config.SLM_NUM_CTX},
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
