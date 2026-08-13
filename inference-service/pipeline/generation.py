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

# The zero-shot condition needs its own prompt, not the one above with an empty
# context block.
#
# The retrieval prompt says "based STRICTLY on the legal context provided" and
# "UNKNOWN if the provided context is insufficient". Rendered with no context
# those two lines make UNKNOWN the only correct answer to every question, and a
# model that follows instructions will abstain on all of them -- measured at
# 93% with a 14B model, against 13% with an 8B one that ignored the
# instruction. That measures instruction-following, not what the baseline is
# for.
#
# Zero-shot is meant to answer the question the retrieval conditions answer,
# using what the model already knows instead of retrieved text. UNKNOWN stays
# available, because the unanswerable stratum needs it, but nothing here
# invites it.
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
    # Keyed on the configured strategy rather than on whether the context came
    # back empty: a retrieval strategy that found nothing must still be judged
    # against the retrieval prompt, because retrieving nothing is its result.
    if config.ACTIVE_STRATEGY == "zero_shot":
        prompt = _ZERO_SHOT_PROMPT.format(query=query)
    else:
        prompt = _AUDIT_PROMPT.format(context=context.to_prompt_block(), query=query)
    response = _client().chat(
        model=config.SLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        format=ComplianceDecision.model_json_schema(),
        # num_ctx must be set explicitly. Ollama defaults to a small window and
        # silently truncates anything beyond it -- from the end, which is
        # exactly where the attached provisions sit. Ten chunks of this corpus
        # run about 2.5k tokens at the median and 18k at the tail, so the
        # default would drop the cited provision on most requests and the
        # attachment mechanism would appear to do nothing.
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
