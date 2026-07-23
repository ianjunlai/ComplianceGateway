"""LLM-as-judge for Faithfulness.

Judge = Claude (different vendor family from the GPT-4o generator — mitigates
generator/judge same-source bias). Receives ONLY synthetic queries and system
outputs — never real student data.

Faithfulness = supported claims / total claims, judged against:
  - the RETRIEVED context for RAG strategies;
  - the GOLD chunks for zero_shot.
"""
import json
import time

import anthropic

import config

_JUDGE_PROMPT = """You are evaluating the faithfulness of a compliance auditor's reasoning.

# Reference legal context
{context}

# Auditor's reasoning
{reasoning}

Task:
1. Decompose the reasoning into atomic factual/legal claims.
2. For each claim, decide whether it is SUPPORTED by the reference context or NOT_SUPPORTED
   (uses information absent from, or contradicting, the context).
3. Judge strictly: a claim citing a clause that does not appear in the context is NOT_SUPPORTED.

Return STRICT JSON:
{{"claims": [{{"claim": "...", "verdict": "SUPPORTED|NOT_SUPPORTED"}}]}}
"""


def judge_faithfulness(reasoning: str, context_text: str, max_attempts: int = 3) -> dict:
    """Returns {"faithfulness": float | None, "claims": [...]}.

    faithfulness is None when the reasoning contains no checkable claims
    (e.g. a bare abstention): excluded from aggregation, NOT scored 0 —
    punishing an honest abstention as maximally unfaithful would invert
    the metric's meaning.
    """
    client = anthropic.Anthropic()
    for attempt in range(1, max_attempts + 1):
        try:
            response = client.messages.create(
                model=config.JUDGE_MODEL,
                max_tokens=2000,
                messages=[{
                    "role": "user",
                    "content": _JUDGE_PROMPT.format(context=context_text, reasoning=reasoning),
                }],
            )
            break
        except Exception:
            if attempt == max_attempts:
                raise
            time.sleep(2 ** attempt)
    text = response.content[0].text
    # tolerate judge wrapping JSON in prose/code fences
    start, end = text.find("{"), text.rfind("}")
    data = json.loads(text[start:end + 1])
    claims = data.get("claims", [])
    supported = sum(1 for c in claims if c.get("verdict") == "SUPPORTED")
    return {
        "faithfulness": supported / len(claims) if claims else None,
        "claims": claims,
    }
