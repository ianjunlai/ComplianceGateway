"""LLM-as-judge for Faithfulness.

Judge and generator should be different vendor families to mitigate
same-source bias (see config.JUDGE_PROVIDER / EXTRACTION_PROVIDER). Receives
ONLY synthetic queries and system outputs — never real student data.

Faithfulness = supported claims / total claims, judged against:
  - the RETRIEVED context for RAG strategies;
  - the GOLD chunks for zero_shot.
"""
import config
from common.llm_clients import complete_json

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
    prompt = _JUDGE_PROMPT.format(context=context_text, reasoning=reasoning)
    data, _usage = complete_json(
        config.JUDGE_PROVIDER, config.JUDGE_MODEL, prompt, max_attempts=max_attempts,
    )
    claims = data.get("claims", [])
    supported = sum(1 for c in claims if c.get("verdict") == "SUPPORTED")
    return {
        "faithfulness": supported / len(claims) if claims else None,
        "claims": claims,
    }
