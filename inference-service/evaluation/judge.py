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
1. Decompose the reasoning into atomic claims ABOUT THE LAW OR THE FACTS OF THE CASE.
2. For each claim, decide whether it is SUPPORTED by the reference context or NOT_SUPPORTED
   (uses information absent from, or contradicting, the context).
3. Judge strictly: a claim citing a clause that does not appear in the context is NOT_SUPPORTED.

Do NOT treat statements about the auditor's own certainty or process as claims. Sentences
such as "there is insufficient information to decide", "the context does not address this"
or "further review is required" describe the reasoning, not the law, and there is nothing
in the context that could support or contradict them. Omit them. If the reasoning consists
only of such statements, return an empty list — an honest abstention has no claims to be
faithful about, and scoring it as fully faithful would reward abstaining.

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
    # The default 2000-token cap truncates a decomposition of any length: the
    # judge restates each claim before scoring it, so output grows with the
    # reasoning it is given. 8000 because a reasoning judge's thinking length
    # is not stable enough to size the budget to the typical case -- measured
    # completions sit around 300 tokens, but one call in three overran 4000,
    # and a truncation costs three attempts before the score is lost.
    data, _usage = complete_json(
        config.JUDGE_PROVIDER, config.JUDGE_MODEL, prompt,
        max_attempts=max_attempts, max_tokens=8000,
    )
    claims = data.get("claims", [])
    supported = sum(1 for c in claims if c.get("verdict") == "SUPPORTED")
    return {
        "faithfulness": supported / len(claims) if claims else None,
        "claims": claims,
    }
