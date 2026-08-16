"""LLM-as-judge for faithfulness: split the reasoning into claims and mark each as supported by the context or not."""
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
    """Returns {"faithfulness": float | None, "claims": [...]}."""
    prompt = _JUDGE_PROMPT.format(context=context_text, reasoning=reasoning)
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
