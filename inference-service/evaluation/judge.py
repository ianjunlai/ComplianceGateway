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
    # reasoning it is given. Observed truncating on the first real call, and a
    # truncation costs three attempts before the row is abandoned.
    data, _usage = complete_json(
        config.JUDGE_PROVIDER, config.JUDGE_MODEL, prompt,
        max_attempts=max_attempts, max_tokens=4000,
    )
    claims = data.get("claims", [])
    supported = sum(1 for c in claims if c.get("verdict") == "SUPPORTED")
    return {
        "faithfulness": supported / len(claims) if claims else None,
        "claims": claims,
    }


# Faithfulness asks whether the reasoning is grounded in the context. It cannot
# distinguish reasoning that reaches the right answer for the right reason from
# reasoning that reaches it by accident, because both cite the same clauses --
# and that distinction is what an audit trail is for. The rubric below scores
# the reasoning against the scenario the question was built from, which the
# generator recorded as fields rather than prose.
#
# Kept as a separate call, not folded into the faithfulness prompt: the
# faithfulness numbers already reported must stay comparable, and a rubric item
# the SLM never had a chance to address should not drag down a grounding score.
_RUBRIC_PROMPT = """You are auditing a compliance decision for completeness.

# Reference legal context
{context}

# The request
{question}

# What the request actually involves
- purpose: {purpose}
- personal data: {data_category}
- data subject: {data_subject}
- recipient: {recipient}
- crosses a border: {cross_border}
- lawful basis it would rely on: {legal_basis}
- jurisdiction binding the institution: {jurisdiction}

# The auditor's reasoning
{reasoning}

Score each item independently. An item is MET only if the reasoning shows it was
recognised — restating the request is not recognition, and a correct conclusion
reached without addressing the item is not recognition either.

  identifies_data_category  the reasoning is about this kind of personal data
  identifies_legal_basis    it names, or unambiguously describes, the lawful basis
  applies_national_law      it uses the national provision, not the GDPR alone
  addresses_transfer        it addresses the cross-border element, or MET if none
  no_invented_clause        it invents no obligation absent from the context

Return STRICT JSON:
{{"items": {{"identifies_data_category": "MET|NOT_MET", "identifies_legal_basis":
"MET|NOT_MET", "applies_national_law": "MET|NOT_MET", "addresses_transfer":
"MET|NOT_MET", "no_invented_clause": "MET|NOT_MET"}}, "note": "one sentence"}}
"""

_RUBRIC_ITEMS = ("identifies_data_category", "identifies_legal_basis",
                 "applies_national_law", "addresses_transfer", "no_invented_clause")


def judge_decision_rubric(reasoning: str, context_text: str, question: dict,
                          max_attempts: int = 3) -> dict | None:
    """Returns {"rubric_score": float, "items": {...}, "note": str}, or None.

    None when the question carries no `scenario` block -- the single-tier
    dataset predates it, and scoring those against an empty rubric would
    manufacture a difference between datasets rather than between strategies.

    NOTE the rubric treats abstention differently from faithfulness, on purpose.
    Faithfulness returns None for a claim-free abstention and drops it from the
    mean, because an abstention makes no claim that could be unfaithful. The
    rubric scores it 0.2: it answers "was the request analysed", and an
    abstention analysed none of it. The 0.2 rather than 0.0 is
    `no_invented_clause`, which an abstention satisfies trivially. Report the
    two side by side -- a strategy that abstains often will show high
    faithfulness and low rubric, and that pairing is the finding, not a
    contradiction.
    """
    scenario = question.get("scenario")
    if not scenario:
        return None
    prompt = _RUBRIC_PROMPT.format(
        context=context_text, reasoning=reasoning,
        question=question.get("query_text", ""),
        jurisdiction=question.get("jurisdiction", "unspecified"),
        purpose=scenario.get("purpose") or "unspecified",
        data_category=scenario.get("data_category") or "unspecified",
        data_subject=scenario.get("data_subject") or "unspecified",
        recipient=scenario.get("recipient") or "unspecified",
        cross_border="yes" if scenario.get("cross_border") else "no",
        legal_basis=scenario.get("legal_basis") or "unspecified",
    )
    data, _usage = complete_json(
        config.JUDGE_PROVIDER, config.JUDGE_MODEL, prompt,
        max_attempts=max_attempts, max_tokens=4000,
    )
    items = data.get("items", {})
    # Absent items count as NOT_MET rather than being dropped: dividing by the
    # number of items the judge happened to return would score a judge that
    # answered one item out of five as a perfect run.
    met = sum(1 for k in _RUBRIC_ITEMS if items.get(k) == "MET")
    return {
        "rubric_score": met / len(_RUBRIC_ITEMS),
        "items": {k: items.get(k, "NOT_MET") for k in _RUBRIC_ITEMS},
        "note": (data.get("note") or "").strip(),
    }
