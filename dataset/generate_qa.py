"""Synthetic QA dataset generation.

GPT-4o + real corpus chunks -> audit-scenario QA pairs with gold decisions and
gold source chunk IDs (ground truth by construction).

Stratification (n = 150-200):
    single-hop 35% | multi-hop 35% | trap 20% | unanswerable 10%

Quality control: after generation, a 15% stratified sample goes to
human verification — see verification_sample.json output.

Run (needs the QA_GENERATION_PROVIDER's API key in inference-service/.env):
    python generate_qa.py --n 160
"""
import argparse
import json
import random
import re
import sys
from pathlib import Path

# reuse inference-service chunking + config
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "inference-service"))

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / "inference-service" / ".env")

import config  # noqa: E402
from common.llm_clients import complete_json  # noqa: E402
from ingestion.chunking import load_corpus  # noqa: E402

CORPUS_DIR = Path(__file__).parent / "corpus"
OUT_FILE = Path(__file__).parent / "qa_dataset.json"
SAMPLE_FILE = Path(__file__).parent / "verification_sample.json"

STRATA = {"single": 0.35, "multi": 0.35, "trap": 0.20, "unanswerable": 0.10}
SOURCE_SYSTEMS = ["uni_a", "uni_b", "uni_c", "uni_d", "uni_e"]  # simulated federated institutions

_PROMPTS = {
    "single": """Given the legal clause(s) below, write ONE realistic audit request that a
university system would submit (e.g., transferring student grades, sharing transcripts,
processing applications), answerable from THIS clause alone.
Decide the correct outcome (APPROVE or DENY) strictly from the clause.""",
    "multi": """Given the legal clauses below, write ONE realistic audit request whose correct
decision REQUIRES COMBINING the clauses (e.g., a transfer permitted by one clause only
under a safeguard defined in another). Single-clause reasoning must be insufficient.""",
    "trap": """Given the legal clauses below, write ONE audit request that SOUNDS compliant
(mentions consent, agreements, legitimate purposes...) but actually VIOLATES the clauses
below in a subtle way. Correct decision must be DENY.""",
    "unanswerable": """Given the legal clauses below as the ONLY available legal corpus, write ONE
audit request that CANNOT be decided from them (it concerns an obligation or scenario these
clauses do not cover). The correct decision is UNKNOWN. Do not reference the clauses' topics
directly.""",
}

# gold_decision is described rather than shown as "APPROVE|DENY|UNKNOWN":
# generators sometimes echo that pipe-separated form verbatim as the answer.
_FORMAT = """
Return STRICT JSON with exactly these three keys:
  "query_text":    the audit request
  "gold_decision": choose ONE of these three words and nothing else: APPROVE, DENY, UNKNOWN
  "rationale":     one sentence

Legal clauses:
"""


_ART_REF_RE = re.compile(r"Article\s+(\d+[a-z]?)")

# A citation immediately followed by one of these is to a DIFFERENT
# instrument (the EU Treaty, the Charter, another directive/regulation), not
# to GDPR's own numbering -- e.g. Recital 165 cites "Article 17 TFEU", which
# would otherwise false-match GDPR's own Article 17 (an unrelated topic,
# "right to erasure") purely by numeric coincidence.
_OTHER_INSTRUMENT_MARKERS = ("TFEU", "Charter", "Directive", "Regulation (EC)", "Regulation (EU)")


def _reference_pairs(chunks) -> list[tuple]:
    """(chunk, referenced chunk) pairs via explicit 'Article N' cross-references
    to GDPR's own numbering.

    Multi-hop questions must combine clauses that are genuinely connected:
    random pairs would test the combination of unrelated text, which no graph
    traversal could (or should) bridge — biasing the evaluation against GraphRAG in
    exactly the category where it is supposed to help.
    """
    by_art: dict[str, list] = {}
    for c in chunks:
        m = re.match(r"gdpr-art-(\d+[a-z]?)", c.chunk_id)
        if m:
            by_art.setdefault(m.group(1), []).append(c)
    pairs = []
    for c in chunks:
        own = re.match(r"gdpr-art-(\d+[a-z]?)", c.chunk_id)
        refs = set()
        for m in _ART_REF_RE.finditer(c.text):
            tail = c.text[m.end():m.end() + 20].lstrip()
            if any(tail.startswith(marker) for marker in _OTHER_INSTRUMENT_MARKERS):
                continue  # citing a different instrument, not GDPR itself
            refs.add(m.group(1))
        for ref in refs:
            if own and ref == own.group(1):
                continue  # self-reference
            for target in by_art.get(ref, []):
                pairs.append((c, target))
    return pairs


def _build_reference_graph(ref_pairs: list[tuple]) -> dict[str, list]:
    """chunk_id -> genuinely cross-referenced Chunk objects, both directions."""
    graph: dict[str, list] = {}
    for a, b in ref_pairs:
        graph.setdefault(a.chunk_id, []).append(b)
        graph.setdefault(b.chunk_id, []).append(a)
    return graph


def _sample_chunks(chunks, ref_pairs, ref_graph, hop_type: str, k: int):
    """Returns (sampled chunks, whether a genuine cross-reference pair was used).

    For k > 2, a third chunk is added only if it is ITSELF genuinely
    cross-referenced to one of the pair (via ref_graph) — padding with an
    unrelated random chunk would let a retrieval system get penalised on
    Recall@K for correctly NOT retrieving a clause the question never
    actually depends on.
    """
    if hop_type in ("multi", "trap") and ref_pairs:
        base = list(random.choice(ref_pairs))
        if k > 2:
            used = {b.chunk_id for b in base}
            candidates = [
                c for c in ref_graph.get(base[0].chunk_id, []) + ref_graph.get(base[1].chunk_id, [])
                if c.chunk_id not in used
            ]
            if candidates:
                base.append(random.choice(candidates))
            # else: no genuine third connection exists in this corpus -- fall
            # back to the pair alone rather than pad with unrelated noise.
        return base, True
    return random.sample(chunks, k), False


VALID_DECISIONS = {"APPROVE", "DENY", "UNKNOWN"}
# The decision each stratum is defined to produce; anything else means the
# generator ignored the instruction and the item is unusable as ground truth.
EXPECTED_DECISION = {"trap": "DENY", "unanswerable": "UNKNOWN"}


def _validate_item(item: dict, hop_type: str) -> str | None:
    """Returns a rejection reason, or None if the item is usable.

    Generators sometimes echo the format specification instead of choosing
    ("APPROVE|DENY|UNKNOWN"), or pick a decision that contradicts the stratum
    they were asked for. Such an item silently corrupts every metric computed
    from it -- a decision outside the label set can never be matched by any
    system, so it depresses accuracy for all strategies equally and invisibly.
    """
    if not item.get("query_text", "").strip():
        return "empty query_text"
    decision = item.get("gold_decision", "")
    if decision not in VALID_DECISIONS:
        return f"gold_decision {decision!r} is not one of {sorted(VALID_DECISIONS)}"
    expected = EXPECTED_DECISION.get(hop_type)
    if expected and decision != expected:
        return f"{hop_type} item must be gold_decision={expected}, got {decision!r}"
    return None


def generate(n_total: int, seed: int = 42, max_retries_per_item: int = 3) -> list[dict]:
    random.seed(seed)
    chunks = load_corpus(CORPUS_DIR)
    if not chunks:
        raise SystemExit(f"No corpus under {CORPUS_DIR} — see corpus/README.md")
    ref_pairs = _reference_pairs(chunks)
    ref_graph = _build_reference_graph(ref_pairs)
    print(f"{len(ref_pairs)} cross-reference pairs found in corpus")
    dataset = []
    qid = 0
    rejected = 0
    for hop_type, share in STRATA.items():
        for _ in range(round(n_total * share)):
            qid += 1
            item = sampled = used_ref = None
            for attempt in range(1, max_retries_per_item + 1):
                k = {"single": 1, "multi": random.choice([2, 3]), "trap": 2, "unanswerable": 2}[hop_type]
                sampled, used_ref = _sample_chunks(chunks, ref_pairs, ref_graph, hop_type, k)
                clause_block = "\n\n".join(f"[{c.chunk_id}]\n{c.text}" for c in sampled)
                candidate, _usage = complete_json(
                    config.QA_GENERATION_PROVIDER, config.QA_GENERATION_MODEL,
                    _PROMPTS[hop_type] + _FORMAT + clause_block,
                    temperature=0.8,  # diversity across queries
                )
                reason = _validate_item(candidate, hop_type)
                if reason is None:
                    item = candidate
                    break
                rejected += 1
                print(f"  rejected q-{qid:03d} attempt {attempt}/{max_retries_per_item}: {reason}")
            if item is None:
                print(f"  SKIPPED q-{qid:03d} [{hop_type}]: no valid item after "
                      f"{max_retries_per_item} attempts")
                continue
            dataset.append({
                "query_id": f"q-{qid:03d}",
                "query_text": item["query_text"],
                "source_system": random.choice(SOURCE_SYSTEMS),
                "hop_type": hop_type,
                "cross_ref_sampled": used_ref,  # genuine cross-reference pair vs random fallback
                "gold_decision": item["gold_decision"],
                # unanswerable: no gold chunks by definition (excluded from Recall@K)
                "gold_chunk_ids": [] if hop_type == "unanswerable" else [c.chunk_id for c in sampled],
                "generator_rationale": item.get("rationale", ""),
            })
            print(f"generated {dataset[-1]['query_id']} [{hop_type}] {item['gold_decision']}"
                  + (" (cross-ref pair)" if used_ref else ""))
    if rejected:
        print(f"\n{rejected} generation(s) rejected by validation and retried")
    return dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=160)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dataset = generate(args.n, args.seed)
    OUT_FILE.write_text(json.dumps(dataset, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n{len(dataset)} QA pairs -> {OUT_FILE}")

    # Human-verification sample: 15% stratified, PLUS 100% of unanswerable —
    # their gold=UNKNOWN cannot be guaranteed by construction (the generator
    # saw 2 chunks, not the whole corpus)
    random.seed(args.seed + 1)
    sample = []
    for ht in STRATA:
        pool = [d for d in dataset if d["hop_type"] == ht]
        if ht == "unanswerable":
            sample.extend(pool)
        else:
            sample.extend(random.sample(pool, max(1, round(len(pool) * 0.15))))
    sample = [dict(s, human_verdict="") for s in sample]  # CORRECT | INCORRECT | AMBIGUOUS
    SAMPLE_FILE.write_text(json.dumps(sample, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"{len(sample)} items for human verification -> {SAMPLE_FILE}")


if __name__ == "__main__":
    main()
