"""Synthetic QA dataset generation.

GPT-4o + real corpus chunks -> audit-scenario QA pairs with gold decisions and
gold source chunk IDs (ground truth by construction).

Stratification (n = 150-200):
    single-hop 35% | multi-hop 35% | trap 20% | unanswerable 10%

Quality control: after generation, a 15% stratified sample goes to
human verification — see verification_sample.json output.

Run (needs OPENAI_API_KEY in inference-service/.env):
    python generate_qa.py --n 160
"""
import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

# reuse inference-service chunking + config
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "inference-service"))

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / "inference-service" / ".env")

from openai import OpenAI  # noqa: E402

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

_FORMAT = """
Return STRICT JSON:
{"query_text": "...", "gold_decision": "APPROVE|DENY|UNKNOWN", "rationale": "one sentence"}

Legal clauses:
"""


_ART_REF_RE = re.compile(r"Article\s+(\d+[a-z]?)")


def _reference_pairs(chunks) -> list[tuple]:
    """(chunk, referenced chunk) pairs via explicit 'Article N' cross-references.

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
        for ref in set(_ART_REF_RE.findall(c.text)):
            if own and ref == own.group(1):
                continue  # self-reference
            for target in by_art.get(ref, []):
                pairs.append((c, target))
    return pairs


def _sample_chunks(chunks, ref_pairs, hop_type: str, k: int):
    """Returns (sampled chunks, whether a genuine cross-reference pair was used)."""
    if hop_type in ("multi", "trap") and ref_pairs:
        base = list(random.choice(ref_pairs))
        if k > 2:
            used = {b.chunk_id for b in base}
            pool = [c for c in chunks if c.chunk_id not in used]
            base += random.sample(pool, k - 2)
        return base, True
    return random.sample(chunks, k), False


def generate(n_total: int, seed: int = 42) -> list[dict]:
    random.seed(seed)
    chunks = load_corpus(CORPUS_DIR)
    if not chunks:
        raise SystemExit(f"No corpus under {CORPUS_DIR} — see corpus/README.md")
    client = OpenAI()
    ref_pairs = _reference_pairs(chunks)
    print(f"{len(ref_pairs)} cross-reference pairs found in corpus")
    dataset = []
    qid = 0
    for hop_type, share in STRATA.items():
        for _ in range(round(n_total * share)):
            qid += 1
            k = {"single": 1, "multi": random.choice([2, 3]), "trap": 2, "unanswerable": 2}[hop_type]
            sampled, used_ref = _sample_chunks(chunks, ref_pairs, hop_type, k)
            clause_block = "\n\n".join(f"[{c.chunk_id}]\n{c.text}" for c in sampled)
            for attempt in range(1, 4):
                try:
                    response = client.chat.completions.create(
                        model="gpt-4o",
                        messages=[{"role": "user", "content": _PROMPTS[hop_type] + _FORMAT + clause_block}],
                        response_format={"type": "json_object"},
                        temperature=0.8,  # diversity across queries
                    )
                    break
                except Exception:
                    if attempt == 3:
                        raise
                    time.sleep(2 ** attempt)
            item = json.loads(response.choices[0].message.content)
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
