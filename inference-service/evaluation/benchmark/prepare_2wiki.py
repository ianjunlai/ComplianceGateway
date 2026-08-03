# -*- coding: utf-8 -*-
"""Convert HippoRAG's released 2WikiMultihopQA subset into this project's schema.

Why this benchmark, and why this copy of it. The GDPR results show no graph
configuration beating dense retrieval, and that claim is only defensible once
the implementations have been shown to reproduce a published advantage
somewhere. 2Wiki is the one benchmark where that advantage is decisive --
HippoRAG reports R@5 89.1 against ColBERTv2's 68.2, where MuSiQue's margin is
2.7 and on HotpotQA HippoRAG loses outright. Taking the authors' own evaluation
subset rather than resampling the source dataset removes any question about how
the sample was drawn.

Source (branch `legacy`, the NeurIPS'24 release the published numbers come from):
    OSU-NLP-Group/HippoRAG  data/2wikimultihopqa_corpus.json   6,119 passages
    OSU-NLP-Group/HippoRAG  data/2wikimultihopqa.json          1,000 questions

Passage titles are unique across the corpus and are what `supporting_facts`
names, so the title is the join key; chunk ids are positional so that no
slugging rule can collapse two distinct titles into one id.

    python -m evaluation.benchmark.prepare_2wiki                 # full: 1000 q, 6119 passages
    python -m evaluation.benchmark.prepare_2wiki --sample 20     # dry run, corpus narrowed to match
"""
import argparse
import json
import random
import urllib.request
from collections import Counter
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
OUT_DIR = _REPO / "dataset" / "benchmark"
RAW_DIR = OUT_DIR / "raw"

_BASE = "https://raw.githubusercontent.com/OSU-NLP-Group/HippoRAG/legacy/data/"
_FILES = {"corpus": "2wikimultihopqa_corpus.json", "qa": "2wikimultihopqa.json"}

SEED = 42


def _fetch(name: str, raw_dir: Path) -> list:
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / _FILES[name]
    if not path.exists():
        url = _BASE + _FILES[name]
        print(f"downloading {url}")
        urllib.request.urlretrieve(url, path)
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sample", type=int, default=0,
                    help="use only N questions, and narrow the corpus to the passages "
                         "those questions offer as candidates. Samples nest, so a small "
                         "rehearsal run is a down payment on a larger one rather than "
                         "throwaway spend.")
    ap.add_argument("--restrict-to-cache", metavar="EXTRACTION_CACHE_JSON",
                    help="keep only passages present in this extraction cache, and drop "
                         "questions whose gold becomes unreachable. For when a run ends "
                         "short of the corpus -- an exhausted quota, say. The alternative "
                         "is a corpus with holes where gold passages should be, which "
                         "depresses every strategy equally and silently.")
    ap.add_argument("--raw-dir", default=str(RAW_DIR))
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args()

    raw_dir, out_dir = Path(args.raw_dir), Path(args.out_dir)
    corpus = _fetch("corpus", raw_dir)
    qa = _fetch("qa", raw_dir)
    print(f"source: {len(corpus)} passages, {len(qa)} questions")

    if args.sample:
        # Shuffle once, then slice — so sample(20) is a prefix of sample(200)
        # and the passages extracted for a rehearsal are still cached when the
        # real run happens. random.sample(k=20) and random.sample(k=200) draw
        # unrelated sets even from the same seed, which would make every
        # rehearsal a write-off.
        qa = list(qa)
        random.Random(SEED).shuffle(qa)
        qa = qa[:args.sample]
        # Keep every passage these questions can see, gold and distractor alike:
        # dropping the distractors would leave a corpus where retrieval is
        # trivial and the comparison meaningless.
        keep = {t for q in qa for t, _ in q["context"]}
        corpus = [c for c in corpus if c["title"] in keep]
        print(f"sampled: {len(corpus)} passages, {len(qa)} questions")

    chunk_id_of = {}
    rows = []
    for i, c in enumerate(corpus):
        cid = f"2wiki-{i:05d}"
        chunk_id_of[c["title"]] = cid
        rows.append({"chunk_id": cid, "source": "2wiki", "title": c["title"], "text": c["text"]})

    if args.restrict_to_cache:
        # Ids are positional over the *unrestricted* corpus, so they are assigned
        # above and filtered here — never renumbered, or they would stop matching
        # the extraction cache they are being restricted to.
        extracted = set(json.loads(
            Path(args.restrict_to_cache).read_text(encoding="utf-8"))["chunks"])
        before = len(rows)
        rows = [r for r in rows if r["chunk_id"] in extracted]
        chunk_id_of = {t: c for t, c in chunk_id_of.items() if c in extracted}
        print(f"restricted to cache: {before} -> {len(rows)} passages "
              f"({before - len(rows)} not extracted)")

    queries, unresolved = [], Counter()
    for q in qa:
        gold_titles = {t for t, _ in q["supporting_facts"]}
        missing = gold_titles - chunk_id_of.keys()
        if missing:
            # Fail loudly: a gold label that resolves to nothing scores zero
            # recall for every strategy at once, which reads as a finding
            # rather than as the mapping bug it is.
            unresolved.update(missing)
            continue
        queries.append({
            "query_id": q["_id"],
            "query_text": q["question"],
            "hop_type": q["type"],          # compositional | comparison | bridge_comparison | inference
            "gold_chunk_ids": sorted(chunk_id_of[t] for t in gold_titles),
            "answer": q["answer"],
        })

    if unresolved and not args.restrict_to_cache:
        raise SystemExit(
            f"{sum(unresolved.values())} gold title(s) across {len(unresolved)} distinct "
            f"names are absent from the corpus, e.g. {list(unresolved)[:5]}. "
            f"Refusing to write a dataset whose ground truth is unreachable.")
    if unresolved:
        # Expected under --restrict-to-cache and reported rather than raised:
        # these questions are dropped, not answered against a corpus missing
        # their evidence. The count belongs in the write-up.
        print(f"dropped {len(qa) - len(queries)} question(s) whose gold was not extracted")

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "2wiki_corpus.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    (out_dir / "2wiki_qa.json").write_text(
        json.dumps(queries, ensure_ascii=False, indent=1), encoding="utf-8")

    by_type = Counter(q["hop_type"] for q in queries)
    gold_sizes = Counter(len(q["gold_chunk_ids"]) for q in queries)
    words = [len(r["text"].split()) for r in rows]
    print(f"\nwrote {out_dir / '2wiki_corpus.json'}  ({len(rows)} passages, "
          f"median {sorted(words)[len(words) // 2]} words)")
    print(f"wrote {out_dir / '2wiki_qa.json'}       ({len(queries)} queries, all gold resolved)")
    print(f"  by hop_type      : {dict(by_type)}")
    print(f"  gold per query   : {dict(sorted(gold_sizes.items()))}")


if __name__ == "__main__":
    main()
