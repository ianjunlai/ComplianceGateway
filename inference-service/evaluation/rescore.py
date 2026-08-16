# -*- coding: utf-8 -*-
"""Re-score a completed run from the stored rows, without re-running inference."""
import argparse
import json
import sys
from pathlib import Path

_SERVICE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_SERVICE))

from evaluation.stats import bootstrap_ci, mcnemar_test   # noqa: E402

_REPO = _SERVICE.parent
RESULTS = _REPO / "results"
DATASET = _REPO / "dataset" / "qa_dataset.json"
ORDER = ["zero_shot", "vector_rag", "hybrid", "light_rag", "hippo_rag"]


def load(run_id: str, dataset: Path):
    qa = {q["query_id"]: q for q in json.loads(dataset.read_text(encoding="utf-8"))}
    preds = {}
    for f in sorted(RESULTS.glob(f"*-{run_id}.jsonl")):
        name = f.stem[: -(len(run_id) + 1)]
        rows = [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]
        preds[name] = {r["query_id"]: r.get("prediction") for r in rows if "prediction" in r}
    if not preds:
        raise SystemExit(f"no *-{run_id}.jsonl under {RESULTS}")
    return qa, preds


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-id", default="server1")
    ap.add_argument("--dataset", default=str(DATASET))
    args = ap.parse_args()

    qa, preds = load(args.run_id, Path(args.dataset))
    names = [n for n in ORDER if n in preds] + [n for n in preds if n not in ORDER]

    answerable = [q for q, v in qa.items() if v["gold_decision"] in ("APPROVE", "DENY")]
    deny = [q for q, v in qa.items() if v["gold_decision"] == "DENY"]
    unk_sound = [q for q, v in qa.items()
                 if v["gold_decision"] == "UNKNOWN" and not v["gold_chunk_ids"]]
    unk_excluded = [q for q, v in qa.items()
                    if v["gold_decision"] == "UNKNOWN" and v["gold_chunk_ids"]]

    print(f"run-id {args.run_id}   {len(qa)} queries, {len(names)} strategies\n")
    print("subsets")
    print(f"  answerable (gold APPROVE/DENY)        n={len(answerable)}")
    print(f"  of which DENY, used for FPR           n={len(deny)}")
    print(f"  unanswerable, no gold clause          n={len(unk_sound)}")
    print(f"  EXCLUDED: UNKNOWN with a gold clause  n={len(unk_excluded)}"
          f"   (label defined against one sampled clause, not the corpus)")

    def correct(name, ids):
        p = preds[name]
        return [p.get(q) == qa[q]["gold_decision"] for q in ids]

    print("\n" + "=" * 76)
    print("1  ANSWERABLE ONLY — the questions with a real answer")
    print("=" * 76)
    head = f"{'strategy':<12}{'accuracy':>10}{'95% CI':>18}{'abstained':>11}{'wrong call':>12}"
    print(head); print("-" * len(head))
    for n in names:
        c = correct(n, answerable)
        ci = bootstrap_ci([1.0 if x else 0.0 for x in c])
        ab = sum(1 for q in answerable if preds[n].get(q) == "UNKNOWN") / len(answerable)
        acc = sum(c) / len(c)
        interval = "[%.3f, %.3f]" % (ci["ci_low"], ci["ci_high"])
        print(f"{n:<12}{acc:>10.3f}{interval:>18}{ab:>11.3f}{1 - acc - ab:>12.3f}")

    print("\n" + "=" * 76)
    print("2  FALSE APPROVAL RATE by stratum — the dangerous failure")
    print("   Computed only on gold=DENY, so the excluded labels cannot affect it.")
    print("=" * 76)
    strata = sorted({qa[q]["hop_type"] for q in deny})
    head = f"{'strategy':<12}" + "".join(f"{s:>14}" for s in strata) + f"{'overall':>10}"
    print(head); print("-" * len(head))
    for n in names:
        line = f"{n:<12}"
        for s in strata:
            ids = [q for q in deny if qa[q]["hop_type"] == s]
            if ids:
                fa = sum(1 for q in ids if preds[n].get(q) == "APPROVE")
                line += f"{fa / len(ids):>14.3f}"
            else:
                line += f"{'-':>14}"
        fa_all = sum(1 for q in deny if preds[n].get(q) == "APPROVE")
        line += f"{fa_all / len(deny):>10.3f}"
        print(line)

    print("\n  paired difference in FPR against zero_shot (95% bootstrap CI)")
    base = "zero_shot" if "zero_shot" in preds else names[0]
    b_fa = [1.0 if preds[base].get(q) == "APPROVE" else 0.0 for q in deny]
    for n in names:
        if n == base:
            continue
        n_fa = [1.0 if preds[n].get(q) == "APPROVE" else 0.0 for q in deny]
        ci = bootstrap_ci([a - b for a, b in zip(n_fa, b_fa)])
        sig = "significant" if (ci["ci_low"] > 0 or ci["ci_high"] < 0) else "not significant"
        print(f"    {n:<12}{ci['mean']:+8.3f}   [{ci['ci_low']:+.3f}, {ci['ci_high']:+.3f}]   {sig}")

    print("\n" + "=" * 76)
    print(f"3  CORRECT ABSTENTION on the {len(unk_sound)} sound unanswerable items")
    print("   Small n: report the interval, not the point estimate.")
    print("=" * 76)
    head = f"{'strategy':<12}{'abstained':>11}{'95% CI':>18}"
    print(head); print("-" * len(head))
    for n in names:
        vals = [1.0 if preds[n].get(q) == "UNKNOWN" else 0.0 for q in unk_sound]
        ci = bootstrap_ci(vals)
        interval = "[%.3f, %.3f]" % (ci["ci_low"], ci["ci_high"])
        print(f"{n:<12}{ci['mean']:>11.3f}{interval:>18}")

    print("\n" + "=" * 76)
    print("4  McNEMAR, paired, on the answerable subset")
    print("=" * 76)
    for base in [b for b in ("zero_shot", "vector_rag") if b in preds]:
        print(f"\n  against {base}")
        print(f"    {'strategy':<12}{'wins':>7}{'losses':>8}{'p':>10}")
        cb = correct(base, answerable)
        for n in names:
            if n == base:
                continue
            r = mcnemar_test(correct(n, answerable), cb)
            print(f"    {n:<12}{r['b']:>7}{r['c']:>8}{r['p_value']:>10.3f}")
    print("\n  'wins' = the strategy is right where the baseline is wrong; p is exact binomial.")


if __name__ == "__main__":
    main()
