# -*- coding: utf-8 -*-
"""Fill in judge scores on rows that already have a prediction.

A judge outage does not invalidate a run: the SLM's decision and the retrieved
chunk ids are recorded per row as they complete, and scoring them is a pure
function of `reasoning` + the retrieved context. Only the API calls were lost.

`run_eval --resume` cannot recover this. It treats a row as finished when it
has a `prediction`, which is the right rule for resuming an interrupted pass --
re-running the pipeline would spend GPU hours re-deriving answers that are
already on disk -- but it means a row that was predicted and then failed to be
scored is skipped forever. This runs the judge, and only the judge, over
exactly those rows.

    # what is missing, without calling anything
    python -m evaluation.rejudge --run-id full0810 --dataset ../dataset/crosstier_qa_full.json --check

    # fill it in
    python -m evaluation.rejudge --run-id full0810 --dataset ../dataset/crosstier_qa_full.json
"""
import argparse
import json
import logging
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

_SERVICE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_SERVICE))

import config                                        # noqa: E402
from evaluation.run_eval import (                    # noqa: E402
    RESULTS_DIR, _reference_context, _summarize,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("rejudge")


class _StoredResult:
    """The parts of a pipeline result the judge needs, rebuilt from a saved row.

    `_reference_context` and the judge take a result object, not a dict, so the
    row is wrapped rather than the call sites being changed -- keeping one code
    path for how the reference context is assembled. If that path ever diverges
    between run_eval and here, the scores stop being comparable.
    """

    def __init__(self, row: dict) -> None:
        self.reasoning = row.get("reasoning", "")
        self.retrieved_chunk_ids = row.get("retrieved_chunk_ids", [])


def _needs(row: dict, scenario_available: bool) -> bool:
    if "prediction" not in row:
        return False            # never answered; rejudging cannot invent one
    if row.get("faithfulness", "missing") == "missing":
        return True
    # faithfulness is legitimately None for a claim-free abstention, so its
    # presence -- not its truthiness -- is what marks the row as scored.
    return scenario_available and "rubric_score" not in row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--dataset", required=True,
                    help="the question set; supplies the scenario block the rubric needs")
    ap.add_argument("--strategies", default="",
                    help="comma-separated; default is every result file for this run-id")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--check", action="store_true",
                    help="report what is missing and exit without calling the API")
    args = ap.parse_args()

    questions = {q["query_id"]: q
                 for q in json.loads(Path(args.dataset).read_text(encoding="utf-8"))}
    scenario_available = any(q.get("scenario") for q in questions.values())
    if not scenario_available:
        log.warning("no scenario block in %s — only faithfulness will be filled in",
                    args.dataset)

    chunk_texts = json.loads(
        (Path(config.ARTIFACTS_DIR) / "chunk_texts.json").read_text(encoding="utf-8"))

    if args.strategies:
        names = [s.strip() for s in args.strategies.split(",") if s.strip()]
    else:
        names = sorted(p.name[:-len(f"-{args.run_id}.json")]
                       for p in RESULTS_DIR.glob(f"*-{args.run_id}.json"))
    if not names:
        raise SystemExit(f"no result files matching *-{args.run_id}.json in {RESULTS_DIR}")

    total_missing = 0
    for name in names:
        out = RESULTS_DIR / f"{name}-{args.run_id}.json"
        rows = json.loads(out.read_text(encoding="utf-8"))["rows"]
        missing = [r for r in rows if _needs(r, scenario_available)]
        total_missing += len(missing)
        print(f"  {name:<16}{len(missing):>4} of {len(rows)} rows unscored")
        if args.check or not missing:
            continue

        # The strategy decides which context the judge is shown: zero_shot is
        # judged against gold, everything else against what it retrieved. Set it
        # per file, exactly as run_eval would have.
        config.ACTIVE_STRATEGY = name

        def score(row: dict) -> None:
            from evaluation.judge import judge_faithfulness, judge_decision_rubric
            q = questions.get(row["query_id"])
            if q is None:
                log.warning("%s not in the dataset, skipping", row["query_id"])
                return
            result = _StoredResult(row)
            context = _reference_context(q, result, chunk_texts)
            try:
                row["faithfulness"] = judge_faithfulness(result.reasoning, context)["faithfulness"]
            except Exception as e:                                   # noqa: BLE001
                log.warning("%s faithfulness failed: %s", row["query_id"], str(e)[:160])
            try:
                rubric = judge_decision_rubric(result.reasoning, context, q)
                if rubric is not None:
                    row["rubric_score"] = rubric["rubric_score"]
                    row["rubric_items"] = rubric["items"]
            except Exception as e:                                   # noqa: BLE001
                log.warning("%s rubric failed: %s", row["query_id"], str(e)[:160])

        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            list(pool.map(score, missing))

        still = [r for r in rows if _needs(r, scenario_available)]
        summary = _summarize(rows)
        out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2),
                       encoding="utf-8")
        # The .jsonl is the resume ledger; leaving it stale would make a later
        # --resume rebuild the .json from unscored rows and silently undo this.
        jsonl = RESULTS_DIR / f"{name}-{args.run_id}.jsonl"
        jsonl.write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
            encoding="utf-8")
        log.info("%s: %d scored, %d still missing -> %s",
                 name, len(missing) - len(still), len(still), out.name)

    if args.check:
        print(f"\n{total_missing} rows would be judged "
              f"(2 API calls each, ~{total_missing * 2} calls)")


if __name__ == "__main__":
    main()
