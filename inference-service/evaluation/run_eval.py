"""Single-strategy evaluation runner (reasoning quality + retrieval efficacy).

Runs the full QA set through the pipeline for ONE strategy (single-user, no
load), records per-query outputs + per-stage timings, then computes metrics.
Repeat with each ACTIVE_STRATEGY value; load testing is driven by JMeter instead.

Progress is appended to results/<strategy>-<run_id>.jsonl as each query
completes, and a failed query is recorded and skipped rather than ending the
run. Both matter at this scale: one strategy is a few hours of local inference.

Usage:
    python -m evaluation.run_eval --strategy hybrid --dataset ../dataset/qa_dataset.json
    python -m evaluation.run_eval --strategy hybrid --judge   # add faithfulness pass
    python -m evaluation.run_eval --strategy hybrid --run-id 20260730-1200 --resume
"""
import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import config
from common.schemas import AuditRequestEvent
from evaluation.recall import recall_values
from evaluation.stats import bootstrap_ci, decision_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("run_eval")

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", required=True,
                        choices=["zero_shot", "vector_rag", "hybrid", "light_rag", "hippo_rag"])
    parser.add_argument("--dataset", default=str(Path(__file__).resolve().parents[2] / "dataset" / "qa_dataset.json"))
    parser.add_argument("--judge", action="store_true", help="also run the faithfulness judge (costs API calls)")
    parser.add_argument("--warmup", type=int, default=3,
                        help="discarded warm-up runs before measurement (model/index load)")
    parser.add_argument("--run-id", default=datetime.now().strftime("%Y%m%d-%H%M%S"))
    parser.add_argument("--resume", action="store_true",
                        help="continue a previous run's .jsonl (pass the same --run-id)")
    args = parser.parse_args()

    config.ACTIVE_STRATEGY = args.strategy  # override before pipeline builds the strategy
    from pipeline.pipeline import run_pipeline

    queries = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    log.info("Evaluating strategy=%s on %d queries", args.strategy, len(queries))

    chunk_texts: dict[str, str] = {}
    if args.judge:
        chunk_texts = json.loads(
            (Path(config.ARTIFACTS_DIR) / "chunk_texts.json").read_text(encoding="utf-8"))

    for i in range(args.warmup):
        run_pipeline(AuditRequestEvent(
            request_id=f"warmup-{i}",
            source_system="warmup",
            timestamp=datetime.now(timezone.utc).isoformat(),
            audit_query=queries[0]["query_text"],
        ))
        log.info("Warm-up %d/%d done (discarded)", i + 1, args.warmup)

    # Rows are appended to a JSONL as they complete. A full pass is hours of
    # local inference per strategy, so a failure at query 150 must not discard
    # the 149 already paid for; --resume picks the same file back up.
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    progress = RESULTS_DIR / f"{args.strategy}-{args.run_id}.jsonl"
    rows: list[dict] = []
    if args.resume and progress.exists():
        # A retried query appends a second line for the same id, so the file is
        # collapsed by id keeping the latest attempt — otherwise the retry and
        # the failure it replaced would both reach the summary.
        latest: dict[str, dict] = {}
        for line in progress.read_text(encoding="utf-8").splitlines():
            if line:
                r = json.loads(line)
                latest[r["query_id"]] = r
        rows = list(latest.values())
        log.info("Resuming: %d queries already attempted in %s", len(rows), progress.name)
    # Only a query that actually produced a prediction is finished. Previously
    # failed queries are re-attempted: resuming past them would make a run with
    # any transient failure impossible to complete.
    done = {r["query_id"] for r in rows if "prediction" in r}
    rows = [r for r in rows if "prediction" in r]
    if args.resume:
        log.info("%d already scored, %d to run", len(done), len(queries) - len(done))

    with progress.open("a", encoding="utf-8") as fh:
        for q in queries:
            if q["query_id"] in done:
                continue
            event = AuditRequestEvent(
                request_id=q["query_id"],
                source_system=q.get("source_system", "eval"),
                timestamp=datetime.now(timezone.utc).isoformat(),
                audit_query=q["query_text"],
            )
            row = {
                "query_id": q["query_id"],
                "hop_type": q["hop_type"],
                "gold_decision": q["gold_decision"],
                "gold_chunk_ids": q["gold_chunk_ids"],
            }
            try:
                result = run_pipeline(event)
                row.update({
                    "prediction": result.decision,
                    "reasoning": result.reasoning,
                    "retrieved_chunk_ids": result.retrieved_chunk_ids,
                    "stage_timings_ms": result.stage_timings_ms,
                })
                if args.judge:
                    row["faithfulness"] = _judge_row(q, result, chunk_texts)
                log.info("%s gold=%s pred=%s", q["query_id"], q["gold_decision"], result.decision)
            except Exception as e:  # noqa: BLE001 — one bad query must not end the run
                # Recorded, never silently dropped: a query the system could not
                # answer is a result, but it is not a wrong ANSWER and is kept
                # out of the accuracy denominator.
                row["error"] = f"{type(e).__name__}: {e}"
                log.exception("%s FAILED, continuing", q["query_id"])
            rows.append(row)
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()

    summary = _summarize(rows)
    out = RESULTS_DIR / f"{args.strategy}-{args.run_id}.json"
    out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2), encoding="utf-8")
    log.info("Summary: %s", json.dumps(summary, indent=2))
    log.info("Saved -> %s", out)


def _judge_row(q: dict, result, chunk_texts: dict[str, str]) -> float | None:
    """Faithfulness reference context: zero_shot is judged against GOLD chunks.

    RAG strategies are judged against the EXACT context the SLM generated from
    (top GENERATION_CONTEXT_K of the ranked list) — judging against the full
    RETRIEVAL_K list would credit hallucinations that happen to coincide with
    chunks the model never saw.
    """
    from evaluation.judge import judge_faithfulness

    if config.ACTIVE_STRATEGY == "zero_shot":
        ids = q["gold_chunk_ids"]
    else:
        ids = result.retrieved_chunk_ids[: config.GENERATION_CONTEXT_K]
    context = "\n\n".join(f"[{cid}]\n{chunk_texts.get(cid, '')}" for cid in ids)
    return judge_faithfulness(result.reasoning, context)["faithfulness"]


def _summarize(rows: list[dict]) -> dict:
    # Failed queries are reported as a count, not folded into the metrics: a
    # missing prediction can never equal the gold label, so counting it would
    # depress accuracy identically for every strategy and hide the failure.
    failed = [r for r in rows if "prediction" not in r]
    rows = [r for r in rows if "prediction" in r]
    if not rows:
        return {"n_scored": 0, "n_failed": len(failed)}

    predictions = [r["prediction"] for r in rows]
    golds = [r["gold_decision"] for r in rows]
    retrieval_ms = [r["stage_timings_ms"].get("retrieval_ms", 0) for r in rows]

    summary: dict = {
        "n_scored": len(rows),
        "n_failed": len(failed),
        "failed_query_ids": [r["query_id"] for r in failed],
        "decision": decision_metrics(predictions, golds),
    }
    if config.ACTIVE_STRATEGY != "zero_shot":
        # Recall carries a CI like every other aggregate: RQ2 compares recall
        # ACROSS paradigms, and a bare mean cannot support that comparison.
        summary["recall@5"] = bootstrap_ci(recall_values(rows, 5))
        summary["recall@10"] = bootstrap_ci(recall_values(rows, 10))
        summary["recall_n_scored"] = len(recall_values(rows, 5))  # unanswerable excluded
        summary["retrieval_latency_ms"] = bootstrap_ci([float(v) for v in retrieval_ms])
    faith_vals = [r["faithfulness"] for r in rows if r.get("faithfulness") is not None]
    if faith_vals:
        summary["faithfulness"] = bootstrap_ci(faith_vals)
        summary["faithfulness_n_scored"] = len(faith_vals)  # claims-free abstentions excluded
    # per hop_type breakdown (single/multi/trap/unanswerable)
    summary["by_hop_type"] = {
        ht: decision_metrics(
            [r["prediction"] for r in rows if r["hop_type"] == ht],
            [r["gold_decision"] for r in rows if r["hop_type"] == ht],
        )
        for ht in sorted({r["hop_type"] for r in rows})
    }
    return summary


if __name__ == "__main__":
    main()
