"""Run one strategy over the question set, one request at a time, recording
per-query outputs and per-stage timings."""
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
    # Kept in step with the factory in pipeline.strategies: a name accepted here
    # must be one build_strategy() knows, and vice versa.
    parser.add_argument("--strategy", required=True,
                        choices=["zero_shot", "vector_rag", "hybrid", "light_rag",
                                 "hippo_rag"])
    parser.add_argument("--dataset", default=str(Path(__file__).resolve().parents[2] / "dataset" / "qa_dataset.json"))
    parser.add_argument("--judge", action="store_true", help="also run the faithfulness judge (costs API calls)")
    parser.add_argument("--warmup", type=int, default=3,
                        help="discarded warm-up runs before measurement (model/index load)")
    parser.add_argument("--run-id", default=datetime.now().strftime("%Y%m%d-%H%M%S"))
    parser.add_argument("--judge-workers", type=int, default=8,
                        help="concurrent judge calls. The judge is network-bound and "
                             "~25x slower than the GPU pipeline it scores, so this sets "
                             "the wall clock of a --judge run. 1 restores the fully "
                             "sequential behaviour earlier results were produced with.")
    parser.add_argument("--limit", type=int, default=0,
                        help="evaluate only the first N queries; for verifying a "
                             "configuration before committing to a full pass")
    parser.add_argument("--resume", action="store_true",
                        help="continue a previous run's .jsonl (pass the same --run-id)")
    args = parser.parse_args()

    config.ACTIVE_STRATEGY = args.strategy  # override before pipeline builds the strategy
    from pipeline.pipeline import run_pipeline

    queries = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    if args.limit:
        queries = queries[:args.limit]
    log.info("Evaluating strategy=%s on %d queries (judge=%s, judge_workers=%d)",
             args.strategy, len(queries), args.judge, args.judge_workers)

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

    # Appended per query: a full pass is hours of inference, so a failure at
    # query 150 must not discard the 149 already paid for.
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    progress = RESULTS_DIR / f"{args.strategy}-{args.run_id}.jsonl"
    rows: list[dict] = []
    if args.resume and progress.exists():
        # A retry appends a second line for the same id, so collapse by id and
        # keep the latest, or both attempts reach the summary.
        latest: dict[str, dict] = {}
        for line in progress.read_text(encoding="utf-8").splitlines():
            if line:
                r = json.loads(line)
                latest[r["query_id"]] = r
        rows = list(latest.values())
        log.info("Resuming: %d queries already attempted in %s", len(rows), progress.name)
    # Only a query with a prediction is finished; failures are re-attempted, or
    # one transient error would make the run impossible to complete.
    done = {r["query_id"] for r in rows if "prediction" in r}
    rows = [r for r in rows if "prediction" in r]
    if args.resume:
        log.info("%d already scored, %d to run", len(done), len(queries) - len(done))

    pending = [q for q in queries if q["query_id"] not in done]

    # Opposite bottlenecks, so the two stages are batched rather than
    # interleaved: the pipeline is GPU-bound and strictly serial at ~1.5 s,
    # while judging is network-bound at ~38 s and leaves the process idle. Run
    # serially the judge would be most of the experiment. The cost is resume
    # granularity, since rows are written per batch, not per query.
    batch_size = max(1, args.judge_workers) if args.judge else 1
    with progress.open("a", encoding="utf-8") as fh:
        for start in range(0, len(pending), batch_size):
            batch = pending[start:start + batch_size]
            staged: list[tuple[dict, dict, object]] = []   # (row, q, result|None)

            for q in batch:
                row = {
                    "query_id": q["query_id"],
                    "hop_type": q["hop_type"],
                    "gold_decision": q["gold_decision"],
                    "gold_chunk_ids": q["gold_chunk_ids"],
                }
                try:
                    result = run_pipeline(AuditRequestEvent(
                        request_id=q["query_id"],
                        source_system=q.get("source_system", "eval"),
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        audit_query=q["query_text"],
                    ))
                    row.update({
                        "prediction": result.decision,
                        "reasoning": result.reasoning,
                        "retrieved_chunk_ids": result.retrieved_chunk_ids,
                        "context_chunk_ids": result.context_chunk_ids,
                        "stage_timings_ms": result.stage_timings_ms,
                    })
                    log.info("%s gold=%s pred=%s",
                             q["query_id"], q["gold_decision"], result.decision)
                except Exception as e:  # noqa: BLE001 — one bad query must not end the run
                    # Recorded, never silently dropped: a query the system could
                    # not answer is a result, but it is not a wrong ANSWER and is
                    # kept out of the accuracy denominator.
                    row["error"] = f"{type(e).__name__}: {e}"
                    log.exception("%s FAILED, continuing", q["query_id"])
                    result = None
                staged.append((row, q, result))

            if args.judge:
                _judge_batch(staged, chunk_texts, args.judge_workers)

            for row, _q, _r in staged:
                rows.append(row)
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()

    summary = _summarize(rows)
    out = RESULTS_DIR / f"{args.strategy}-{args.run_id}.json"
    out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2), encoding="utf-8")
    log.info("Summary: %s", json.dumps(summary, indent=2))
    log.info("Saved -> %s", out)


def _judge_batch(staged: list[tuple[dict, dict, object]], chunk_texts: dict[str, str],
                 workers: int) -> None:
    """Score a batch's rows in parallel, writing the scores back into each
    row."""
    from concurrent.futures import ThreadPoolExecutor

    todo = [(row, q, r) for row, q, r in staged if r is not None]
    if not todo:
        return

    def score(item) -> None:
        row, q, result = item
        try:
            row["faithfulness"] = _judge_row(q, result, chunk_texts)
        except Exception as e:  # noqa: BLE001
            log.warning("%s faithfulness failed: %s", q["query_id"], e)

    if workers <= 1:
        for item in todo:
            score(item)
        return
    with ThreadPoolExecutor(max_workers=min(workers, len(todo))) as pool:
        list(pool.map(score, todo))


def _judge_row(q: dict, result, chunk_texts: dict[str, str]) -> float | None:
    """Faithfulness reference context: zero_shot is judged against GOLD
    chunks."""
    from evaluation.judge import judge_faithfulness

    context = _reference_context(q, result, chunk_texts)
    return judge_faithfulness(result.reasoning, context)["faithfulness"]


def _reference_context(q: dict, result, chunk_texts: dict[str, str]) -> str:
    """Exactly what the model read, and nothing else."""
    if config.ACTIVE_STRATEGY == "zero_shot":
        return "\n\n".join(f"[{cid}]\n{chunk_texts.get(cid, '')}"
                           for cid in q["gold_chunk_ids"])
    ids = (getattr(result, "context_chunk_ids", None)
           or result.retrieved_chunk_ids[: config.GENERATION_CONTEXT_K])
    return "\n\n".join(f"[{cid}]\n{chunk_texts.get(cid, '')}" for cid in ids)


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
    # What the model actually read, attachments included.
    ctx_sizes = [len(r["context_chunk_ids"]) for r in rows if r.get("context_chunk_ids")]
    if ctx_sizes:
        summary["context_size"] = bootstrap_ci([float(n) for n in ctx_sizes])
    # per hop_type breakdown (cross_tier / single / unanswerable)
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
