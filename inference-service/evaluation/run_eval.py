"""Single-strategy evaluation runner (reasoning quality + retrieval efficacy).

Runs the full QA set through the pipeline for ONE strategy (single-user, no
load), records per-query outputs + per-stage timings, then computes metrics.
Repeat with each ACTIVE_STRATEGY value; load testing is driven by JMeter instead.

Usage:
    python -m evaluation.run_eval --strategy hybrid --dataset ../dataset/qa_dataset.json
    python -m evaluation.run_eval --strategy hybrid --judge   # add faithfulness pass
"""
import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import config
from common.schemas import AuditRequestEvent
from evaluation.recall import mean_recall
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

    rows = []
    for q in queries:
        event = AuditRequestEvent(
            request_id=q["query_id"],
            source_system=q.get("source_system", "eval"),
            timestamp=datetime.now(timezone.utc).isoformat(),
            audit_query=q["query_text"],
        )
        result = run_pipeline(event)
        row = {
            "query_id": q["query_id"],
            "hop_type": q["hop_type"],
            "gold_decision": q["gold_decision"],
            "gold_chunk_ids": q["gold_chunk_ids"],
            "prediction": result.decision,
            "reasoning": result.reasoning,
            "retrieved_chunk_ids": result.retrieved_chunk_ids,
            "stage_timings_ms": result.stage_timings_ms,
        }
        if args.judge:
            row["faithfulness"] = _judge_row(q, result, chunk_texts)
        rows.append(row)
        log.info("%s gold=%s pred=%s", q["query_id"], q["gold_decision"], result.decision)

    summary = _summarize(rows)
    out = RESULTS_DIR / f"{args.strategy}-{args.run_id}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
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
    predictions = [r["prediction"] for r in rows]
    golds = [r["gold_decision"] for r in rows]
    retrieval_ms = [r["stage_timings_ms"].get("retrieval_ms", 0) for r in rows]

    summary: dict = {"decision": decision_metrics(predictions, golds)}
    if config.ACTIVE_STRATEGY != "zero_shot":
        summary["recall@5"] = mean_recall(rows, 5)
        summary["recall@10"] = mean_recall(rows, 10)
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
