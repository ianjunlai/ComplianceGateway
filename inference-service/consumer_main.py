"""EDA-mode AI consumer.

Pulls Audit_Request_Events one at a time (concurrency = 1: the poll loop is
strictly serial), runs the GraphRAG pipeline, publishes Audit_Result_Events.

Delivery semantics: at-least-once — offset is committed only AFTER the result
(or DLQ record) is produced. Failed messages go to Audit_DLQ_Topic.
"""
import json
import logging
import re
from datetime import datetime, timezone

from confluent_kafka import Consumer, KafkaError, Producer

import config
from common.schemas import AuditRequestEvent, AuditResultEvent
from pipeline.pipeline import run_pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("consumer")


def _queue_wait_ms(event: AuditRequestEvent) -> int:
    """Gateway ingress -> consumer pickup (queue wait)."""
    try:
        ts = re.sub(r"(\.\d{1,6})\d*", r"\1", event.timestamp.replace("Z", "+00:00"))
        ingress = datetime.fromisoformat(ts)
        return max(0, int((datetime.now(timezone.utc) - ingress).total_seconds() * 1000))
    except ValueError:
        log.warning("Unparseable ingress timestamp: %s", event.timestamp)
        return 0


def _on_kafka_error(err: KafkaError) -> None:
    """Surface librdkafka-level events into the application log.

    Broker and group-coordinator request timeouts are reported only through
    this callback. Without it they are invisible here, yet they are what breaks
    offset commits: OffsetCommit goes to the group coordinator, so a coordinator
    that stops answering both loses the commit and eventually evicts the
    consumer from the group — the exact condition behind a redelivery loop.
    """
    if err.fatal():
        log.error("Kafka client error (fatal): %s", err)
    else:
        log.warning("Kafka client event: %s", err)


def _on_commit(err: KafkaError, partitions: list) -> None:
    """Outcome of an offset commit.

    Commits stay asynchronous, and this callback is what makes them auditable.
    A commit lost to a coordinator timeout otherwise leaves no trace at all: the
    consumer proceeds as if it had committed, the message is redelivered after
    the next rebalance, and it is reprocessed — republishing its result each
    time — for as long as the condition lasts.

    Committing synchronously would catch the failure inline, but confluent-kafka
    exposes no timeout on commit(): with the coordinator unreachable it blocks
    the poll loop indefinitely, which was observed here to stall the consumer
    outright. A logged duplicate is recoverable; a stalled consumer is not.
    """
    if err:
        log.error(
            "Offset commit FAILED (%s) for %s — the affected message will be "
            "redelivered and its result published more than once",
            err, partitions,
        )


def main() -> None:
    consumer = Consumer(
        {
            "bootstrap.servers": config.KAFKA_BOOTSTRAP,
            "group.id": config.CONSUMER_GROUP,
            "enable.auto.commit": False,          # commit after result publication
            "auto.offset.reset": "earliest",
            # generous poll interval: one SLM inference can take minutes under load
            "max.poll.interval.ms": 600_000,
            # Inference runs in THIS process, and saturates the CPU for the whole
            # of it. The client's background I/O threads are starved alongside
            # everything else, so heartbeats can stall for an entire inference:
            # session.timeout.ms must exceed the worst-case inference time, or
            # the consumer is evicted mid-request and its uncommitted message is
            # redelivered and reprocessed, publishing that result twice.
            "session.timeout.ms": 300_000,
            "heartbeat.interval.ms": 10_000,
            # Docker Desktop's port forwarding drops connections left idle
            # through a long inference; keepalives hold them open.
            "socket.keepalive.enable": True,
            "error_cb": _on_kafka_error,
            "on_commit": _on_commit,
        }
    )
    producer = Producer(
        {
            "bootstrap.servers": config.KAFKA_BOOTSTRAP,
            "socket.keepalive.enable": True,
            "error_cb": _on_kafka_error,
        }
    )
    consumer.subscribe([config.REQUEST_TOPIC])
    log.info("Consumer started. strategy=%s topic=%s", config.ACTIVE_STRATEGY, config.REQUEST_TOPIC)

    # Offsets already handled in this process. At-least-once delivery makes a
    # repeat legitimate, but a repeat means the previous commit did not land and
    # the duplicate result inflates the throughput and latency figures — so it
    # must be recorded, not absorbed silently.
    processed: set[tuple[int, int]] = set()

    try:
        while True:
            msg = consumer.poll(timeout=1.0)
            if msg is None:
                continue
            if msg.error():
                if msg.error().code() != KafkaError._PARTITION_EOF:
                    log.error("Kafka error: %s", msg.error())
                continue

            position = (msg.partition(), msg.offset())
            if position in processed:
                log.warning(
                    "REDELIVERY of %s[%d] offset %d — an earlier commit did not land; "
                    "this request's result will be published again",
                    msg.topic(), msg.partition(), msg.offset(),
                )
            processed.add(position)

            raw = msg.value().decode("utf-8")
            try:
                event = AuditRequestEvent.model_validate_json(raw)
                log.info("Processing %s from %s", event.request_id, event.source_system)
                result = run_pipeline(event, queue_wait_ms=_queue_wait_ms(event))
                producer.produce(
                    config.RESULT_TOPIC,
                    key=event.request_id.encode(),
                    value=result.model_dump_json().encode(),
                )
                log.info(
                    "Done %s decision=%s total=%sms",
                    event.request_id,
                    result.decision,
                    result.stage_timings_ms.get("total_ms"),
                )
            except Exception as e:  # noqa: BLE001 — any failure routes to DLQ
                log.exception("Pipeline failure, routing to DLQ")
                _to_dlq(producer, raw, e)
            finally:
                pending = producer.flush(10)
                if pending:
                    # Committing anyway keeps a poison message from blocking the
                    # single-partition queue, but the affected request will stay
                    # PENDING on the gateway — surface it loudly.
                    log.error("%d undelivered result/DLQ message(s) at commit", pending)
                consumer.commit(msg)  # outcome reported by _on_commit
    except KeyboardInterrupt:
        log.info("Shutting down")
    finally:
        consumer.close()


def _to_dlq(producer: Producer, raw: str, error: Exception) -> None:
    dlq_record = {
        "original_message": raw,
        "error": f"{type(error).__name__}: {error}",
        "failed_at": datetime.now(timezone.utc).isoformat(),
    }
    producer.produce(config.DLQ_TOPIC, value=json.dumps(dlq_record).encode())

    # Also emit an ERROR result so the gateway error-rate accounting sees it.
    # The fields are read off the raw JSON rather than the validated model: the
    # common failure is a message that IS valid JSON but fails schema validation
    # (e.g. a null field from a mis-bound gateway payload). Re-validating here
    # would fail identically, emit nothing, and leave the request PENDING forever
    # with the gateway still reporting zero errors.
    try:
        payload = json.loads(raw)
        request_id = payload.get("request_id")
    except (ValueError, AttributeError):
        request_id = None

    if not request_id:
        log.error("Malformed message carries no request_id; the DLQ record is its only trace")
        return

    err_result = AuditResultEvent(
        request_id=request_id,
        source_system=payload.get("source_system") or "unknown",
        decision="ERROR",
        strategy=config.ACTIVE_STRATEGY,
        completed_at=datetime.now(timezone.utc).isoformat(),
        error=f"{type(error).__name__}: {error}",
    )
    producer.produce(
        config.RESULT_TOPIC,
        key=request_id.encode(),
        value=err_result.model_dump_json().encode(),
    )


if __name__ == "__main__":
    main()
