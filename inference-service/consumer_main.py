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


def main() -> None:
    consumer = Consumer(
        {
            "bootstrap.servers": config.KAFKA_BOOTSTRAP,
            "group.id": config.CONSUMER_GROUP,
            "enable.auto.commit": False,          # commit after result publication
            "auto.offset.reset": "earliest",
            # generous poll interval: one SLM inference can take minutes under load
            "max.poll.interval.ms": 600_000,
        }
    )
    producer = Producer({"bootstrap.servers": config.KAFKA_BOOTSTRAP})
    consumer.subscribe([config.REQUEST_TOPIC])
    log.info("Consumer started. strategy=%s topic=%s", config.ACTIVE_STRATEGY, config.REQUEST_TOPIC)

    try:
        while True:
            msg = consumer.poll(timeout=1.0)
            if msg is None:
                continue
            if msg.error():
                if msg.error().code() != KafkaError._PARTITION_EOF:
                    log.error("Kafka error: %s", msg.error())
                continue

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
                consumer.commit(msg)
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

    # Also emit an ERROR result so the gateway error-rate accounting sees it
    try:
        event = AuditRequestEvent.model_validate_json(raw)
        err_result = AuditResultEvent(
            request_id=event.request_id,
            source_system=event.source_system,
            decision="ERROR",
            strategy=config.ACTIVE_STRATEGY,
            completed_at=datetime.now(timezone.utc).isoformat(),
            error=f"{type(error).__name__}: {error}",
        )
        producer.produce(
            config.RESULT_TOPIC,
            key=event.request_id.encode(),
            value=err_result.model_dump_json().encode(),
        )
    except ValueError:
        pass  # unparseable message: DLQ record is the only trace


if __name__ == "__main__":
    main()
