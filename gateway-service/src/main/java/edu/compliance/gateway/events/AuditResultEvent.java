package edu.compliance.gateway.events;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;

import java.util.List;
import java.util.Map;

/**
 * Audit_Result_Event consumed from Audit_Result_Topic.
 * Mirror of inference-service/common/schemas.py::AuditResultEvent.
 *
 * stageTimingsMs keys: queue_wait_ms, ner_ms, retrieval_ms, generation_ms, total_ms.
 */
@JsonIgnoreProperties(ignoreUnknown = true)
public record AuditResultEvent(
        @JsonProperty("request_id") String requestId,
        @JsonProperty("source_system") String sourceSystem,
        @JsonProperty("decision") String decision,          // APPROVE | DENY | UNKNOWN | ERROR
        @JsonProperty("reasoning") String reasoning,
        @JsonProperty("retrieved_chunk_ids") List<String> retrievedChunkIds,
        @JsonProperty("strategy") String strategy,
        @JsonProperty("stage_timings_ms") Map<String, Long> stageTimingsMs,
        @JsonProperty("completed_at") String completedAt,   // ISO-8601 UTC
        @JsonProperty("error") String error
) {
}
