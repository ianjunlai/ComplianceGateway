package edu.compliance.gateway.events;

import com.fasterxml.jackson.annotation.JsonProperty;

/**
 * Audit_Request_Event, serialized as JSON onto Audit_Request_Topic.
 * Field names are the cross-language contract with the Python consumer
 * (inference-service/common/schemas.py) — keep them in sync.
 */
public record AuditRequestEvent(
        @JsonProperty("request_id") String requestId,
        @JsonProperty("source_system") String sourceSystem,
        @JsonProperty("timestamp") String timestamp,       // ISO-8601 UTC
        @JsonProperty("audit_query") String auditQuery
) {
}
