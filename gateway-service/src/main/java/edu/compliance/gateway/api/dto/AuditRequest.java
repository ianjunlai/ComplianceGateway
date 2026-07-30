package edu.compliance.gateway.api.dto;

import com.fasterxml.jackson.annotation.JsonProperty;
import jakarta.validation.constraints.NotBlank;

/**
 * Inbound client payload: metadata + unstructured query only.
 * Deliberately contains no retrieval-strategy field — strategy selection is a
 * consumer-side configuration, invisible to federated clients.
 *
 * Field names are snake_case to match the Audit_Request_Event contract these
 * values are published under, so the whole external surface reads the same way.
 *
 * The constraints are load-bearing, not decoration: a field that fails to bind
 * arrives as null, and without them the gateway accepts the request, returns
 * 202 with a request id, and publishes an event whose null fields the Python
 * consumer cannot validate. That request then never completes — it is answered
 * as PENDING for as long as it is polled.
 */
public record AuditRequest(
        @JsonProperty("source_system")
        @NotBlank(message = "source_system is required")
        String sourceSystem,

        @JsonProperty("audit_query")
        @NotBlank(message = "audit_query is required")
        String auditQuery
) {
}
