package edu.compliance.gateway.api.dto;

import com.fasterxml.jackson.annotation.JsonProperty;

/**
 * HTTP 202 body for the EDA mode.
 *
 * request_id is snake_case like the rest of the API surface; the JMeter plan
 * extracts it from here to drive the result-polling loop.
 */
public record AuditSubmittedResponse(
        @JsonProperty("request_id") String requestId,
        @JsonProperty("status") String status
) {
}
