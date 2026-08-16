package edu.compliance.gateway.api.dto;

import com.fasterxml.jackson.annotation.JsonProperty;

/**
 * HTTP 202 body for the EDA mode.
 */
public record AuditSubmittedResponse(
        @JsonProperty("request_id") String requestId,
        @JsonProperty("status") String status
) {
}
