package edu.compliance.gateway.api.dto;

/** HTTP 202 body for the EDA mode. */
public record AuditSubmittedResponse(
        String requestId,
        String status
) {
}
