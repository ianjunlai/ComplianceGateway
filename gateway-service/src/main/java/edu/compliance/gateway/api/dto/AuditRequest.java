package edu.compliance.gateway.api.dto;

/**
 * Inbound client payload: metadata + unstructured query only.
 * Deliberately contains no retrieval-strategy field — strategy selection is a
 * consumer-side configuration, invisible to federated clients.
 */
public record AuditRequest(
        String sourceSystem,
        String auditQuery
) {
}
