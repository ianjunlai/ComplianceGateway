package edu.compliance.gateway.api.dto;

import com.fasterxml.jackson.annotation.JsonProperty;
import jakarta.validation.constraints.NotBlank;

public record AuditRequest(
        @JsonProperty("source_system")
        @NotBlank(message = "source_system is required")
        String sourceSystem,

        @JsonProperty("audit_query")
        @NotBlank(message = "audit_query is required")
        String auditQuery
) {
}
