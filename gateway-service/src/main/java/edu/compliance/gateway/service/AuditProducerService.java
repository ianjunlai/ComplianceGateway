package edu.compliance.gateway.service;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import edu.compliance.gateway.events.AuditRequestEvent;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.kafka.core.KafkaTemplate;
import org.springframework.stereotype.Service;

import java.time.Instant;
import java.time.temporal.ChronoUnit;
import java.util.UUID;

/**
 * EDA mode publisher: wraps the payload into an
 * Audit_Request_Event and publishes it to Audit_Request_Topic.
 */
@Service
public class AuditProducerService {

    private final KafkaTemplate<String, String> kafkaTemplate;
    private final ObjectMapper objectMapper;
    private final MetricsService metrics;

    @Value("${gateway.topics.request}")
    private String requestTopic;

    public AuditProducerService(KafkaTemplate<String, String> kafkaTemplate,
                                ObjectMapper objectMapper,
                                MetricsService metrics) {
        this.kafkaTemplate = kafkaTemplate;
        this.objectMapper = objectMapper;
        this.metrics = metrics;
    }

    /** Publishes the event and returns the assigned request UUID. */
    public String publish(String sourceSystem, String auditQuery) {
        String requestId = UUID.randomUUID().toString();
        // Contract: ISO-8601 UTC at millisecond precision. Java's default
        // Instant#toString() emits nanosecond precision, which older Python
        // fromisoformat implementations reject — truncate to keep the wire
        // format unambiguous regardless of the consumer's Python version.
        AuditRequestEvent event = new AuditRequestEvent(
                requestId, sourceSystem,
                Instant.now().truncatedTo(ChronoUnit.MILLIS).toString(), auditQuery);
        try {
            String json = objectMapper.writeValueAsString(event);
            // request_id as the record key: stable ordering within the single partition
            kafkaTemplate.send(requestTopic, requestId, json);
            metrics.recordSubmitted(requestId);
            return requestId;
        } catch (JsonProcessingException e) {
            throw new IllegalStateException("Failed to serialize AuditRequestEvent", e);
        }
    }
}
