package edu.compliance.gateway.service;

import com.fasterxml.jackson.databind.ObjectMapper;
import edu.compliance.gateway.events.AuditResultEvent;
import org.springframework.kafka.annotation.KafkaListener;
import org.springframework.stereotype.Service;

import java.util.Map;
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Result store: consumes Audit_Result_Topic so that
 *  - JMeter can measure async E2E latency by polling GET /api/v1/audit/{id}
 *  - the dashboard has a data source.
 *
 * In-memory map is sufficient for the experiment (single gateway instance, bounded run length).
 */
@Service
public class ResultStoreService {

    private final Map<String, AuditResultEvent> results = new ConcurrentHashMap<>();
    private final ObjectMapper objectMapper;
    private final MetricsService metrics;

    public ResultStoreService(ObjectMapper objectMapper, MetricsService metrics) {
        this.objectMapper = objectMapper;
        this.metrics = metrics;
    }

    @KafkaListener(topics = "${gateway.topics.result}")
    public void onResult(String message) {
        try {
            AuditResultEvent event = objectMapper.readValue(message, AuditResultEvent.class);
            results.put(event.requestId(), event);
            metrics.recordCompleted(event);
        } catch (Exception e) {
            metrics.recordError("result-deserialization: " + e.getMessage());
        }
    }

    public Optional<AuditResultEvent> get(String requestId) {
        return Optional.ofNullable(results.get(requestId));
    }
}
