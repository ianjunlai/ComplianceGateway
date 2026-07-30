package edu.compliance.gateway.api;

import edu.compliance.gateway.api.dto.AuditRequest;
import edu.compliance.gateway.api.dto.AuditSubmittedResponse;
import edu.compliance.gateway.service.AuditProducerService;
import edu.compliance.gateway.service.MetricsService;
import edu.compliance.gateway.service.ResultStoreService;
import edu.compliance.gateway.service.SyncInferenceClient;
import jakarta.validation.Valid;
import org.springframework.context.support.DefaultMessageSourceResolvable;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.MethodArgumentNotValidException;
import org.springframework.web.bind.annotation.*;

import java.util.List;
import java.util.Map;

/**
 * Single public surface of the compliance gateway.
 *
 * Integration modes:
 *   POST /api/v1/audit                EDA          -> 202 + request_id
 *   POST /api/v1/audit/sync           sync-unbounded
 *   POST /api/v1/audit/sync-throttled sync-throttled (HTTP-layer queue)
 *
 * Support endpoints:
 *   GET  /api/v1/audit/{id}           result polling (JMeter E2E measurement)
 *   GET  /api/v1/metrics              dashboard snapshot
 */
@RestController
@RequestMapping("/api/v1")
public class AuditController {

    private final AuditProducerService producer;
    private final ResultStoreService resultStore;
    private final SyncInferenceClient syncClient;
    private final MetricsService metrics;

    public AuditController(AuditProducerService producer,
                           ResultStoreService resultStore,
                           SyncInferenceClient syncClient,
                           MetricsService metrics) {
        this.producer = producer;
        this.resultStore = resultStore;
        this.syncClient = syncClient;
        this.metrics = metrics;
    }

    // ---- EDA mode -------------------------------------------------------

    @PostMapping("/audit")
    public ResponseEntity<AuditSubmittedResponse> submit(@Valid @RequestBody AuditRequest request) {
        String requestId = producer.publish(request.sourceSystem(), request.auditQuery());
        return ResponseEntity.status(HttpStatus.ACCEPTED)
                .body(new AuditSubmittedResponse(requestId, "QUEUED"));
    }

    @GetMapping("/audit/{requestId}")
    public ResponseEntity<?> getResult(@PathVariable String requestId) {
        return resultStore.get(requestId)
                .<ResponseEntity<?>>map(ResponseEntity::ok)
                .orElseGet(() -> ResponseEntity.ok(
                        Map.of("request_id", requestId, "status", "PENDING")));
    }

    // ---- Synchronous baselines -------------------------------------------

    @PostMapping("/audit/sync")
    public ResponseEntity<?> submitSync(@Valid @RequestBody AuditRequest request) {
        try {
            return ResponseEntity.ok(
                    syncClient.callUnbounded(request.sourceSystem(), request.auditQuery()));
        } catch (Exception e) {
            return ResponseEntity.status(HttpStatus.GATEWAY_TIMEOUT)
                    .body(Map.of("error", e.getClass().getSimpleName() + ": " + e.getMessage()));
        }
    }

    @PostMapping("/audit/sync-throttled")
    public ResponseEntity<?> submitSyncThrottled(@Valid @RequestBody AuditRequest request) {
        try {
            return ResponseEntity.ok(
                    syncClient.callThrottled(request.sourceSystem(), request.auditQuery()));
        } catch (IllegalStateException e) {
            return ResponseEntity.status(HttpStatus.SERVICE_UNAVAILABLE)
                    .body(Map.of("error", e.getMessage()));
        } catch (Exception e) {
            return ResponseEntity.status(HttpStatus.GATEWAY_TIMEOUT)
                    .body(Map.of("error", e.getClass().getSimpleName() + ": " + e.getMessage()));
        }
    }

    // ---- Dashboard -------------------------------------------------------

    @GetMapping("/metrics")
    public Map<String, Object> metrics() {
        return metrics.snapshot();
    }

    // ---- Validation ------------------------------------------------------

    /**
     * Rejects a malformed payload at ingress with the offending field named.
     *
     * The default 400 body reports only that validation failed. Naming the
     * field matters here because the likeliest cause is a client sending the
     * wrong key — camelCase, say — which binds to null and is otherwise
     * indistinguishable from an omitted field.
     */
    @ExceptionHandler(MethodArgumentNotValidException.class)
    @ResponseStatus(HttpStatus.BAD_REQUEST)
    public Map<String, Object> onInvalidRequest(MethodArgumentNotValidException e) {
        // The messages carry the JSON field names; getField() would report the
        // Java property (sourceSystem), which is not what the client sent.
        List<String> violations = e.getBindingResult().getFieldErrors().stream()
                .map(DefaultMessageSourceResolvable::getDefaultMessage)
                .toList();
        return Map.of(
                "error", "invalid audit request",
                "violations", violations,
                "expected_fields", Map.of(
                        "source_system", "string, non-blank",
                        "audit_query", "string, non-blank"));
    }
}
