package edu.compliance.gateway.service;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.web.client.ClientHttpRequestFactories;
import org.springframework.boot.web.client.ClientHttpRequestFactorySettings;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestClient;

import java.time.Duration;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.Semaphore;
import java.util.concurrent.TimeUnit;

/**
 * Synchronous integration baselines.
 *
 * sync-unbounded: every Tomcat worker thread blocks directly on the inference
 * HTTP call — the naive integration pattern.
 *
 * sync-throttled: identical call behind a FAIR semaphore (permits=1), i.e. an
 * HTTP-layer queue. Connections remain open while queued, which is exactly the
 * failure mode the EDA condition avoids (timeouts, retry amplification).
 */
@Service
public class SyncInferenceClient {

    private final RestClient restClient;
    private final Semaphore throttle;
    private final long acquireTimeoutMs;

    public SyncInferenceClient(
            @Value("${gateway.inference.sync-url}") String syncUrl,
            @Value("${gateway.inference.timeout-ms}") long timeoutMs,
            @Value("${gateway.throttle.permits}") int permits,
            @Value("${gateway.throttle.acquire-timeout-ms}") long acquireTimeoutMs) {
        var settings = ClientHttpRequestFactorySettings.DEFAULTS
                .withConnectTimeout(Duration.ofSeconds(5))
                .withReadTimeout(Duration.ofMillis(timeoutMs));
        this.restClient = RestClient.builder()
                .requestFactory(ClientHttpRequestFactories.get(settings))
                .baseUrl(syncUrl)
                .build();
        this.throttle = new Semaphore(permits, true);
        this.acquireTimeoutMs = acquireTimeoutMs;
    }

    /** sync-unbounded condition. */
    public Map<?, ?> callUnbounded(String sourceSystem, String auditQuery) {
        return doCall(sourceSystem, auditQuery);
    }

    /** sync-throttled condition. Throws IllegalStateException on queue timeout (-> 503). */
    public Map<?, ?> callThrottled(String sourceSystem, String auditQuery) {
        boolean acquired = false;
        try {
            acquired = throttle.tryAcquire(acquireTimeoutMs, TimeUnit.MILLISECONDS);
            if (!acquired) {
                throw new IllegalStateException("throttle-queue-timeout");
            }
            return doCall(sourceSystem, auditQuery);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IllegalStateException("throttle-interrupted", e);
        } finally {
            if (acquired) {
                throttle.release();
            }
        }
    }

    private Map<?, ?> doCall(String sourceSystem, String auditQuery) {
        Map<String, String> payload = Map.of(
                "request_id", UUID.randomUUID().toString(),
                "source_system", sourceSystem,
                "audit_query", auditQuery);
        return restClient.post()
                .contentType(MediaType.APPLICATION_JSON)
                .body(payload)
                .retrieve()
                .body(Map.class);
    }
}
