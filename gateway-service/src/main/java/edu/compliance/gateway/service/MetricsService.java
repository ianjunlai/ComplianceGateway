package edu.compliance.gateway.service;

import edu.compliance.gateway.events.AuditResultEvent;
import org.springframework.stereotype.Service;

import java.time.Instant;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Deque;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Lightweight gateway-side metrics: queue depth, counters,
 * gateway-observed E2E latency, and a rolling window of recent results for
 * the dashboard. Queue depth here = submitted - completed - errors, valid
 * because the experiment runs a single gateway instance.
 *
 * Kafka-side consumer lag (broker truth) is additionally recorded by the
 * loadtest tooling; see loadtest/README.md.
 */
@Service
public class MetricsService {

    private static final int RECENT_WINDOW = 50;

    private final AtomicLong submitted = new AtomicLong();
    private final AtomicLong completed = new AtomicLong();
    private final AtomicLong errors = new AtomicLong();

    /** requestId -> gateway ingress instant, for E2E latency (EDA path). */
    private final Map<String, Instant> submittedAt = new ConcurrentHashMap<>();

    /** Rolling window of recent completions (dashboard table). */
    private final Deque<Map<String, Object>> recent = new ArrayDeque<>();

    public void recordSubmitted(String requestId) {
        submitted.incrementAndGet();
        submittedAt.put(requestId, Instant.now());
    }

    public void recordCompleted(AuditResultEvent event) {
        if (event.error() != null && !event.error().isBlank()) {
            errors.incrementAndGet();
        } else {
            completed.incrementAndGet();
        }
        Instant start = submittedAt.remove(event.requestId());
        Long e2eMs = start == null ? null : Instant.now().toEpochMilli() - start.toEpochMilli();

        // snake_case throughout, matching AuditResultEvent and the stage-timing
        // keys nested inside it — this map is hand-built, so the convention has
        // to be applied here rather than inherited from the record.
        Map<String, Object> row = new LinkedHashMap<>();
        row.put("request_id", event.requestId());
        row.put("source_system", event.sourceSystem());
        row.put("decision", event.decision());
        row.put("strategy", event.strategy());
        row.put("e2e_ms", e2eMs);
        row.put("stage_timings_ms", event.stageTimingsMs());
        row.put("completed_at", event.completedAt());
        synchronized (recent) {
            recent.addFirst(row);
            if (recent.size() > RECENT_WINDOW) {
                recent.removeLast();
            }
        }
    }

    public void recordError(String detail) {
        errors.incrementAndGet();
    }

    public Map<String, Object> snapshot() {
        Map<String, Object> snap = new LinkedHashMap<>();
        long sub = submitted.get();
        long done = completed.get();
        long err = errors.get();
        snap.put("submitted", sub);
        snap.put("completed", done);
        snap.put("errors", err);
        snap.put("queue_depth", Math.max(0, sub - done - err));
        List<Map<String, Object>> recentCopy;
        synchronized (recent) {
            recentCopy = new ArrayList<>(recent);
        }
        snap.put("recent", recentCopy);
        return snap;
    }
}
