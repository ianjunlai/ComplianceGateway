package edu.compliance.gateway;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

/**
 * Event-Driven AI Compliance Gateway.
 *
 * Exposes three integration modes:
 *   POST /api/v1/audit                -> EDA (Kafka, HTTP 202)
 *   POST /api/v1/audit/sync           -> synchronous, unbounded concurrency
 *   POST /api/v1/audit/sync-throttled -> synchronous, HTTP-layer queue (permits=1)
 */
@SpringBootApplication
public class GatewayApplication {

    public static void main(String[] args) {
        SpringApplication.run(GatewayApplication.class, args);
    }
}
