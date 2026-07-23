package edu.compliance.gateway.config;

import org.apache.kafka.clients.admin.NewTopic;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.kafka.config.TopicBuilder;

/**
 * Topic definitions. partitions = 1 is a deliberate experimental constraint:
 * Kafka parallelism equals partition count, and the edge deployment has a single
 * GPU consumer. 
 */
@Configuration
public class KafkaTopicConfig {

    @Value("${gateway.topics.request}")
    private String requestTopic;

    @Value("${gateway.topics.result}")
    private String resultTopic;

    @Value("${gateway.topics.dlq}")
    private String dlqTopic;

    @Bean
    public NewTopic auditRequestTopic() {
        return TopicBuilder.name(requestTopic).partitions(1).replicas(1).build();
    }

    @Bean
    public NewTopic auditResultTopic() {
        return TopicBuilder.name(resultTopic).partitions(1).replicas(1).build();
    }

    @Bean
    public NewTopic auditDlqTopic() {
        return TopicBuilder.name(dlqTopic).partitions(1).replicas(1).build();
    }
}
