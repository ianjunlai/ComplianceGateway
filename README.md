# ComplianceGateway

AI Compliance Gateway with Kafka Broker for Federated Education Systems.

## Architecture (4 layers)

```
JMeter / clients
      │  HTTP POST /api/v1/audit  →  202 Accepted
      ▼
gateway-service (Spring Boot :8080)
      │  Audit_Request_Topic            Audit_Result_Topic
      ▼                                        ▲
Kafka broker (Docker :9092) ───────────────────┤
      │ concurrency = 1                        │
      ▼                                        │
inference-service (Python)  ── GraphRAG pipeline ── Ollama (SLM :11434)
      │
      ▼
Neo4j (:7687) — knowledge graph + vector indexes
```

Neo4j stores both the graph and the embeddings through its native vector
indexes, so a single Cypher statement can combine similarity search with
traversal.

Three integration modes: `/audit` (EDA), `/audit/sync` (unbounded),
`/audit/sync-throttled` (HTTP-layer queue, permits=1).

## Prerequisites

- Docker Desktop (Kafka, Neo4j) — no Docker permission? see [SERVER_DEPLOYMENT.md](SERVER_DEPLOYMENT.md)
- Java 21 + Maven (gateway-service)
- Python 3.11 (inference-service, dataset scripts)
- [Ollama](https://ollama.com) with the SLM pulled:
  `ollama pull llama3.1:8b-instruct-q4_K_M`
- API keys in `inference-service/.env` (copy from `.env.example`):
  `OPENAI_API_KEY` (offline extraction + QA generation), `ANTHROPIC_API_KEY` (judge)

## Quick start

```bash
# 1. Infrastructure
docker compose up -d

# 2. Python environment
cd inference-service
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt

# 3. Build indexes (offline, one-off; requires corpus in dataset/corpus/)
python -m ingestion.build_indexes

# 4. Start the AI consumer (EDA mode)
python consumer_main.py
#    ...and the sync API (for the synchronous baselines)
uvicorn sync_api:app --port 8000

# 5. Gateway
cd ../gateway-service
mvn spring-boot:run

# 6. Dashboard
cd ../dashboard && python -m http.server 5173
# open http://localhost:5173
```

## Repository layout

| Path | Language | Contents |
|---|---|---|
| `gateway-service/` | Java 21 / Spring Boot 3 | REST endpoints, Kafka producer, result store |
| `inference-service/` | Python 3.11 | Kafka consumer, GraphRAG pipelines, indexing, evaluation |
| `dataset/` | Python | Legal corpus and synthetic QA generation |
| `loadtest/` | JMeter | Concurrent load test plans |
| `dashboard/` | HTML/JS | Live monitoring page |
