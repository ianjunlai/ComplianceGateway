# ComplianceGateway

AI Compliance Gateway with Kafka Broker for Federated Education Systems.

Audits data-transfer requests against a three-tier legal corpus — the GDPR,
three national data protection acts, and four university policies — using a
language model held on the institution's own hardware, because the personal
data involved cannot be sent to a third-party API.

## Architecture (4 layers)

```
JMeter / clients
      │  HTTP POST /api/v1/audit  →  202 Accepted
      ▼
gateway-service (Spring Boot :8080, :28080 on the measurement server)
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

## Retrieval

Five conditions: `zero_shot` (no retrieval), `vector_rag` (dense), and
`hybrid`, `light_rag`, `hippo_rag` reproducing three published GraphRAG
paradigms. Two mechanisms apply to all four retrieval conditions alike, so they
are properties of the platform rather than experimental variables:

- **Jurisdiction scoping** — candidates are restricted to EU law plus the state
  of the requesting institution, read from the request envelope.
- **Citation attachment** — provisions a retrieved clause cites are inserted
  after it rather than made to compete on similarity.

## Prerequisites

- Docker Desktop (Kafka, Neo4j) — no Docker permission? see [SERVER_DEPLOYMENT.md](SERVER_DEPLOYMENT.md)
- Java 21 + Maven (gateway-service)
- Python 3.11 (inference-service, dataset scripts)
- [Ollama](https://ollama.com) with the SLM pulled:
  `ollama pull qwen2.5:14b-instruct-q8_0` (about 16 GB of VRAM)
- API key in `inference-service/.env` (copy from `.env.example`). Offline stages
  only — extraction, question generation and the judge. The audit path never
  calls a cloud API.

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

# 4. Load the citation edges into the graph
python -m ingestion.load_implements --edges ../dataset/corpus/full_citations.json

# 5. Start the AI consumer (EDA mode)
python consumer_main.py
#    ...and the sync API (for the synchronous baselines)
uvicorn sync_api:app --port 8000

# 6. Gateway
cd ../gateway-service
mvn spring-boot:run
```

To run the whole evaluation instead of the services, use `./run_all.sh`, which
drives `run_full_experiment.sh` (E1 retrieval, E2 decisions) then
`loadtest/run_e3.sh` (E3 load). See [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

## Repository layout

| Path | Language | Contents |
|---|---|---|
| `gateway-service/` | Java 21 / Spring Boot 3 | REST endpoints, Kafka producer, result store |
| `inference-service/` | Python 3.11 | Kafka consumer, GraphRAG pipelines, indexing, evaluation |
| `dataset/` | Python | Legal corpus, citation extraction, question generation |
| `loadtest/` | JMeter | Concurrent load test plans |
| `dashboard/` | HTML/JS | Live monitoring page |
