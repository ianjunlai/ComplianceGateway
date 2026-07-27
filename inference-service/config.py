"""Central configuration for the inference service.

All experimentally relevant hyperparameters live here so they can be
reported as disclosed parameters.
"""
import os

from dotenv import load_dotenv

load_dotenv()

# --- Kafka (contract shared with gateway-service/application.yml) ---
KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "localhost:9092")
REQUEST_TOPIC = "Audit_Request_Topic"
RESULT_TOPIC = "Audit_Result_Topic"
DLQ_TOPIC = "Audit_DLQ_Topic"
CONSUMER_GROUP = "ai-inference-consumer"

# --- Storage ---
# Neo4j holds both the knowledge graph and the vectors: its native vector
# indexes let a single Cypher statement do similarity search and traversal in
# one round trip, and keep chunk/entity identity in one place.
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "compliance123")

# Vector index names (multi-paradigm indexes)
INDEX_CHUNKS = "chunk_vec"      # Vector RAG + chunk lookups
INDEX_ENTITIES = "entity_vec"   # entity linking + LightRAG low-level
INDEX_EDGES = "edge_vec"        # LightRAG high-level (relationship index)
VECTOR_DIM = 1024               # bge-large-en-v1.5

# --- Local models (online audit path -- never swapped for a cloud API) ---
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
SLM_MODEL = os.getenv("SLM_MODEL", "llama3.1:8b-instruct-q4_K_M")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5")

# --- Offline cloud models (public legal text / synthetic data / system outputs
# only). PROVIDER selects the client via common/llm_clients.py; MODEL is
# whatever that provider calls it. Switching provider is just these two env
# vars plus that provider's API key -- see .env.example.
EXTRACTION_PROVIDER = os.getenv("EXTRACTION_PROVIDER", "openai")
EXTRACTION_MODEL = os.getenv("EXTRACTION_MODEL", "gpt-4o")
JUDGE_PROVIDER = os.getenv("JUDGE_PROVIDER", "anthropic")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "claude-sonnet-5")
QA_GENERATION_PROVIDER = os.getenv("QA_GENERATION_PROVIDER", "openai")
QA_GENERATION_MODEL = os.getenv("QA_GENERATION_MODEL", "gpt-4o")

# --- Retrieval hyperparameters (disclosed experimental parameters) ---
ACTIVE_STRATEGY = os.getenv("ACTIVE_STRATEGY", "hybrid")
RETRIEVAL_K = 10               # ranked chunks retrieved per query (one list serves Recall@5 and @10)
GENERATION_CONTEXT_K = 5       # chunks the SLM actually sees (fixed, independent of retrieval K)
GRAPH_HOPS = 2                 # Hybrid traversal depth
ENTITY_LINK_THRESHOLD = 0.75   # cosine sim tau for query-entity -> graph-node
DEDUP_THRESHOLD = 0.90         # entity dedup during ingestion
PPR_ALPHA = 0.5                # HippoRAG PPR: probability of following an edge
                               # (1 - PPR_ALPHA = restart probability, tuned to
                               # 0.5 in the HippoRAG paper)
LIGHTRAG_NEIGHBOUR_DECAY = 0.5  # score multiplier for one-hop-expanded evidence

# --- Artifacts (built offline by ingestion.build_indexes) ---
ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
