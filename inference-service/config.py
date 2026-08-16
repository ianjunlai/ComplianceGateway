"""Central configuration for the inference service. Every experimental parameter
is disclosed here."""
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

# --- Offline cloud models, on public legal text only. PROVIDER selects the
# client in common/llm_clients.py; switching provider needs only these vars
# plus that provider's key. See .env.example.
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
# Hybrid traversal depth. On 2Wiki 21% of gold pairs sit 3-4 relations apart
# and are invisible at 2, so it stays overridable.
GRAPH_HOPS = int(os.getenv("GRAPH_HOPS", "2"))
# Follow IMPLEMENTS one hop past the entity traversal. Off by default: every
# reported GDPR figure was produced without it.
HYBRID_FOLLOW_IMPLEMENTS = os.getenv("HYBRID_FOLLOW_IMPLEMENTS", "0") not in ("0", "false", "False")
# Query entity -> graph node.
ENTITY_LINK_TOP_K = int(os.getenv("ENTITY_LINK_TOP_K", "3"))

# Citation attachment: cited provisions enter the context directly instead of
# competing on similarity (see pipeline/attachment.py).
ATTACH_CITATIONS = os.getenv("ATTACH_CITATIONS", "1") not in ("0", "false", "False")
# Ollama's default window is small and truncates silently from the end, which
# is where attachments sit, so it must be set.
SLM_NUM_CTX = int(os.getenv("SLM_NUM_CTX", "16384"))
ATTACH_PER_CHUNK = int(os.getenv("ATTACH_PER_CHUNK", "2"))
ATTACH_CONTEXT_CAP = int(os.getenv("ATTACH_CONTEXT_CAP", "10"))
ENTITY_LINK_THRESHOLD = 0.75   # cosine sim tau, used only when TOP_K > 1
# HippoRAG E'.
SYNONYM_EDGES_PER_ENTITY = float(os.getenv("SYNONYM_EDGES_PER_ENTITY", "1.93"))
# Set to pin the cutoff instead of deriving it; used for the threshold ablation.
SYNONYM_THRESHOLD = (float(os.environ["SYNONYM_THRESHOLD"])
                     if os.getenv("SYNONYM_THRESHOLD") else None)
PPR_ALPHA = 0.5                # HippoRAG PPR edge probability; 1 - it restarts
LIGHTRAG_NEIGHBOUR_DECAY = 0.5  # score multiplier for one-hop-expanded evidence
# LightRAG publishes no passage ranking, but Recall@k needs one: rank admitted
# clauses by query similarity (True) or by the graph element that surfaced them
# (False, the original choice and the defect hybrid_graph.py was fixed for).
LIGHTRAG_RANK_BY_QUERY = os.getenv("LIGHTRAG_RANK_BY_QUERY", "1") not in ("0", "false", "False")

# --- Artifacts (built offline by ingestion.build_indexes) ---
# Overridable so a second corpus does not overwrite the first: the caches and
# matrices are keyed by chunk_id alone. The benchmark run uses artifacts_2wiki/.
ARTIFACTS_DIR = os.getenv(
    "ARTIFACTS_DIR", os.path.join(os.path.dirname(__file__), "artifacts"))

# Domain of the extraction and query-NER prompts.
EXTRACTION_PROFILE = os.getenv("EXTRACTION_PROFILE", "legal")
