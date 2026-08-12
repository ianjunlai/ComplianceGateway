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
# Hybrid traversal depth. Overridable because it is an ablation knob: measured
# on 2Wiki, 21% of gold passage pairs sit 3-4 relations apart and are invisible
# at 2.
GRAPH_HOPS = int(os.getenv("GRAPH_HOPS", "2"))
# Follow Chunk-[:IMPLEMENTS]->Chunk citations one hop past the entity traversal.
# Off by default: the edges exist only in the three-tier pilot corpus, and every
# GDPR figure reported so far was produced without them.
HYBRID_FOLLOW_IMPLEMENTS = os.getenv("HYBRID_FOLLOW_IMPLEMENTS", "0") not in ("0", "false", "False")
# Entry points the vec_* variants take before expanding. Swept on the pilot:
# the best value differed between corpora (5 there, 10 on GDPR) and the
# difference was inside the noise at n=38, so it is disclosed rather than tuned.
VECTOR_EXPAND_ENTRY_K = int(os.getenv("VECTOR_EXPAND_ENTRY_K", "5"))
# Query entity -> graph node. HippoRAG specifies the argmax, exactly one node
# per query entity with no threshold. That is set to 3 here, and the deviation
# is deliberate: the paper's rule assumes GPT-3.5-quality query NER, and this
# project extracts query entities with a local 1B model. Measured on
# 2WikiMultihopQA, switching from top-3 to the paper's argmax cost 12 points of
# R@5 on both hybrid (0.737 -> 0.619) and hippo_rag (0.646 -> 0.526), because a
# single wrong argmax leaves the traversal with no usable anchor whereas
# top-3 keeps a correct one in reach. Set to 1 to reproduce the paper's rule.
ENTITY_LINK_TOP_K = int(os.getenv("ENTITY_LINK_TOP_K", "3"))

# Citation attachment. A provision the retrieved clause cites is added to the
# generation context directly, rather than competing for a rank on query
# similarity -- see pipeline/attachment.py for why ranking cannot deliver it.
# Applied to every strategy, so it is a property of the platform rather than an
# experimental condition.
ATTACH_CITATIONS = os.getenv("ATTACH_CITATIONS", "1") not in ("0", "false", "False")
# Context window for the local SLM. Ollama's default is small and truncates
# silently from the end, which is where attached provisions are placed, so it
# has to be set rather than inherited. 16384 covers the worst case in this
# corpus (about 18k tokens for ten of the longest chunks) with room for the
# prompt; on an 80 GB card the KV cache for a 14B model at this length is a
# small fraction of memory.
SLM_NUM_CTX = int(os.getenv("SLM_NUM_CTX", "16384"))
ATTACH_PER_CHUNK = int(os.getenv("ATTACH_PER_CHUNK", "2"))
ATTACH_CONTEXT_CAP = int(os.getenv("ATTACH_CONTEXT_CAP", "10"))
ENTITY_LINK_THRESHOLD = 0.75   # cosine sim tau, used only when TOP_K > 1
# HippoRAG E'. The paper tunes a cosine cutoff of 0.8 on 100 MuSiQue training
# questions, with ColBERTv2/Contriever. That number transfers across neither
# encoder nor corpus: on bge-large, reaching the paper's published density needs
# 0.726 over 2WikiMultihopQA and 0.786 over the GDPR corpus, and the paper's own
# 0.8 yields a graph five times too sparse. So the DENSITY is the disclosed
# parameter and the cutoff is solved for per corpus at build time.
#   82,526 synonym edges over 42,694 entities on 2Wiki = 1.93 per entity.
# Entities are never merged on similarity -- see ingestion/dedup.py.
SYNONYM_EDGES_PER_ENTITY = float(os.getenv("SYNONYM_EDGES_PER_ENTITY", "1.93"))
# Set to pin the cutoff instead of deriving it; used for the threshold ablation.
SYNONYM_THRESHOLD = (float(os.environ["SYNONYM_THRESHOLD"])
                     if os.getenv("SYNONYM_THRESHOLD") else None)
PPR_ALPHA = 0.5                # HippoRAG PPR: probability of following an edge
                               # (1 - PPR_ALPHA = restart probability, tuned to
                               # 0.5 in the HippoRAG paper)
LIGHTRAG_NEIGHBOUR_DECAY = 0.5  # score multiplier for one-hop-expanded evidence
# LightRAG publishes no passage ranking -- it returns entities and relations and
# is evaluated on generation win-rate. Recall@k needs one, so: rank the clauses
# the dual-level retrieval admits by query-vector similarity (True), or by the
# similarity of whichever graph element surfaced them (False, the original
# choice, and the same defect hybrid_graph.py was fixed for).
LIGHTRAG_RANK_BY_QUERY = os.getenv("LIGHTRAG_RANK_BY_QUERY", "1") not in ("0", "false", "False")

# --- Artifacts (built offline by ingestion.build_indexes) ---
# Overridable so a second corpus can be built without overwriting the first:
# the extraction cache, HippoRAG matrices and chunk_texts.json are all keyed by
# chunk_id only, so two corpora sharing a directory would silently corrupt each
# other. The supplementary benchmark run sets this to artifacts_2wiki/.
ARTIFACTS_DIR = os.getenv(
    "ARTIFACTS_DIR", os.path.join(os.path.dirname(__file__), "artifacts"))

# Domain of the extraction and query-NER prompts. "legal" is what every
# reported GDPR result was produced with; "general" is domain-neutral and is
# used only for the public-benchmark validity check, where a legal extractor
# applied to Wikipedia would build an empty graph and make the graph strategies
# look broken for a reason unrelated to their implementation.
EXTRACTION_PROFILE = os.getenv("EXTRACTION_PROFILE", "legal")
