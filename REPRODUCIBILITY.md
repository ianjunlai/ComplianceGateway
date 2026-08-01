# Reproducibility

Everything needed to rebuild the system and repeat the experiments: software
versions, model versions, all configuration, the data, and the run procedure.
See [README.md](README.md) for a shorter quick start.

## 1. Software versions

| Component | Version |
|---|---|
| Java | 21 (LTS) |
| Python | 3.11 |
| Docker | Desktop (any recent) — optional, see §6 Step 3b |
| Kafka | `apache/kafka:3.7.0` (KRaft, single broker) |
| Neo4j | `neo4j:5.24-community` |
| Spring Boot | 3.3.5 |
| Ollama | 0.4 or later |
| Apache JMeter | 5.6 |

Python dependencies are listed in [inference-service/requirements.txt](inference-service/requirements.txt)
and Java dependencies in [gateway-service/pom.xml](gateway-service/pom.xml).
Before final submission, freeze exact versions with `pip freeze > requirements.lock.txt`
so the environment can be recreated byte-for-byte.

### Measurement hardware

All reported results — E1, E2 and E3 — were produced on:

| | |
|---|---|
| CPU | 2 × Intel Xeon Silver 4314 @ 2.40 GHz (32 cores / 64 threads) |
| Memory | 1.0 TiB |
| GPU | NVIDIA A100 80 GB PCIe, **pinned to device 1** via `CUDA_VISIBLE_DEVICES=1` |
| OS | Ubuntu 20.04.6 LTS, kernel 5.15 |

This is a shared machine, which matters only for E3: latency and throughput are
sensitive to another tenant landing on the same GPU, so `nvidia-smi` is recorded
before and after every concurrency level and any level that overlapped a
neighbour's job is re-run. E1 and E2 report retrieval and faithfulness scores,
which do not depend on timing, so contention cannot affect them.

**A GPU is not optional for the online path.** Ollama's official builds support
CUDA and ROCm only; on a machine with Intel integrated graphics it silently falls
back to CPU. Decode of `llama3.1:8b-instruct-q4_K_M` is then bound by memory
bandwidth — on dual-channel DDR5-5600 (~89.6 GB/s against a 4.9 GB model) that is
roughly 10 tok/s, and a single RAG query costs on the order of 90 s rather than 6.
E1/E2 merely become an overnight job at that speed, but **E3 stops measuring what
it is for**: with `inference.timeout-ms` at 240 s and no throughput gain from CPU
concurrency, the sync-unbounded condition returns 100% 504 from C=10 upward
instead of degrading gradually, and the EDA backlog at C=100 takes hours to drain.

## 2. Model versions

| Role | Model | Developed by | Served via | Where it runs |
|---|---|---|---|---|
| Online inference (audit path) | `llama3.1:8b-instruct-q4_K_M` | Meta | Ollama | Local GPU |
| Embeddings | `BAAI/bge-large-en-v1.5` (1024-dim) | BAAI | local | Local |
| Offline graph extraction | `qwen-plus` | Alibaba | DashScope | Cloud, public text only |
| Synthetic QA generation | `deepseek-r1` | DeepSeek | DashScope | Cloud, public text only |
| Faithfulness judge | `qwen-max` | Alibaba | DashScope | Cloud, synthetic data only |

**Provider is not the same as vendor.** All three cloud roles use
`PROVIDER=alibaba`, but that names the API endpoint (DashScope), not who built
the model: DashScope hosts third-party models alongside Alibaba's own, and
`deepseek-r1` is DeepSeek's. The judge therefore comes from a different model
family than both the data generator and the system under evaluation
(Llama 3.1), which is what mitigates same-source preference bias — describe it
by model family in the thesis, not by the `PROVIDER` value, or a reader will
conclude the generator and judge are related when they are not.

`.env.example` still ships the original OpenAI/Anthropic defaults. To reproduce
the reported runs, set the five variables above explicitly and supply
`ALIBABA_API_KEY`.

**Laptop substitution.** Development and smoke-testing on a machine without a
capable GPU used `SLM_MODEL=llama3.2:1b`. That model is not fit for the
reported experiments — it fabricates clause citations and produces decisions
that contradict their own stated reasoning. Set `SLM_MODEL` back to the 8B
model before any run whose numbers will be reported.

The cloud models are used only offline, on public legal text and synthetic
data. No request containing personal data is sent to a cloud model. Provider
and model are set by environment variables (see `.env.example`); switching
provider needs only those variables plus the matching API key.

## 3. Configuration and hyperparameters

All experimentally relevant parameters live in
[inference-service/config.py](inference-service/config.py) so they can be
reported and changed in one place.

| Parameter | Value | Meaning |
|---|---|---|
| `RETRIEVAL_K` | 10 | ranked clauses retrieved per query |
| `GENERATION_CONTEXT_K` | 5 | clauses shown to the SLM |
| `GRAPH_HOPS` | 2 | Hybrid traversal depth |
| `ENTITY_LINK_THRESHOLD` | 0.75 | min cosine similarity to link a query entity to a node |
| `DEDUP_THRESHOLD` | 0.90 | min cosine similarity to merge two entities |
| `PPR_ALPHA` | 0.5 | HippoRAG: edge-follow probability (1 − value = restart) |
| `LIGHTRAG_NEIGHBOUR_DECAY` | 0.5 | LightRAG: score multiplier for one-hop-expanded clauses |
| `LOW_LEVEL_HITS_PER_MENTION` | 3 | LightRAG: entity hits kept per seed |
| `MAX_CHUNK_TOKENS` | 800 | oversized-chunk split threshold (ingestion) |
| generation temperature | 0 | constrained decoding, for reproducibility |

Fixed random seeds: synthetic QA generation and the statistical bootstrap both
use seed 42.

## 4. Data

- **Corpus** (`dataset/corpus/`): GDPR English text (articles and selected
  recitals) and four university data-protection policies, **345 chunks total**:

  | Source | Chunks |
  |---|---|
  | GDPR articles + recitals | 288 |
  | Trinity College Dublin | 20 |
  | Cambridge | 19 |
  | Göttingen | 14 |
  | Limerick | 4 |

  Record each document's source URL and retrieval date in
  `dataset/corpus/sources.md`. See `dataset/corpus/README.md` for the layout.
- **Synthetic evaluation set** (`dataset/qa_dataset.json`): 160 queries
  generated by `dataset/generate_qa.py`; ground truth is fixed by construction.
  Stratified single 56 / multi 56 / trap 32 / unanswerable 16. The schema is
  described in the thesis (§4.2.4).

The corpus is frozen before indexing and dataset generation: changing the
source text would shift chunk boundaries and invalidate the ground-truth
chunk IDs.

## 5. Determinism notes

- The knowledge graph is built **once** and reused by every strategy, because
  LLM extraction is not perfectly repeatable. Rebuilding per strategy would
  let the graph, rather than the retrieval method, explain differences.
- What makes "once" enforceable across machines is
  `artifacts/extraction_cache.json`: every chunk's extraction result is stored
  as it is produced, keyed by `provider:model`, and replayed on later runs so
  no API call is repeated. Carrying that file to another machine reproduces the
  *same* graph rather than a similar one — see §6 step 4.
- Local generation runs at temperature 0, so it is near-deterministic; exact
  reproduction is still subject to GPU floating-point non-determinism.
- After building the graph, **audit `artifacts/dedup_report.json`** to confirm
  no legally distinct entities were merged (for example, *controller* vs
  *processor*).

## 6. Procedure

```bash
# 1. Infrastructure
docker compose up -d                     # Kafka + Neo4j
#    no Docker (shared server, no root)? see "Step 3b" below

# 2. Python environment
cd inference-service
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt

# 3. Local model + keys
ollama pull llama3.1:8b-instruct-q4_K_M
copy .env.example .env
#    then set ALIBABA_API_KEY and the five model variables from §2 --
#    the shipped defaults point at OpenAI/Anthropic and will not reproduce
#    the reported runs

# 4. MOVING TO ANOTHER MACHINE: copy artifacts/ across FIRST (see below)

# 5. Build the graph and indexes (offline; needs corpus in dataset/corpus/)
python -m ingestion.build_indexes         # -> Neo4j graph + vector indexes, artifacts/, cost report

# 6. Generate the evaluation dataset -- ONLY if you do not already have one
python ../dataset/generate_qa.py --n 160  # -> dataset/qa_dataset.json + verification_sample.json

# 7. Run the services -- ONE inference backend at a time, see note below
#    gateway
cd ../gateway-service && mvn spring-boot:run
#    then EITHER the consumer (EDA condition) ...
cd ../inference-service && python consumer_main.py
#    ... OR the sync API (both synchronous conditions)
uvicorn sync_api:app --port 8000

# 8. Reasoning-quality and retrieval experiments (E1, E2)
python -m evaluation.run_eval --strategy zero_shot  --judge --run-id <id>
python -m evaluation.run_eval --strategy vector_rag --judge --run-id <id>
python -m evaluation.run_eval --strategy hybrid     --judge --run-id <id>
python -m evaluation.run_eval --strategy light_rag  --judge --run-id <id>
python -m evaluation.run_eval --strategy hippo_rag  --judge --run-id <id>
#    -> results/<strategy>-<run-id>.json  (+ .jsonl written as it goes)
#    interrupted? rerun the SAME command with --resume appended

# 9. Load experiment (E3)
jmeter -n -t ../loadtest/compliance_gateway.jmx \
       -JTHREADS=10 -JRAMP=10 -JDURATION=120 -l results/eda-c10.jtl
#    one condition per run; enable the matching Thread Group in the GUI first
#    full profile, failure-mode taxonomy and sizing: loadtest/README.md
```

Steps 5 and 6 need only a cloud API key and Docker — no GPU. Steps 7–9 need the
local GPU. Run each evaluation strategy in its own process, since the active
strategy is fixed when the process starts.

### Steps 7 and 9 — one inference backend at a time

The consumer and `sync_api` each load their own copy of the 1.3 GB embedding
model and each drive the same single GPU, so running both costs 4–6 GB before
Kafka, Neo4j, the gateway, Ollama and JMeter's own 1 GB heap, and a measurement
taken with both up describes their contention rather than the condition under
test. On an 11.6 GB machine this drove a single inference from 160 s to over
26 minutes through paging. Bring up whichever backend the current condition
needs, and stop it before switching.

Before starting a level, rehearse the JMeter plan against
`loadtest/stub_inference.py` — same contract, fixed delay instead of a pipeline,
so extractor and timeout mistakes surface in seconds rather than after an hour
of real inference. Then measure one request through `POST /api/v1/audit/sync`
against the real backend and size the concurrency levels from it: EDA drain time
is roughly `threads × single-request time`, which is what sets the wall-clock
cost of the whole experiment.

### Step 3b — deployment without Docker

Shared GPU servers frequently forbid Docker. Nothing in this project requires it:
every service ships a tarball that runs from `$HOME` with no root, and every
endpoint the code talks to is read from an environment variable, so the whole
stack relocates without a code change.

**Check for a container runtime first.** `which apptainer singularity podman
nerdctl` — research clusters commonly carry Apptainer, and if it is there the
compose images run directly (`apptainer run docker://neo4j:5.24-community`),
which is closer to the reference deployment than reassembling it by hand. Also
run `ss -ltn` and see which of 7474 / 7687 / 9092 / 8080 / 8000 / 11434 are
already occupied; remap the collisions rather than contending for the port.

Otherwise, from tarballs:

```bash
conda create -n cg python=3.11 && conda activate cg
conda install -c conda-forge openjdk=21 maven   # Neo4j 5.24 and the gateway both need 21
pip install -r inference-service/requirements.txt

# Neo4j 5.24 community (unix tarball) — set listen addresses in conf/neo4j.conf if remapped
bin/neo4j-admin dbms set-initial-password compliance123 && bin/neo4j start

# Kafka 3.7.0, KRaft single node. Set auto.create.topics.enable=false in
# config/kraft/server.properties to match docker-compose.yml: the gateway creates
# its three topics explicitly with partitions=1, and silent auto-creation would
# give them the broker default instead
bin/kafka-storage.sh format -t $(bin/kafka-storage.sh random-uuid) -c config/kraft/server.properties
bin/kafka-server-start.sh -daemon config/kraft/server.properties

# Ollama tarball bundles its own CUDA runtime — no root, no systemd unit
export OLLAMA_MODELS=$HOME/ollama/models
CUDA_VISIBLE_DEVICES=1 $HOME/ollama/bin/ollama serve &
ollama pull llama3.1:8b-instruct-q4_K_M
```

Then point the configuration at whatever ports survived — `NEO4J_URI`,
`KAFKA_BOOTSTRAP` and `OLLAMA_HOST` in `.env`; `SERVER_PORT`,
`SPRING_KAFKA_BOOTSTRAP_SERVERS` and `GATEWAY_INFERENCE_SYNC_URL` as environment
overrides for the gateway.

**Set `CUDA_VISIBLE_DEVICES` on the Python processes too, not just on Ollama.**
The embedding model is constructed without a device argument
([pipeline/embeddings.py](inference-service/pipeline/embeddings.py)), so
sentence-transformers auto-selects `cuda:0` — on a shared box that is whichever
card other tenants are already using. Pinning both to the same free device is
also what keeps the E3 measurement attributable to one GPU.

### Step 4 — moving to another machine

`inference-service/artifacts/` and `.env` are both gitignored, so a fresh
`git clone` has neither. The Neo4j graph lives in a Docker volume and is not in
git either, so it must be rebuilt on the new machine regardless — and
`artifacts/extraction_cache.json` is what makes that rebuild free and identical.
**Without it, step 5 silently re-runs all 345 extractions**, re-spending the
~430k-token, ~90-minute budget and producing a graph that differs from the one
the reported results were measured on.

Copy by hand, before step 5:

| Path | Size | Contents |
|---|---|---|
| `inference-service/artifacts/` | 2.2 MB | extraction cache, HippoRAG matrices, dedup report, chunk texts |
| `inference-service/.env` | 1 KB | API keys and model selection |

Verify the transfer worked by reading step 5's log: it must report
**`345/345 chunks served from cache`** and make no API calls. If it starts
calling the API, stop it — the cache did not transfer.

Then confirm the rebuild matches before committing hours of inference:

```bash
# graph: 345 Chunk, 1673 Entity, 3684 RELATES, 4394 MENTIONED_IN
# SHOW INDEXES -> chunk_vec, entity_vec, edge_vec all ONLINE
python evaluation/ablation/compare_all_strategies.py
#   expect  vector_rag 0.537 / hybrid 0.537 / light_rag 0.196 / hippo_rag 0.171
```

A mismatch means the graph is not the one the reported retrieval numbers came
from, and the evaluation would not be comparable to them.

## 7. Outputs

| File | Produced by | Contents |
|---|---|---|
| `artifacts/extraction_cache.json` | build_indexes | per-chunk extraction results keyed by `provider:model`; replayed instead of re-calling the API. **The one file that must survive a move between machines.** |
| `artifacts/chunk_texts.json` | build_indexes | chunk_id → text, used by the faithfulness judge |
| `artifacts/dedup_report.json` | build_indexes | every cross-name entity merge, for audit |
| `artifacts/indexing_cost_report.json` | build_indexes | per-stage time, tokens, embeddings, storage. Empty `tokens` on a fully cached run is expected — see `artifacts/extraction_token_usage_note.md` for the measured figures to cite. |
| `artifacts/hippo_*.npz`, `*.json` | build_indexes | HippoRAG matrices and maps |
| `dataset/qa_dataset.json` | generate_qa | the evaluation set |
| `dataset/verification_sample.json` | generate_qa | the sample for human review |
| `results/<strategy>-<run-id>.jsonl` | run_eval | one row per query, appended as it completes; the resume point |
| `results/<strategy>-<run-id>.json` | run_eval | per-query outputs and summary metrics |
| `docs/retrieval_ablation.md` | — | the retrieval ablation write-up; scripts in `inference-service/evaluation/ablation/` |
