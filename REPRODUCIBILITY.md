# Reproducibility

Everything needed to rebuild the system and repeat the experiments: software
versions, model versions, all configuration, the data, and the run procedure.
See [README.md](README.md) for a shorter quick start.

## 1. Software versions

| Component | Version |
|---|---|
| Java | 21 (LTS) |
| Python | 3.11 |
| Docker | Desktop (any recent) |
| Kafka | `apache/kafka:3.7.0` (KRaft, single broker) |
| Neo4j | `neo4j:5.24-community` |
| Spring Boot | 3.3.5 |
| Ollama | 0.4 or later |
| Apache JMeter | 5.6 |

Python dependencies are listed in [inference-service/requirements.txt](inference-service/requirements.txt)
and Java dependencies in [gateway-service/pom.xml](gateway-service/pom.xml).
Before final submission, freeze exact versions with `pip freeze > requirements.lock.txt`
so the environment can be recreated byte-for-byte.

## 2. Model versions

| Role | Model | Developed by | Served via | Where it runs |
|---|---|---|---|---|
| Online inference (audit path) | `llama3.1:8b-instruct-q4_K_M` | Meta | Ollama | Local GPU |
| Embeddings | `BAAI/bge-large-en-v1.5` (1024-dim) | BAAI | local | Local |
| Offline graph extraction | `deepseek-v3.2` | DeepSeek | DashScope | Cloud, public text only |
| Synthetic QA generation | `qwen3.7-max` | Alibaba | DashScope | Cloud, public text only |
| Faithfulness judge | `glm-5.2` | Zhipu AI | DashScope | Cloud, synthetic data only |

**Provider is not the same as vendor.** All three cloud roles use
`PROVIDER=alibaba`, but that names the API endpoint (DashScope), not who built
the model: DashScope hosts third-party models alongside Alibaba's own. The
judge (Zhipu), the data generator (Alibaba) and the system under evaluation
(Llama 3.1) therefore come from three different model families, which is what
mitigates same-source preference bias — describe it by model family in the
thesis, not by the `PROVIDER` value, or a reader will conclude the generator
and judge are related when they are not.

**Report the model that answered, not the one requested.** Some names are
floating aliases — `qwen-plus` and `qwen-plus-latest` move with releases — so
the provider may serve a different, pinned model. `complete_json` records the
served name in its usage dict and warns once per run when it differs.

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
| `RETRIEVAL_K` | 10 | ranked clauses retrieved per query (one list serves R@2/@5/@10) |
| `GENERATION_CONTEXT_K` | 5 | clauses shown to the SLM **and to the judge** |
| `GRAPH_HOPS` | 2 | Hybrid traversal depth |
| `ENTITY_LINK_THRESHOLD` | 0.75 | min cosine similarity to link a query entity to a node |
| `ENTITY_LINK_TOP_K` | 3 | nodes linked per query entity (see deviation note below) |
| `PPR_ALPHA` | 0.5 | HippoRAG: edge-follow probability (1 − value = restart) |
| `SYNONYM_EDGES_PER_ENTITY` | 1.93 | HippoRAG E′: target synonym-edge density; τ is derived from it |
| `LIGHTRAG_NEIGHBOUR_DECAY` | 0.5 | LightRAG: score multiplier for one-hop-expanded clauses |
| `LIGHTRAG_RANK_BY_QUERY` | true | LightRAG: rank admitted clauses by query similarity |
| `HYBRID_FOLLOW_IMPLEMENTS` | false | whether Hybrid's traversal also follows citation edges |
| `EXTRACTION_PROFILE` | `legal` | selects the extraction/NER prompt pair (`legal` \| `general`) |
| generation temperature | 0 | constrained decoding, for reproducibility |

`GENERATION_CONTEXT_K` governs both the SLM's context and the judge's reference
context, and it must: a judge shown less than the SLM saw marks grounded claims
unsupported, and one shown more credits hallucinations against clauses the model
never read.

**Entity linking deviates from the HippoRAG paper**, which links each query
entity to its single nearest node (argmax). Linking to the top 3 was worth about
12 points of R@5 on this corpus; the paper's setting is reproducible with
`ENTITY_LINK_TOP_K=1`. Disclose the deviation rather than the improvement.

**`SYNONYM_THRESHOLD` is unset by default** and derived from the density target
instead. A fixed cosine cutoff is not transferable across corpus and encoder;
the run logs the weakest edge actually kept, and that is the number to report.

**Entity dedup is exact-match only** (case- and whitespace-normalised). An
earlier embedding-similarity merge at 0.90 was removed after it collapsed
legally and factually distinct terms — `13 may 1840`/`13 may 1846`,
`johann wilhelm bach`/`johann christoph bach`.

Fixed random seeds: synthetic QA generation and the statistical bootstrap both
use seed 42.

## 4. Data

Two corpora, built by different scripts and evaluated separately.

**(a) Three-tier corpus** (`dataset/corpus/full_corpus.json`, built by
`dataset/build_full_corpus.py`) — **959 chunks** across three levels of
jurisdiction:

  | Tier | Source | Chunks |
  |---|---|---|
  | regional | GDPR articles + recitals | 288 |
  | national | UK Data Protection Act 2018 | 285 |
  | national | Irish Data Protection Act 2018 | 243 |
  | national | German Federal Data Protection Act (BDSG) | 86 |
  | institutional | Trinity College Dublin | 20 |
  | institutional | Cambridge | 19 |
  | institutional | Göttingen | 14 |
  | institutional | Limerick | 4 |

**Citation graph** (`dataset/corpus/full_citations.json`, built by
`dataset/extract_citations.py`): 1,414 drafter-written edges — 1,206 `CITES`
(within one instrument) and 208 `IMPLEMENTS` (crossing a tier). These are regex
extractions, not LLM output, and each resolves to a chunk id that exists.

The provenance distinction is the point of the experiment: `RELATES` edges are
inferred by an LLM from co-mention, `CITES`/`IMPLEMENTS` were written by the
legislative drafters. Measured on the graph, one hop from five vector entry
points admits **99.1%** of chunks along `RELATES` and **7.1%** along citations.

**(b) GDPR-only corpus** (`dataset/corpus/pilot_corpus.json`, 345 chunks) — the
earlier single-tier set, retained because the first round of E1/E2 results was
measured on it.

- **Cross-tier evaluation set** (`dataset/crosstier_qa_full.json`): 78 queries
  from an earlier round, generated by a script since superseded by
  `dataset/generate_qa_v2.py` and removed (it remains in the git history). Each is built from one
  `IMPLEMENTS` edge, so `gold_chunk_ids` is exactly the national provision and
  the GDPR article it gives effect to — **fixed by construction, never chosen by
  the generator**. Edges with `resolution=spread` are excluded: they point at a
  paragraph other than the one cited, so the pair may not belong together.
  Distribution UK 38 / DE 26 / IE 14. No human review (documented limitation).

  Each query also carries a `scenario` block recording what the request
  involves — `purpose`, `data_category`, `data_subject`, `recipient`,
  `cross_border`, `legal_basis`. These feed the decision rubric (§Outputs) and
  the analysis.

  **The scenario is deliberately not written into `query_text`.** Spelling the
  data category and lawful basis out in the question would hand keywords to
  dense retrieval and narrow the gap the experiment measures. For the same
  reason **question length is a controlled condition**: the prompt caps it at
  60 words, because an unconstrained generator writes 99-word case studies
  against the pre-scenario set's median of 36, and the extra prose is retrieval
  surface. Both the generator and the runner report the median and warn.
- **Single-tier evaluation set** (`dataset/qa_dataset.json`): 160 queries from
  `dataset/generate_qa.py`, stratified single 56 / multi 56 / trap 32 /
  unanswerable 16.

Record each document's source URL and retrieval date in
`dataset/corpus/sources.md`. See `dataset/corpus/README.md` for the layout.

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
#    shared server, no Docker permission? -> SERVER_DEPLOYMENT.md

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

# 4. MOVING TO ANOTHER MACHINE: copy artifacts/ across FIRST (see below).
#    Not needed for the three-tier run, which extracts fresh into artifacts_full/.

# 5. THE THREE-TIER EXPERIMENT (corpus -> graph -> questions -> E2 -> E1)
#    One script, nine steps, resumable, with a check after each that can
#    otherwise produce plausible-looking wrong data. ~2.3M tokens overnight.
cd .. && ./run_full_experiment.sh --dry-run
./run_full_experiment.sh
#    -> results/<strategy>-<run-id>.json, results/full/logs/e2.log
python -m evaluation.rescore --run-id <id> --dataset ../dataset/crosstier_qa_full.json

#    --- or, for the earlier single-tier (GDPR-only) run, steps 6-8 by hand ---

# 6. Build the graph and indexes (offline; needs corpus in dataset/corpus/)
python -m ingestion.build_indexes         # -> Neo4j graph + vector indexes, artifacts/, cost report

# 7. Generate the evaluation dataset -- ONLY if you do not already have one
python ../dataset/generate_qa.py --n 160  # -> dataset/qa_dataset.json + verification_sample.json

# 8. Run the services -- ONE inference backend at a time, see note below
#    gateway
cd ../gateway-service && mvn spring-boot:run
#    then EITHER the consumer (EDA condition) ...
cd ../inference-service && python consumer_main.py
#    ... OR the sync API (both synchronous conditions)
uvicorn sync_api:app --port 8000

# 9. Reasoning-quality and retrieval experiments (E1, E2) for the single-tier set
python -m evaluation.run_eval --strategy zero_shot  --judge --run-id <id>
python -m evaluation.run_eval --strategy vector_rag --judge --run-id <id>
python -m evaluation.run_eval --strategy hybrid     --judge --run-id <id>
python -m evaluation.run_eval --strategy light_rag  --judge --run-id <id>
python -m evaluation.run_eval --strategy hippo_rag  --judge --run-id <id>
#    -> results/<strategy>-<run-id>.json  (+ .jsonl written as it goes)
#    interrupted? rerun the SAME command with --resume appended
#    retrieval only, the four strategies Recall is defined for:
python evaluation/ablation/compare_all_strategies.py --paired-ci \
  --strategies vector_rag,hybrid,light_rag,hippo_rag

# 10. Load experiment (E3)
jmeter -n -t ../loadtest/compliance_gateway.jmx \
       -JEDA_THREADS=10 -JRAMP=10 -JDURATION=120 -l results/eda-c10.jtl
#    one condition per run, selected by which thread count is non-zero:
#    -JEDA_THREADS / -JSYNC_THREADS / -JTHROTTLE_THREADS (all default to 0)
#    full 45-run matrix: ./loadtest/run_e3.sh  (see SERVER_DEPLOYMENT.md)
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

### Step 4 — moving to another machine

`inference-service/artifacts/` and `.env` are both gitignored, so a fresh
`git clone` has neither. The Neo4j graph is not in git either, so it must be
rebuilt on the new machine regardless.

**For the three-tier run, only `.env` needs copying.** It extracts all 959
chunks fresh under `deepseek-v3.2` into a separate `artifacts_full/`, so the
GDPR-only cache would not be reused even if present.

**For reproducing the single-tier GDPR run, copy `artifacts/` first.**
`artifacts/extraction_cache.json` is what makes that rebuild free and identical;
without it the build re-runs all 345 extractions, re-spending the ~430k-token,
~90-minute budget and producing a graph that differs from the measured one. The
cache is keyed by `provider:model:profile` and a mismatch is a hard error, so
the failure is loud — but the log should still report
**`345/345 chunks served from cache`** and make no API calls.

| Path | Size | Contents |
|---|---|---|
| `inference-service/artifacts/` | 2.2 MB | extraction cache, HippoRAG matrices, dedup report, chunk texts |
| `inference-service/.env` | 1 KB | API keys and model selection |

Then confirm the rebuild matches before committing hours of inference:

```bash
# graph: 345 Chunk, 1673 Entity, 3684 RELATES, 4394 MENTIONED_IN
# SHOW INDEXES -> chunk_vec, entity_vec, edge_vec all ONLINE
python evaluation/ablation/compare_all_strategies.py
#   expect R@10  vector_rag 0.537 / hybrid 0.537 / light_rag 0.567 / hippo_rag 0.179
```

A mismatch means the graph is not the one the reported retrieval numbers came
from, and the evaluation would not be comparable to them.

**Check all three vector indexes, not just `chunk_vec`.** Chunk embeddings are
written before the entity/relation pass, so a crash between the two leaves a
graph that looks populated — every chunk present and embedded, `chunk_vec`
ONLINE — while `hybrid`, `light_rag` and `hippo_rag` retrieve nothing at all,
because entity linking has no index to query. Three strategies at 0.000 reads
as a finding rather than a crash. `run_full_experiment.sh` step 5 asserts this;
a manual rebuild has to check it by hand.

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
| `results/<strategy>-<run-id>` comparison | compare_all_strategies | Recall@2/@5/@10 per strategy, see `inference-service/evaluation/ablation/README.md` |

### The two judge metrics

Both are produced by `--judge`, both scored against the **same** context the SLM
generated from, and they answer different questions. Report them side by side.

| | asks | abstention |
|---|---|---|
| `faithfulness` | are the claims grounded in the context? | `None` — excluded from the mean |
| `rubric_score` | was the request actually analysed? | `0.2` — scored |

`rubric_score` is the fraction of five items met: `identifies_data_category`,
`identifies_legal_basis`, `applies_national_law`, `addresses_transfer`,
`no_invented_clause`. `rubric_by_item` in the summary breaks it down, and
`applies_national_law` is the one the three-tier corpus exists to test — an
answer that reasons from the GDPR alone and ignores the state fails it while
scoring well on everything else.

The abstention asymmetry is deliberate. A claim-free abstention makes no claim
that could be unfaithful, so faithfulness excludes it; but it analysed nothing,
so the rubric scores it near zero (0.2 is `no_invented_clause`, which an
abstention satisfies trivially). **A strategy that abstains often will show high
faithfulness and low rubric — that pairing is the finding, not a contradiction**,
and it is what the earlier "retrieval raises the false-approval rate" result
needed in order to be explained rather than just reported.

The rubric returns `None` for datasets with no `scenario` block, so the
single-tier runs are unaffected and remain comparable to their own history.
