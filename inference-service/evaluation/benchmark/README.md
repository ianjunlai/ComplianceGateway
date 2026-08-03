# Supplementary experiment — retrieval on a public benchmark

The GDPR results show no graph configuration beating dense retrieval. That claim is only
defensible once the same implementations have been shown to reproduce a published
advantage somewhere else, so this runs the identical four strategies, retrieval only, on
**2WikiMultihopQA**.

2Wiki is the one benchmark where the published margin is decisive: HippoRAG reports
**R@5 89.1 against ColBERTv2's 68.2**. On MuSiQue the margin is 2.7, and on HotpotQA
HippoRAG *loses* (R@2 60.5 vs 64.7) — which is itself worth citing, because it makes
"graph retrieval below dense retrieval" a published, corpus-dependent outcome rather than
a symptom of a broken implementation.

**Success is the ordering, not the number.** Different embeddings (bge-large), extraction
model (qwen-plus) and entity linking mean 89.1 is not reproducible here and is not the
target. The claim under test is that the graph strategies rank above `vector_rag`. A
failure is equally informative and better learned now than at the viva.

## The prompts must be switched

`EXTRACTION_PROFILE=general` is required for every step below.
[`extraction.py`](../../ingestion/extraction.py) and [`ner.py`](../../pipeline/ner.py) each
carry two prompts, and the default (`legal`) names the entity kinds GDPR needs — actors,
data categories, safeguards, jurisdictions. Pointed at an encyclopaedia paragraph about a
film director it extracts almost nothing, so the graph strategies would lose for a reason
that has nothing to do with what is being tested. The substitution is a domain adapter and
belongs in the write-up; HippoRAG likewise uses a general-purpose OpenIE prompt.

`build_ner_cache.py` refuses to run under the wrong profile. The extraction cache key
includes the profile, so a legal cache can never be silently replayed for this corpus.

## Scale

The run is sized to a 1M-token extraction budget. Input is measurable in advance —
116 tokens of instructions plus a mean 95 tokens of passage, so 211 per call — and
questions bring about 7.4 candidate passages each:

| Questions | Passages | Estimated tokens (output at 1.5× input) |
|---|---|---|
| 100 | 759 | ~400k |
| **200** | **1,478** | **~780k** |
| 300 | 2,083 | ~1.10M — over budget |
| 1,000 (full) | 6,119 | ~3.2M |

**200 questions is the operating point.** 200 paired observations detect a 20-point
recall difference with room to spare, and the corpus is still four times the size of the
GDPR one. Samples nest — `--sample 20` is a prefix of `--sample 200` — so the rehearsal
below is a down payment, not throwaway spend.

Absolute recall will run higher than HippoRAG's published figures because the corpus is a
quarter of theirs. That is expected and does not matter: the comparison that carries the
argument is internal, graph strategies against `vector_rag` on the same corpus.

## Procedure

Two things must be separated from the GDPR experiment, or it gets destroyed: the graph
store and the artifacts directory. `build_indexes` opens with
`MATCH (n) DETACH DELETE n`, and the extraction cache, HippoRAG matrices and
`chunk_texts.json` are keyed by `chunk_id` alone.

```bash
# A second Neo4j, behind a compose profile so `docker compose up -d` never starts it
docker compose --profile benchmark up -d neo4j-benchmark

export EXTRACTION_PROFILE=general
export ARTIFACTS_DIR=$PWD/artifacts_2wiki
export NEO4J_URI=bolt://localhost:7688      # 7687 stays the GDPR graph

# 1. Fetch and convert — rehearse with --sample 20 (191 passages, ~90k tokens) first
python -m evaluation.benchmark.prepare_2wiki --sample 200

# 2. Build the graph
python -m ingestion.build_indexes \
    --corpus-json ../dataset/benchmark/2wiki_corpus.json --workers 8

# 3. Query seeds (local SLM, cached, resumable)
python -m evaluation.benchmark.build_ner_cache

# 4. The comparison
python evaluation/ablation/compare_all_strategies.py \
    --dataset ../dataset/benchmark/2wiki_qa.json \
    --ner-cache evaluation/benchmark/ner_seed_cache_2wiki.json \
    --paired-ci

# 5. The diagnostic — run it here, then again against the GDPR graph
python -m evaluation.benchmark.hop_coverage \
    --dataset ../dataset/benchmark/2wiki_qa.json \
    --ner-cache evaluation/benchmark/ner_seed_cache_2wiki.json
```

Returning to the GDPR experiment is unsetting those three variables. Nothing about it was
touched: different container, different volume, different artifacts directory, and the
extraction cache key carries the profile so the two caches can never be confused.

## What the corpus looks like

| | |
|---|---|
| Passages | 6,119 available, unique titles, median 44 words |
| Questions | 1,000 — compositional 413, comparison 244, bridge_comparison 235, inference 108 |
| Gold per question | 2, except bridge_comparison which has 4 |
| Source | `OSU-NLP-Group/HippoRAG`, branch `legacy`, `data/2wikimultihopqa*.json` |

Gold passages are named by title in `supporting_facts`; titles are unique across the
corpus, so the title is the join key and chunk ids are positional (`2wiki-00042`) so that
no slugging rule can merge two distinct titles. `prepare_2wiki.py` refuses to write a
dataset if any gold title fails to resolve — that failure would otherwise score zero recall
for every strategy at once and read as a finding.

## Sanity checks before trusting a number

- `vector_rag` R@5 should land somewhere near ColBERTv2's 68.2 on this corpus. Near zero
  means the id mapping broke, not that dense retrieval failed.
- The GDPR numbers must still reproduce (`0.537 / 0.537 / 0.196 / 0.171`) with the profile
  back to `legal` — nothing here is allowed to disturb the existing results.
