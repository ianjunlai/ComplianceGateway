# Retrieval comparison

`compare_all_strategies.py` runs the retrieval half of the evaluation: every
strategy over one question set, reporting Recall@2/@5/@10 broken down by hop
type, and the paired difference of each strategy against `vector_rag`. It is
step 6 of `run_full_experiment.sh` and can also be run on its own from the
`inference-service/` directory:

```bash
python -m evaluation.ablation.compare_all_strategies \
    --dataset ../dataset/qa_v2.json \
    --ner-cache evaluation/benchmark/ner_seed_cache_v2.json
```

Requires the graph to be built and Neo4j running. Only the NER step needs the
local SLM, and its output is cached, so a run with a cache present needs no
Ollama and spends no API credit.

## NER caches

Query entity extraction is cached per question set, because re-extracting
changes the seeds and therefore the graph-strategy numbers. The caches live in
`evaluation/benchmark/` and are committed:

| Cache | Question set |
|---|---|
| `ner_seed_cache_v2.json` | `dataset/qa_v2.json`, the 96-question set used in the thesis |
| `ner_seed_cache_full.json` | the earlier three-tier set |
| `ner_seed_cache_crosstier.json` | the cross-tier pilot |
| `ner_seed_cache_2wiki.json` | the 2WikiMultihopQA benchmark subset |

`ner_seed_cache.json` in this directory belongs to the first single-tier pilot.
To build a cache for a new question set, use
`evaluation/benchmark/build_ner_cache.py`.

An earlier round of this project kept a set of one-off diagnostic scripts here
for auditing the graph, the dedup step and the HippoRAG projection. They were
investigation tools rather than part of the evaluation and have been removed;
they remain in the git history.
