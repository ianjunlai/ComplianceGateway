# Retrieval ablation experiments

Provenance for every figure in [`docs/retrieval_ablation.md`](../../../docs/retrieval_ablation.md).
Each script is standalone and prints the table it supports. Run from the
`inference-service/` directory:

```bash
python evaluation/ablation/compare_all_strategies.py
```

Requires the GDPR graph to be built and Neo4j running. Only the NER step needs the local
SLM, and its output is cached in `ner_seed_cache.json` (committed), so every script here
runs without Ollama and without spending API credit.

| Script | Supports |
|---|---|
| `compare_all_strategies.py` | §1 — the four strategies, Recall@5/@10 by hop type |
| `check_ann_topk_stability.py` | §1 — approximate-index correction (52% of `k=10` queries differ from an exact ranking) |
| `compare_hybrid_variants.py` | §2 configs 1–3 — hub-count ranking, query-vector ranking, vector-seeded narrowing |
| `compare_graph_expansion.py` | §2 config 4 — vector entry points expanded through the graph |
| `sweep_hub_filter.py` | §2 config 5 — excluding hub entities from traversal |
| `test_specificity_proximity.py` | §2 config 6 — specificity-weighted proximity, β and anchor sweep, with an unweighted control |
| `test_proximity_significance.py` | §2 — paired bootstrap CIs for config 6 |
| `inspect_graph_hubs.py` | §3 — entity mention concentration |
| `audit_graph_schema.py` | §3 — entity and relation type vocabulary |
| `audit_dataset_bias.py` | §4.1 — whether the synthetic questions hand dense retrieval the answer |
| `audit_gold_connectivity.py` | §4.2 — whether the graph connects the clause pairs the questions need |
| `audit_dedup_damage.py` | §4.3 — cross-type merges and their content |
| `diagnose_hybrid_ranking.py` | §5.1 — the traversal reaching 99% of the corpus, gold at rank 100 |
| `check_lightrag_determinism.py` | §5.2 — identical results across repeated runs |
| `audit_hippo_ppr.py` | §5.3 — whether node specificity survives PPR propagation |
| `audit_hippo_projection.py` | §5.3 — published projection against two normalisations |

## Regenerating the NER cache

Delete `ner_seed_cache.json` and run `compare_hybrid_variants.py`; it re-extracts seeds
for its 40 queries (~8s each on a local 1B model) and rewrites the cache. Every other
script reads it. Regenerating changes the seeds and therefore the graph-strategy numbers
slightly, so keep the committed cache unless the point is to test seed sensitivity.
