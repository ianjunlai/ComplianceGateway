# -*- coding: utf-8 -*-
"""Does node specificity survive PPR propagation?

HippoRAG divides each seed's starting mass by the number of passages the entity
occurs in, precisely so ubiquitous entities don't dominate. But specificity is
applied only to the personalization vector; propagation then flows mass along
edges, and this corpus has entities with degree >= 50. This checks where the
mass actually ends up: if the highest-scoring nodes after PPR are the same
corpus-wide hubs, the weighting is being undone downstream and the
implementation being faithful to the paper is not enough to save it.
"""

import sys
from pathlib import Path

_SERVICE = Path(__file__).resolve().parents[2]      # inference-service/
_REPO = _SERVICE.parent
sys.path.insert(0, str(_SERVICE))
import json

from pathlib import Path

import config
from pipeline.strategies.hippo_rag import HippoRagStrategy

HERE = Path(__file__).parent
cache = json.loads((HERE / "ner_seed_cache.json").read_text(encoding="utf-8"))
items = json.loads((_REPO / "dataset" / "qa_dataset.json")
                   .read_text(encoding="utf-8"))
by_id = {q["query_id"]: q for q in items}

s = HippoRagStrategy()
inv_node = {v: k for k, v in s.node_index.items()}

# Corpus-wide hubs, by the same measure node specificity uses.
occ = sorted(((len(v), k) for k, v in s.node_chunks.items()), reverse=True)
hub_ids = {nid for _, nid in occ[:15]}
print("corpus hubs (top 15 by passage count):")
print("   " + ", ".join(s.node_names.get(nid, nid) for _, nid in occ[:5]) + ", ...\n")

from pipeline.entity_linking import link_entities

checked = 0
for qid, seeds in list(cache.items())[:5]:
    seed_ids = [n for n in link_entities(seeds) if n in s.node_index]
    if not seed_ids:
        print(f"{qid}: no seed linked -> empty context")
        continue
    scores = s._personalized_pagerank(seed_ids)
    order = scores.argsort()[::-1][:10]
    top_names = [s.node_names.get(inv_node[int(i)], inv_node[int(i)]) for i in order]
    n_hub = sum(1 for i in order if inv_node[int(i)] in hub_ids)
    seed_specific = sum(1 for n in seed_ids if n not in hub_ids)
    print(f"{qid}: {len(seed_ids)} seeds ({seed_specific} non-hub) "
          f"-> {n_hub}/10 of top PPR nodes are corpus hubs")
    print(f"   top nodes: {', '.join(top_names[:6])}")
    print(f"   gold      : {by_id[qid]['gold_chunk_ids']}")
    checked += 1

print(f"\nchecked {checked} queries")
