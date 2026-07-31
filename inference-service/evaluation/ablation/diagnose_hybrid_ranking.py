# -*- coding: utf-8 -*-
"""Why does Hybrid GraphRAG miss a gold chunk its seeds explicitly name?

q-001's NER produced the seed "GDPR Article 9(2)" and its gold chunk is
gdpr-art-9-2, yet Recall@10 was 0. This reproduces the strategy's own traversal
and reports how wide the 2-hop neighbourhood is and where the gold chunk lands
in the ranking, separating "the graph can't reach it" from "the ranking buries
it".
"""

import sys
from pathlib import Path

_SERVICE = Path(__file__).resolve().parents[2]      # inference-service/
_REPO = _SERVICE.parent
sys.path.insert(0, str(_SERVICE))

import config
from pipeline.entity_linking import link_entities
from pipeline.graph import get_driver

SEEDS = ["University", "Student", "Disability Accommodation", "Health Data",
         "GDPR Article 9(2)", "Staff Bound by Professional Secrecy Obligations"]
GOLD = "gdpr-art-9-2"

# The strategy's query, minus the LIMIT, so the whole ranked list is visible.
QUERY = """
UNWIND $seed_ids AS seed_id
MATCH (seed:Entity {node_id: seed_id})
CALL (seed) {
    MATCH (seed)-[:RELATES*1..%(hops)d]-(nbr:Entity)
    RETURN collect(DISTINCT nbr) AS nbrs
}
WITH seed, nbrs
UNWIND ([seed] + nbrs) AS ent
MATCH (ent)-[:MENTIONED_IN]->(c:Chunk)
WITH c, collect(DISTINCT ent.name) AS entities
RETURN c.chunk_id AS chunk_id, size(entities) AS n_ent
ORDER BY n_ent DESC
""" % {"hops": config.GRAPH_HOPS}

seed_ids = link_entities(SEEDS)

with get_driver().session() as session:
    rows = [(r["chunk_id"], r["n_ent"]) for r in session.run(QUERY, seed_ids=seed_ids)]
    total_chunks = session.run("MATCH (c:Chunk) RETURN count(c) AS n").single()["n"]
    total_ents = session.run("MATCH (e:Entity) RETURN count(e) AS n").single()["n"]

print(f"seed entities linked to graph : {len(seed_ids)} nodes")
print(f"chunks reachable within {config.GRAPH_HOPS} hops : {len(rows)} of {total_chunks} "
      f"({len(rows) / total_chunks:.0%} of the corpus)")
print(f"entities in graph             : {total_ents}")

ranks = {cid: i + 1 for i, (cid, _) in enumerate(rows)}
print(f"\ngold chunk {GOLD!r} rank: {ranks.get(GOLD, 'NOT REACHED')} of {len(rows)}"
      f"   (retrieval keeps the top {config.RETRIEVAL_K})")

print("\ntop 10 by the strategy's ranking (chunk, entities mentioning it):")
for cid, n in rows[:10]:
    print(f"   {cid:22} {n}")
if GOLD in ranks:
    i = ranks[GOLD] - 1
    print(f"   ... gold sits at #{i + 1} with n_ent={rows[i][1]}")
