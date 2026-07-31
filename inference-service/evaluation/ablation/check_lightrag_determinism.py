# -*- coding: utf-8 -*-
"""Does light_rag return the same clauses when asked the same question twice?

Its neighbour expansion applies a LIMIT that discards most of the result set.
Before the fix that LIMIT had no ORDER BY, so the database was free to return a
different subset on an identical re-run -- which would silently make every
reported metric a sample of one arbitrary draw rather than a property of the
method. This runs each query twice in fresh sessions and diffs the output.
"""

import sys
from pathlib import Path

_SERVICE = Path(__file__).resolve().parents[2]      # inference-service/
_REPO = _SERVICE.parent
sys.path.insert(0, str(_SERVICE))
import json

from pathlib import Path

import config
from pipeline.strategies import build_strategy

HERE = Path(__file__).parent
cache = json.loads((HERE / "ner_seed_cache.json").read_text(encoding="utf-8"))
items = {q["query_id"]: q for q in json.loads(
    (_REPO / "dataset" / "qa_dataset.json").read_text(encoding="utf-8"))}

strategy = build_strategy("light_rag")
mismatches = []
for qid, seeds in cache.items():
    q = items[qid]["query_text"]
    a = strategy.retrieve(q, seeds, config.RETRIEVAL_K).chunk_ids
    b = strategy.retrieve(q, seeds, config.RETRIEVAL_K).chunk_ids
    if a != b:
        mismatches.append((qid, a, b))

print(f"{len(cache)} queries retrieved twice each")
if mismatches:
    print(f"NON-DETERMINISTIC on {len(mismatches)} queries:")
    for qid, a, b in mismatches[:3]:
        print(f"  {qid}\n    run 1: {a}\n    run 2: {b}")
else:
    print("identical results on every query -- retrieval is reproducible")
