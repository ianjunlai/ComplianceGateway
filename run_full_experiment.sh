#!/usr/bin/env bash
# The whole three-tier experiment, start to finish, unattended.
#
# Nine steps, each writing a file the next one reads. A step whose output
# already exists is skipped, so an interrupted run resumes rather than
# restarting -- which matters because step 3 is ~1M tokens of extraction and
# step 9 is several hours.
#
# Every step that can produce plausible-looking wrong data is followed by a
# check that fails loudly instead. That is not defensive habit: today an empty
# but ONLINE vector index, a shadowed variable, and a silently truncated
# extraction each produced results that looked like findings.
#
#   ./run_full_experiment.sh --dry-run     # print the plan and the checks
#   ./run_full_experiment.sh               # run it
#   ./run_full_experiment.sh --from 7      # resume from a given step
set -uo pipefail

REPO="${REPO:-$HOME/ComplianceGateway}"
PYTHON="${PYTHON:-python}"
SERVICE="$REPO/inference-service"
RUN_ID="${RUN_ID:-full$(date +%m%d)}"

# The tarball Neo4j on its default port -- step 3 wipes it and rebuilds with
# the three-tier corpus, which is fine because the three-tier corpus contains
# the GDPR one. Nothing is lost that costs money to recover: the GDPR
# extraction cache lives in a separate ARTIFACTS_DIR, so that graph can be
# rebuilt later for the price of embeddings and writes, with no API calls.
# To keep both graphs live at once, unpack a second tarball with
# server.bolt.listen_address=:7689 and set NEO4J_URI accordingly.
export NEO4J_URI="${NEO4J_URI:-bolt://localhost:7687}"
export ARTIFACTS_DIR="${ARTIFACTS_DIR:-$SERVICE/artifacts_full}"
export EXTRACTION_PROFILE=legal
export ENTITY_LINK_TOP_K="${ENTITY_LINK_TOP_K:-3}"
export VECTOR_EXPAND_ENTRY_K="${VECTOR_EXPAND_ENTRY_K:-5}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
WORKERS="${WORKERS:-8}"
N_QUESTIONS="${N_QUESTIONS:-78}"
# The judge is network-bound and ~25x slower than the GPU pipeline it scores,
# so this, not the GPU, sets the wall clock of step 9. Measured on this corpus:
# faithfulness ~13 s and rubric ~25 s per query against glm-5.2, against 1.5 s
# for the pipeline. Serial, step 9 alone is about seven hours.
JUDGE_WORKERS="${JUDGE_WORKERS:-8}"

CORPUS="$REPO/dataset/corpus/full_corpus.json"
CITATIONS="$REPO/dataset/corpus/full_citations.json"
QA="$REPO/dataset/crosstier_qa_full.json"
NER="$SERVICE/evaluation/benchmark/ner_seed_cache_full.json"
LOGS="$REPO/results/full/logs"

# E2 is free, so every variant is measured. E1 costs a judge call per query, so
# it covers the ones that answer a research question: the no-retrieval baseline,
# the dense baseline, the three published paradigms, and the two citation
# variants. vec_relates scores identically to vector_rag by construction (its
# expansion admits 99% of the corpus) and vec_intra was the weakest; both are
# reported from E2 alone.
E2_STRATEGIES="vector_rag,hybrid,light_rag,hippo_rag,vec_relates,vec_intra,vec_implements,vec_cites,vec_both,vec_juris"
E1_STRATEGIES="zero_shot vector_rag hybrid light_rag hippo_rag vec_implements vec_cites vec_juris"

DRY=0; FROM=1
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY=1; shift ;;
    --from) FROM="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

mkdir -p "$LOGS"
log() { printf '\n\033[1m%s  %s\033[0m\n' "$(date +%H:%M:%S)" "$*"; }
die() { printf '\n%s  FAILED: %s\n' "$(date +%H:%M:%S)" "$*" >&2; exit 1; }
step() {  # step <n> <description>
  CURRENT=$1
  [[ $CURRENT -lt $FROM ]] && return 1
  log "[$CURRENT/9] $2"
  [[ $DRY -eq 1 ]] && return 1
  return 0
}

cd "$SERVICE" || die "no $SERVICE"

# ---------------------------------------------------------------- 0. preflight
log "preflight"
if [[ $DRY -eq 0 ]]; then
  "$PYTHON" - <<'PY' || die "environment is not ready"
import os, sys
sys.path.insert(0, ".")
import config
missing = []
if not os.getenv("ALIBABA_API_KEY"): missing.append("ALIBABA_API_KEY")
print(f"  extraction : {config.EXTRACTION_PROVIDER}/{config.EXTRACTION_MODEL}")
print(f"  QA gen     : {config.QA_GENERATION_PROVIDER}/{config.QA_GENERATION_MODEL}")
print(f"  judge      : {config.JUDGE_PROVIDER}/{config.JUDGE_MODEL}")
print(f"  SLM        : {config.SLM_MODEL}")
print(f"  artifacts  : {config.ARTIFACTS_DIR}")
if missing:
    sys.exit(f"  missing env: {missing}")
PY
  # 2>/dev/null: the driver raises a four-deep chained traceback that buries the
  # one line worth reading.
  "$PYTHON" -c "
import sys; sys.path.insert(0,'.')
from pipeline.graph import get_driver
with get_driver().session() as s: s.run('RETURN 1').single()
print('  neo4j      : reachable at ' + __import__('config').NEO4J_URI)" 2>/dev/null \
    || die "Neo4j unreachable at $NEO4J_URI
         start it with:  ~/neo4j-community-5.24.0/bin/neo4j start
         check it with:  ~/neo4j-community-5.24.0/bin/neo4j status"
  # Step 3 opens with MATCH (n) DETACH DELETE n. Record what is being replaced
  # so the log says so, rather than leaving it to be discovered afterwards.
  "$PYTHON" -c "
import sys; sys.path.insert(0,'.')
from pipeline.graph import get_driver
with get_driver().session() as s:
    n = s.run('MATCH (c:Chunk) RETURN count(c) AS n').single()['n']
print(f'  graph      : {n} chunks present, to be REPLACED by the 959-chunk corpus'
      if n else '  graph      : empty')" 2>/dev/null
  curl -sf --max-time 5 "${OLLAMA_HOST:-http://localhost:11434}/api/tags" >/dev/null \
    || die "Ollama unreachable — NER seeds and E1 both need it"
  echo "  ollama     : reachable"
  # One real call per configured model. A model name the provider does not
  # serve is otherwise discovered by step 3 after it has already spent an hour,
  # or by step 9 in the middle of the night.
  "$PYTHON" - <<'PY' || die "a configured model is not usable"
import sys; sys.path.insert(0, ".")
import config
from common.llm_clients import complete_json
bad = []
for label, prov, model in [
        ("extraction", config.EXTRACTION_PROVIDER, config.EXTRACTION_MODEL),
        ("qa_gen", config.QA_GENERATION_PROVIDER, config.QA_GENERATION_MODEL),
        ("judge", config.JUDGE_PROVIDER, config.JUDGE_MODEL)]:
    try:
        # 512, not a token or two: these are reasoning models and spend most of
        # a small budget before emitting anything, so a tight limit fails a
        # model that is in fact fine.
        _, usage = complete_json(prov, model, 'Return JSON {"ok": true}',
                                 max_tokens=512, max_attempts=1)
        print(f"  {label:<11}{prov}/{usage.get('served_model', model)} "
              f"({usage['completion_tokens']} completion tokens)")
    except Exception as exc:
        print(f"  {label:<11}{prov}/{model} FAILED: {type(exc).__name__}: {exc}")
        bad.append(label)
if bad:
    sys.exit(f"  unusable: {bad}")
PY
fi

# ------------------------------------------------------------- 1. build corpus
if step 1 "assemble the 959-chunk three-tier corpus"; then
  [[ -f "$CORPUS" ]] && echo "  exists, skipping" || \
    "$PYTHON" "$REPO/dataset/build_full_corpus.py" || die "corpus build"
fi

# ---------------------------------------------------------- 2. citation edges
if step 2 "extract citation edges (regex, no API)"; then
  # Always regenerated, never skipped: it is free and deterministic, and a
  # stale copy is a real hazard. The committed file was produced before the
  # FOREIGN-exclusion fix and contains no German cross-tier edges at all.
  "$PYTHON" "$REPO/dataset/extract_citations.py" \
    --corpus "$CORPUS" --out "$CITATIONS" || die "citation extraction"
  "$PYTHON" - "$CITATIONS" <<'PY' || die "citation edges are not usable"
import json, sys
from collections import Counter
edges = json.load(open(sys.argv[1], encoding="utf-8"))
imp = [e for e in edges if e["type"] == "IMPLEMENTS"]
usable = [e for e in imp if e["resolution"] in ("exact", "whole")]
juris = Counter(s.split("-")[0] for s in {e["source"] for e in usable})
print(f"  {len(edges)} edges, {len(imp)} cross-tier, "
      f"{len({e['source'] for e in usable})} usable distinct sources: {dict(juris)}")
# Germany cites the GDPR as "Regulation (EU) 2016/679" where the UK and Irish
# Acts name it in prose. A pattern that handles the prose forms leaves the
# German tier with zero edges and the other two looking correct, so the
# per-jurisdiction floor is the check that catches it.
for j in ("ie", "uk", "de"):
    assert juris.get(j, 0) >= 10, f"only {juris.get(j, 0)} usable sources for {j}"
PY
fi

# ------------------------------------------------------ 3. extraction + graph
if step 3 "extract entities/relations and build the graph (~1M tokens)"; then
  # The filter is for the console only; the full log goes to disk. Read
  # PIPESTATUS rather than $? -- under pipefail a grep that matches nothing
  # fails the pipeline even though the build succeeded.
  "$PYTHON" -m ingestion.build_indexes \
      --corpus-json "$CORPUS" --workers "$WORKERS" \
      2>&1 | tee "$LOGS/build.log" | grep -viE "HTTP Request|Batches:"
  [[ ${PIPESTATUS[0]} -eq 0 ]] || die "extraction/graph build (see $LOGS/build.log)"
  # build_indexes asserts every chunk carries an embedding before it exits, so
  # reaching here with a zero status means the graph is populated, not merely
  # created.
  grep -q "Vector indexes populated" "$LOGS/build.log" || die "graph build did not complete"
fi

# ------------------------------------------------------- 4. load citation edges
if step 4 "load CITES / IMPLEMENTS edges"; then
  "$PYTHON" -m ingestion.load_implements --remove >/dev/null 2>&1
  "$PYTHON" -m ingestion.load_implements --edges "$CITATIONS" \
      2>&1 | grep -v notification || die "loading citation edges"
fi

# ------------------------------------------------------------ 5. verify graph
if step 5 "verify the graph before anything expensive depends on it"; then
  "$PYTHON" - <<'PY' || die "graph verification"
import sys; sys.path.insert(0, ".")
from pathlib import Path
import config
from pipeline.graph import get_driver
with get_driver().session() as s:
    q = lambda c: s.run(c).single()["n"]
    chunks = q("MATCH (c:Chunk) RETURN count(c) AS n")
    embedded = q("MATCH (c:Chunk) WHERE c.embedding IS NOT NULL RETURN count(c) AS n")
    entities = q("MATCH (e:Entity) RETURN count(e) AS n")
    ent_emb = q("MATCH (e:Entity) WHERE e.embedding IS NOT NULL RETURN count(e) AS n")
    rel_emb = q("MATCH ()-[r:RELATES]->() WHERE r.embedding IS NOT NULL RETURN count(r) AS n")
    rows = {r["name"]: r["state"] for r in s.run(
        "SHOW INDEXES YIELD name, type, state WHERE type='VECTOR' RETURN name, state")}
    for label, cy in [("Entity", "MATCH (e:Entity) RETURN count(e) AS n"),
                      ("RELATES", "MATCH ()-[r:RELATES]->() RETURN count(r) AS n"),
                      ("SYNONYM", "MATCH ()-[r:SYNONYM]->() RETURN count(r) AS n"),
                      ("CITES", "MATCH ()-[r:CITES]->() RETURN count(r) AS n"),
                      ("IMPLEMENTS", "MATCH ()-[r:IMPLEMENTS]->() RETURN count(r) AS n")]:
        print(f"    {label:<12}{q(cy):>7}")
    print(f"    {'Chunk':<12}{chunks:>7}  ({embedded} embedded)")
    tiers = {(r["t"], r["j"]): r["n"] for r in s.run(
        "MATCH (c:Chunk) RETURN c.tier AS t, c.jurisdiction AS j, count(*) AS n "
        "ORDER BY t, j")}
    print(f"    tiers: {tiers}")
    # Both fields are dropped silently if the corpus JSON lacks them, and a
    # jurisdiction-filtered strategy would then match nothing.
    assert all(t and j for t, j in tiers), f"chunks with no tier/jurisdiction: {tiers}"
    print(f"    {'entity emb':<12}{ent_emb:>7}  of {entities}")
    print(f"    {'RELATES emb':<12}{rel_emb:>7}")
    print(f"    vector indexes: {rows}")
    # An index can be ONLINE and empty: schema survives DETACH DELETE. Retrieval
    # against an empty index returns nothing without raising, and every strategy
    # scores zero at once, which reads as a finding.
    assert embedded == chunks, f"only {embedded}/{chunks} chunks embedded"
    assert q("MATCH ()-[r:IMPLEMENTS]->() RETURN count(r) AS n") > 0, "no cross-tier edges"
    # Chunk embeddings are written first and the entity/relation pass comes
    # after. A crash between the two -- an OOM or a segfault in the embedding
    # model, which is exactly what happened on the smoke run -- leaves a graph
    # that looks populated and passes a chunk-only check, while hybrid,
    # light_rag and hippo_rag silently retrieve nothing: entity linking has no
    # index to query. Three strategies at 0.000 reads as a finding, not a crash.
    for want in ("chunk_vec", "entity_vec", "edge_vec"):
        assert rows.get(want) == "ONLINE", f"vector index {want} is {rows.get(want)!r}, not ONLINE"
    assert ent_emb == entities, f"only {ent_emb}/{entities} entities embedded"
    assert rel_emb > 0, "no RELATES edge carries an embedding; light_rag would return nothing"
# hippo_rag loads these in its constructor and run_eval reads chunk_texts.json,
# so a missing file is at least a loud failure -- but it would surface partway
# through E2, after the other strategies had already run.
art = Path(config.ARTIFACTS_DIR)
for f in ("hippo_adjacency.npz", "hippo_passage_matrix.npz", "hippo_chunk_index.json",
          "hippo_nodes.json", "hippo_node_chunks.json", "chunk_texts.json"):
    assert (art / f).exists(), f"{f} missing from {art} — the build did not finish"
print("  graph OK")
PY
fi

# --------------------------------------------------------------- 6. generate QA
if step 6 "generate $N_QUESTIONS cross-tier questions (~180k tokens)"; then
  if [[ -f "$QA" ]]; then echo "  exists, skipping"; else
    # --corpus/--edges are not optional here: the defaults point at the
    # 414-chunk pilot, whose ids overlap the full corpus, so the wrong dataset
    # would evaluate cleanly against the wrong gold.
    "$PYTHON" "$REPO/dataset/generate_crosstier_qa.py" \
        --n "$N_QUESTIONS" --corpus "$CORPUS" --edges "$CITATIONS" --out "$QA" \
        2>&1 | tail -8 || die "QA generation"
  fi
  "$PYTHON" - "$QA" "$CORPUS" <<'PY' || die "QA validation"
import json, re, sys
from collections import Counter
qa = json.load(open(sys.argv[1], encoding="utf-8"))
ids = {c["chunk_id"] for c in json.load(open(sys.argv[2], encoding="utf-8"))}
# A gold id that names no chunk scores zero recall for every strategy at once,
# which is indistinguishable from a real result until someone checks.
missing = {g for q in qa for g in q["gold_chunk_ids"] if g not in ids}
assert not missing, f"gold ids absent from the corpus: {sorted(missing)[:5]}"
# Gold is fixed by construction here, but a question that names its own answer
# hands it to dense retrieval and makes the graph variants look worse for a
# reason unrelated to graph structure.
leaks = [q for q in qa if re.search(r"\b(Article|Section)\s+\d", q["query_text"])]
if leaks:
    kept = [q for q in qa if q not in leaks]
    json.dump(kept, open(sys.argv[1], "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"  dropped {len(leaks)} questions naming an article/section number")
    qa = kept
assert all(len(q["gold_chunk_ids"]) == 2 for q in qa), "a question lost a gold chunk"
by_juris = Counter(q["jurisdiction"] for q in qa)
assert len(by_juris) == 3, f"a jurisdiction is missing from the question set: {dict(by_juris)}"
# The rubric silently returns None without this block, so E1 would finish with
# no rubric column and nothing to say why.
assert all(q.get("scenario") for q in qa), "questions carry no scenario block"
# Question length is an experimental condition: the pre-scenario set ran a
# median of 36 words, and a much longer question hands dense retrieval keyword
# surface, narrowing the gap being measured.
words = sorted(len(q["query_text"].split()) for q in qa)
median = words[len(words) // 2]
assert median <= 70, f"questions average {median} words; not comparable to the earlier set"
print(f"  {len(qa)} questions, two gold chunks each, all ids resolve; {dict(by_juris)}")
print(f"  median {median} words, scenario block on all")
PY
fi

# ------------------------------------------------------------- 7. NER seeds
if step 7 "extract query seeds (local SLM, no API)"; then
  [[ -f "$NER" ]] && echo "  exists, skipping" || \
    "$PYTHON" -m evaluation.benchmark.build_ner_cache_generic \
      --dataset "$QA" --cache "$NER" 2>&1 | tail -4 || die "NER seeds"
fi

# --------------------------------------------------------------------- 8. E2
if step 8 "E2 — retrieval, all ten variants (no API cost)"; then
  "$PYTHON" evaluation/ablation/compare_all_strategies.py \
      --dataset "$QA" --ner-cache "$NER" \
      --strategies "$E2_STRATEGIES" --paired-ci \
      2>&1 | tee "$LOGS/e2.log" | grep -viE "Batches:|INFO|Warning|HF Hub"
  [[ ${PIPESTATUS[0]} -eq 0 ]] || die "E2 (see $LOGS/e2.log)"
fi

# --------------------------------------------------------------------- 9. E1
# The judge is fed GENERATION_CONTEXT_K=5 chunks -- exactly what the SLM saw,
# not the RETRIEVAL_K=10 ranked list -- so ~1.2k context tokens at the median.
if step 9 "E1 — decisions and faithfulness, 8 strategies (~1.25M tokens, ~1.3 h)"; then
  for s in $E1_STRATEGIES; do
    out="$REPO/results/${s}-${RUN_ID}.json"
    if [[ -f "$out" ]]; then
      echo "  $s already done, skipping"
      continue
    fi
    log "  E1: $s"
    # --resume is always passed: a strategy interrupted mid-pass picks its own
    # .jsonl back up, and a query that failed is retried rather than skipped.
    "$PYTHON" -m evaluation.run_eval --strategy "$s" --judge \
        --dataset "$QA" --run-id "$RUN_ID" --resume \
        --judge-workers "$JUDGE_WORKERS" \
        2>&1 | tee "$LOGS/e1-$s.log" | grep -viE "HTTP Request|Batches:" | tail -25
    # A strategy that dies must not take the other six with it -- the run is
    # overnight and a partial result set is worth far more than none. Its
    # .jsonl is kept, so `--from 9` resumes that strategy where it stopped.
    [[ ${PIPESTATUS[0]} -eq 0 ]] || FAILED="${FAILED:-} $s"
  done
  [[ -n "${FAILED:-}" ]] && echo "  WARNING: incomplete:${FAILED} — rerun with --from 9"
fi

log "done. results in $REPO/results/, logs in $LOGS"
echo "  E2 table : $LOGS/e2.log"
echo "  E1 files : results/*-$RUN_ID.json"
echo
echo "  Re-score E1 with the corrected metrics (excludes unsound UNKNOWN labels,"
echo "  splits answerable from unanswerable, adds McNemar):"
echo "     python -m evaluation.rescore --run-id $RUN_ID --dataset $QA"
