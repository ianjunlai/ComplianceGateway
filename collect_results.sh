#!/usr/bin/env bash
# Bundle everything from a finished run that cannot be regenerated for free.
#
# Run on the SERVER after run_full_experiment.sh finishes. Produces one tarball
# to scp home. The split is by cost, not by size -- the whole bundle is a few
# megabytes; what matters is that two of the files inside represent about 1.2M
# tokens that would otherwise be spent again.
#
#   ./collect_results.sh                  # -> compliance-gateway-<runid>-<date>.tar.gz
#   ./collect_results.sh --check          # list what would go in, take nothing
set -uo pipefail

# Paths below are written relative to $REPO on purpose, so the tarball unpacks
# straight over a clean clone; an absolute ARTIFACTS_DIR would bake this
# machine's layout into it.
REPO="${REPO:-$(cd "$(dirname "$0")" && pwd)}"
RUN_ID="${RUN_ID:-$(ls "$REPO"/results/*-full*.json 2>/dev/null | head -1 |
                    sed 's/.*-\(full[0-9]*\)\.json/\1/')}"
RUN_ID="${RUN_ID:-unknown}"
OUT="$REPO/compliance-gateway-${RUN_ID}-$(date +%Y%m%d).tar.gz"

CHECK=0
[[ "${1:-}" == "--check" ]] && CHECK=1

# Paths are relative to $REPO so the tarball unpacks straight over a clean clone.
#
# MUST KEEP -- cost real money or real GPU hours, and are not deterministic:
#   extraction_cache.json   ~983k tokens of entity/relation extraction
#   crosstier_qa_full.json  ~180k tokens, and regenerating gives DIFFERENT
#                           questions, so old results stop being comparable
#   results/                the experiment itself
#   ner_seed_cache_full.json  free in tokens but needs Ollama and a GPU pass
#
# KEEP -- tiny, and they are the provenance the write-up cites:
#   indexing_cost_report.json, extraction_token_usage_note.md, dedup_report.json
#
# DELIBERATELY OMITTED -- rebuilt from the extraction cache with zero API calls:
#   chunk_texts.json, hippo_*.npz, hippo_*.json, the Neo4j store itself,
#   full_corpus.json and full_citations.json (deterministic, and tracked in git)
WANT=(
  "inference-service/artifacts_full/extraction_cache.json"
  "inference-service/artifacts_full/indexing_cost_report.json"
  "inference-service/artifacts_full/dedup_report.json"
  "inference-service/artifacts_full/extraction_token_usage_note.md"
  "inference-service/evaluation/benchmark/ner_seed_cache_full.json"
  "dataset/crosstier_qa_full.json"
  "dataset/corpus/full_corpus.json"
  "dataset/corpus/full_citations.json"
  "results"
)

cd "$REPO" || exit 1
present=(); missing=()
for p in "${WANT[@]}"; do
  if [[ -e "$p" ]]; then present+=("$p"); else missing+=("$p"); fi
done

printf '\n%-62s %10s\n' "FILE" "SIZE"
printf -- '-%.0s' {1..74}; echo
for p in "${present[@]}"; do
  printf '%-62s %10s\n' "$p" "$(du -sh "$p" 2>/dev/null | cut -f1)"
done
for p in "${missing[@]}"; do printf '%-62s %10s\n' "$p" "MISSING"; done

# The cache is only reusable if the next machine asks for the same model, so
# record what wrote it rather than leaving it to be discovered on a rebuild.
if [[ -f "inference-service/artifacts_full/extraction_cache.json" ]]; then
  python - <<'PY'
import json
p = "inference-service/artifacts_full/extraction_cache.json"
blob = json.load(open(p, encoding="utf-8"))
key, n = blob.get("cache_key"), len(blob.get("chunks", {}))
print(f"\nextraction cache: {n} chunks, written by {key!r}")
print("  To reuse it, the target machine's .env must request exactly that model,")
print("  and ARTIFACTS_DIR must point at artifacts_full/. A mismatch is a hard")
print("  error, not a silent re-extraction -- but it is still a wasted trip.")
PY
fi

if [[ ${#missing[@]} -gt 0 ]]; then
  echo
  echo "WARNING ${#missing[@]} expected path(s) missing — did the run finish?"
fi

if [[ $CHECK -eq 1 ]]; then
  echo; echo "(--check: nothing written)"
  exit 0
fi

tar czf "$OUT" "${present[@]}" || { echo "tar failed" >&2; exit 1; }
echo
echo "wrote $OUT  ($(du -sh "$OUT" | cut -f1))"
echo
echo "On the laptop:"
echo "  scp user@server:$OUT ."
echo "  tar xzf $(basename "$OUT") -C /path/to/ComplianceGateway"
echo
echo "Then rebuild the graph with no API calls (embeddings and writes only):"
echo "  cd inference-service"
echo "  ARTIFACTS_DIR=\$PWD/artifacts_full python -m ingestion.build_indexes \\"
echo "      --corpus-json ../dataset/corpus/full_corpus.json --workers 8"
echo "  python -m ingestion.load_implements --edges ../dataset/corpus/full_citations.json"
echo "  # the log must say 959/959 chunks served from cache"
