#!/usr/bin/env bash
# All three experiments, in one unattended pass.
#
#   E1/E2  corpus -> graph -> questions -> retrieval -> decisions   (run_full_experiment.sh)
#   E3     load test across three architectures                     (loadtest/run_e3.sh)
#
# They run strictly in sequence because both drive the same GPU. Everything
# that can be checked cheaply is checked before the first expensive step, so a
# missing service is discovered now rather than at 3 a.m. with four hours
# already spent.
#
#   ./run_all.sh --check      # verify prerequisites and exit
#   ./run_all.sh              # run everything
#   ./run_all.sh --skip-e3    # E1/E2 only
set -uo pipefail

REPO="${REPO:-$(cd "$(dirname "$0")" && pwd)}"
PYTHON="${PYTHON:-python}"
# The gateway's port. Exported so run_e3.sh uses the same one -- it defaults to
# 8080 independently, and a gateway moved aside for a port clash would
# otherwise pass the check here and fail preflight there, three hours later.
export PORT="${PORT:-8080}"
export SYNC_PORT="${SYNC_PORT:-8000}"
STAMP="$(date +%m%d-%H%M)"
LOGDIR="$REPO/results/full/logs"
mkdir -p "$LOGDIR" "$REPO/results/e3/logs"

SKIP_E3=0; CHECK=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-e3) SKIP_E3=1; shift ;;
    --check)   CHECK=1; shift ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

say() { printf '\n\033[1m=== %s  %s\033[0m\n' "$(date +%H:%M:%S)" "$*"; }
bad=0
need() {  # need <description> <command...>
  if "${@:2}" >/dev/null 2>&1; then
    printf '  ok    %s\n' "$1"
  else
    printf '  MISS  %s\n' "$1"; bad=$((bad + 1))
  fi
}

say "prerequisites"
need "Neo4j on 7687"        bash -c "exec 3<>/dev/tcp/localhost/7687"
need "Ollama on 11434"      curl -sf --max-time 5 http://localhost:11434/api/tags
need "the configured SLM is pulled" bash -c \
  "curl -sf --max-time 5 http://localhost:11434/api/tags | grep -q \"\$(grep '^SLM_MODEL=' \"$REPO/inference-service/.env\" | cut -d= -f2 | cut -d: -f1)\""
if [[ $SKIP_E3 -eq 0 ]]; then
  # E3 does not start these; it only swaps the Python backend between
  # conditions. Missing here means E3 fails after E1 has already run.
  need "Kafka on 9092"          bash -c "exec 3<>/dev/tcp/localhost/9092"
  need "gateway on $PORT"       curl -sf --max-time 5 "http://localhost:$PORT/api/v1/metrics"
  need "JMeter"                 test -x "${JMETER:-$HOME/apache-jmeter-5.6.3/bin/jmeter}"
  # The synchronous conditions route through the gateway to sync_api. run_e3.sh
  # starts sync_api itself, so this only checks the gateway was told where to
  # find it -- a mismatch here is what made the first attempt fail preflight
  # after the EDA runs had already completed.
  printf '  note  gateway sync-url must point at :%s (set GATEWAY_INFERENCE_SYNC_URL)\n' \
    "$SYNC_PORT"
fi
# From the service directory: config reads .env relative to the working
# directory, so running this from the repo root silently loads no keys and
# reports the shipped defaults instead of what the run will actually use.
(cd "$REPO/inference-service" && $PYTHON -c "
import os, sys; sys.path.insert(0, '.')
import config
print(f'  SLM        {config.SLM_MODEL}   num_ctx {config.SLM_NUM_CTX}')
print(f'  judge      {config.JUDGE_MODEL}')
print(f'  extraction {config.EXTRACTION_MODEL}')
print(f'  api key    {\"set\" if os.getenv(\"ALIBABA_API_KEY\") else \"MISSING\"}')
sys.exit(0 if os.getenv('ALIBABA_API_KEY') else 1)
") || bad=$((bad + 1))

if [[ $bad -gt 0 ]]; then
  echo
  echo "$bad prerequisite(s) missing — nothing started."
  [[ $SKIP_E3 -eq 0 ]] && echo "  (--skip-e3 drops the Kafka/gateway/JMeter requirements)"
  exit 1
fi
if [[ $CHECK -eq 1 ]]; then
  echo; echo "all present; --check made no changes"
  exit 0
fi

# ---------------------------------------------------------------- E1 and E2
say "E1 + E2  (graph, questions, retrieval, decisions)"
"$REPO/run_full_experiment.sh" 2>&1 | tee -a "$LOGDIR/run-$STAMP.log"
E12=${PIPESTATUS[0]}
[[ $E12 -eq 0 ]] && say "E1 + E2 finished" || say "E1 + E2 exited $E12 — continuing to E3"

# ---------------------------------------------------------------------- E3
# Runs even when E1/E2 failed. They share the graph but nothing else, and a
# night that produces one experiment is better than a night that produces none.
# Resumable in its own right: run_e3.sh skips any .jtl already written.
if [[ $SKIP_E3 -eq 0 ]]; then
  say "E3  (45 runs: 3 conditions x 5 levels x 3 reps, ~2 min each)"
  "$REPO/loadtest/run_e3.sh" 2>&1 | tee -a "$REPO/results/e3/logs/run-$STAMP.log"
  E3=${PIPESTATUS[0]}
  [[ $E3 -eq 0 ]] && say "E3 finished" || say "E3 exited $E3"
else
  E3="skipped"
fi

# ------------------------------------------------------------------ summary
say "summary"
printf '  E1 + E2 : %s\n' "$([[ $E12 -eq 0 ]] && echo ok || echo "exit $E12")"
printf '  E3      : %s\n' "$E3"
echo
$PYTHON - <<PY
import glob, json, os
os.chdir("$REPO")
runs = sorted(glob.glob("results/*-full*.json"))
if runs:
    print("  E1 results")
    for f in runs:
        try:
            s = json.load(open(f, encoding="utf-8"))["summary"]
        except Exception:
            print(f"    {os.path.basename(f):<34} unreadable"); continue
        acc = (s.get("decision") or {}).get("accuracy")
        fa = (s.get("faithfulness") or {}).get("mean")
        fmt = lambda v: f"{v:.3f}" if isinstance(v, (int, float)) else "  -  "
        print(f"    {os.path.basename(f):<34} n={s.get('n_scored', 0):<4} "
              f"acc={fmt(acc)} faith={fmt(fa)} failed={s.get('n_failed', 0)}")
else:
    print("  no E1 result files")
jtl = glob.glob("results/e3/*.jtl")
print(f"\n  E3: {len(jtl)} of 45 .jtl files present")
PY
echo
echo "  full logs   $LOGDIR/run-$STAMP.log"
echo "  next        cd inference-service && python -m evaluation.rescore \\"
echo "                  --run-id full\$(date +%m%d) --dataset ../dataset/qa_v2.json"
