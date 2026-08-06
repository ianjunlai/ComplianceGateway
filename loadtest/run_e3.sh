#!/usr/bin/env bash
# Drive the full E3 matrix: 3 conditions x 5 concurrency levels x 3 repetitions.
#
# Everything the plan needs is a JMeter property now, so no file is edited
# between runs and no GUI is involved. What still needs orchestrating is the
# inference backend: EDA is served by consumer_main.py and the two synchronous
# conditions by sync_api, and they must never run together because they share
# one GPU -- a measurement taken with both up describes their interference
# rather than either integration mode.
#
#   ./run_e3.sh                      # the whole matrix
#   ./run_e3.sh --dry-run            # print the plan, touch nothing
#   ./run_e3.sh --conditions eda     # one condition
#   ./run_e3.sh --levels 1,10        # a subset, e.g. to calibrate first
#
# Safe to re-run: a level whose .jtl already exists is skipped, so an
# interrupted matrix resumes instead of starting over.
set -uo pipefail

REPO="${REPO:-$HOME/ComplianceGateway}"
JMETER="${JMETER:-$HOME/apache-jmeter-5.6.3/bin/jmeter}"
PYTHON="${PYTHON:-python}"
HOST="${HOST:-localhost}"
PORT="${PORT:-8080}"           # the gateway's port; match SERVER_PORT
SYNC_PORT="${SYNC_PORT:-8000}"
CUDA_DEVICE="${CUDA_VISIBLE_DEVICES:-0}"

CONDITIONS="eda,sync,throttled"
LEVELS="1,10,25,50,100"
REPS="${REPS:-3}"
RAMP="${RAMP:-10}"
DURATION="${DURATION:-120}"
DRY_RUN=0

OUT="$REPO/results/e3"
MANIFEST="$OUT/manifest.csv"
LOGS="$OUT/logs"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)     DRY_RUN=1; shift ;;
    --conditions)  CONDITIONS="$2"; shift 2 ;;
    --levels)      LEVELS="$2"; shift 2 ;;
    --reps)        REPS="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

mkdir -p "$OUT" "$LOGS"
[[ -f "$MANIFEST" ]] || echo "started_at,rep,condition,threads,jtl,samples,gpu_free_mb_before,gpu_free_mb_after,submitted,completed,errors,queue_depth_after" > "$MANIFEST"

log() { printf '%s  %s\n' "$(date +%H:%M:%S)" "$*"; }
metrics() { curl -s --max-time 10 "http://$HOST:$PORT/api/v1/metrics" 2>/dev/null; }
gpu_free() { nvidia-smi --id="$CUDA_DEVICE" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1; }

json_field() {  # json_field <json> <key>
  "$PYTHON" -c 'import json,sys; d=json.loads(sys.argv[1] or "{}"); print(d.get(sys.argv[2],""))' "$1" "$2" 2>/dev/null
}

BACKEND_PID=""
BACKEND_KIND=""

stop_backend() {
  [[ -z "$BACKEND_PID" ]] && return 0
  log "stopping $BACKEND_KIND (pid $BACKEND_PID)"
  kill "$BACKEND_PID" 2>/dev/null
  # The embedding model takes a moment to release; SIGKILL only if it lingers.
  for _ in $(seq 1 20); do kill -0 "$BACKEND_PID" 2>/dev/null || break; sleep 1; done
  kill -0 "$BACKEND_PID" 2>/dev/null && kill -9 "$BACKEND_PID" 2>/dev/null
  BACKEND_PID=""; BACKEND_KIND=""
  sleep 3
}

start_backend() {  # start_backend consumer|sync
  local kind="$1"
  [[ "$BACKEND_KIND" == "$kind" ]] && return 0
  stop_backend
  cd "$REPO/inference-service" || exit 1
  if [[ "$kind" == "consumer" ]]; then
    CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" nohup "$PYTHON" consumer_main.py \
      > "$LOGS/consumer.log" 2>&1 &
  else
    CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" nohup "$PYTHON" -m uvicorn sync_api:app \
      --port "$SYNC_PORT" > "$LOGS/sync_api.log" 2>&1 &
  fi
  BACKEND_PID=$!
  BACKEND_KIND="$kind"
  log "started $kind (pid $BACKEND_PID), loading the embedding model"
  sleep 25
}

# A run that measures nothing looks identical to a healthy one in JMeter's
# summary: a poll returning PENDING is a 200, so a broken pipeline reports 0%
# errors. The only honest check is whether a single request actually completes.
preflight() {
  local body rid decision
  body=$(curl -s --max-time 15 -X POST "http://$HOST:$PORT/api/v1/audit" \
      -H 'Content-Type: application/json' \
      -d '{"source_system":"uni_a","audit_query":"May a university transfer student data to a US partner?"}')
  rid=$(json_field "$body" request_id)
  if [[ -z "$rid" ]]; then
    log "PREFLIGHT FAILED: no request_id from the gateway at $HOST:$PORT"
    log "  response: ${body:0:200}"
    return 1
  fi
  for _ in $(seq 1 60); do
    decision=$(json_field "$(curl -s --max-time 10 "http://$HOST:$PORT/api/v1/audit/$rid")" decision)
    [[ -n "$decision" ]] && { log "preflight ok (decision=$decision)"; return 0; }
    sleep 2
  done
  log "PREFLIGHT FAILED: request $rid never completed in 120s — is the backend consuming?"
  return 1
}

drain() {
  local depth
  for _ in $(seq 1 180); do
    depth=$(json_field "$(metrics)" queue_depth)
    [[ "$depth" == "0" || -z "$depth" ]] && return 0
    sleep 5
  done
  log "WARNING: queue still at ${depth:-?} after 15 min; the next run starts with a backlog"
}

IFS=',' read -ra COND_ARR <<< "$CONDITIONS"
IFS=',' read -ra LEVEL_ARR <<< "$LEVELS"

total=$(( ${#COND_ARR[@]} * ${#LEVEL_ARR[@]} * REPS )); done_n=0
log "matrix: ${#COND_ARR[@]} conditions x ${#LEVEL_ARR[@]} levels x $REPS reps = $total runs"
log "gateway http://$HOST:$PORT   output $OUT"

# Repetition-major: a whole pass over every condition before repeating one, so
# machine drift over the session lands on all conditions equally instead of
# being absorbed entirely by whichever ran last.
for rep in $(seq 1 "$REPS"); do
  for cond in "${COND_ARR[@]}"; do
    case "$cond" in
      eda)       prop="EDA_THREADS";      backend="consumer" ;;
      sync)      prop="SYNC_THREADS";     backend="sync" ;;
      throttled) prop="THROTTLE_THREADS"; backend="sync" ;;
      *) echo "unknown condition: $cond" >&2; exit 2 ;;
    esac

    for c in "${LEVEL_ARR[@]}"; do
      done_n=$((done_n + 1))
      jtl="$OUT/${cond}-c${c}-rep${rep}.jtl"
      if [[ -f "$jtl" ]]; then
        log "[$done_n/$total] skip $cond C=$c rep=$rep (exists)"
        continue
      fi
      if [[ $DRY_RUN -eq 1 ]]; then
        log "[$done_n/$total] would run $cond C=$c rep=$rep -> $(basename "$jtl")  [-J$prop=$c, backend=$backend]"
        continue
      fi

      start_backend "$backend"
      preflight || { log "aborting: fix the backend before spending GPU time on the matrix"; stop_backend; exit 1; }
      drain

      before=$(metrics); gpu_before=$(gpu_free)
      log "[$done_n/$total] $cond C=$c rep=$rep"
      "$JMETER" -n -t "$REPO/loadtest/compliance_gateway.jmx" \
        -JHOST="$HOST" -JPORT="$PORT" -J"$prop"="$c" \
        -JRAMP="$RAMP" -JDURATION="$DURATION" \
        -l "$jtl" > "$LOGS/${cond}-c${c}-rep${rep}.log" 2>&1
      drain
      after=$(metrics); gpu_after=$(gpu_free)

      samples=$(( $(wc -l < "$jtl" 2>/dev/null || echo 1) - 1 ))
      # Zero samples means every Thread Group had 0 threads -- the property name
      # did not reach the plan. Worth stopping for: the rest of the matrix would
      # produce empty files just as quietly.
      if [[ "$samples" -le 0 ]]; then
        log "ERROR: 0 samples. Does the plan still define $prop?"
        stop_backend; exit 1
      fi
      printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
        "$(date -Iseconds)" "$rep" "$cond" "$c" "$(basename "$jtl")" "$samples" \
        "${gpu_before:-}" "${gpu_after:-}" \
        "$(json_field "$after" submitted)" "$(json_field "$after" completed)" \
        "$(json_field "$after" errors)" "$(json_field "$after" queue_depth)" >> "$MANIFEST"
      log "    $samples samples -> $(basename "$jtl")"
    done
  done
done

stop_backend
log "matrix complete. manifest: $MANIFEST"
