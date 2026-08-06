# Load testing plan — E3, RQ3 (thesis 4.5.3)

Compares three integration modes under identical load, same pipeline behind all:

| Condition | Endpoint | Expected failure mode |
|---|---|---|
| EDA | `POST /api/v1/audit` + poll `GET /api/v1/audit/{id}` | none (backlog grows, drains after) |
| sync-unbounded | `POST /api/v1/audit/sync` | thread saturation, timeouts (504) |
| sync-throttled | `POST /api/v1/audit/sync-throttled` | held connections, queue timeouts (503) |

## Request body (all three POST endpoints)

Field names are snake_case, matching the `Audit_Request_Event` contract:

```json
{"source_system": "uni_a", "audit_query": "..."}
```

Both fields are required and non-blank; anything else is rejected at ingress
with `400` and a `violations` list. Note for the JMeter plan: a wrong key name
(camelCase, say) binds to null and is a validation failure, so a mis-typed body
shows up as a wall of 400s rather than as a load result — check one sampler
response before starting a full run.

## Test profile (per condition)

- Concurrency levels: 1, 10, 25, 50, 100 threads
- Per level: ramp-up 10 s → sustained 120 s → stop load → **keep polling until
  backlog drains** (EDA only; measures backlog recovery time, B9)
- **Sampler response timeout, per condition (do not share one value across
  conditions — see rationale below):**
  | Condition | Sampler timeout | Why |
  |---|---|---|
  | `sync-unbounded` | 260 s | > `inference.timeout-ms` (240 s) with margin |
  | `sync-throttled` | 550 s | > `acquire-timeout-ms` + `inference.timeout-ms` (240+240=480 s) — the request can wait a FULL queue turn and THEN still time out on the call itself. A shorter sampler timeout makes JMeter give up before the gateway's own 503, which shows up as a client-side `SocketTimeoutException` instead of a clean 503 and corrupts the failure-mode taxonomy (4.5.3). |
  | EDA (poll loop) | 300 s **per poll cycle is not the bound** — see note below | — |
- **EDA poll-timeout caveat:** at C=100 the request backlog can take well over
  300 s to drain (that is the whole point of B9). If the While-Controller's
  overall timeout is set to 300 s, deep-queue requests will be marked FAILED
  purely because they were still legitimately queued — not because the system
  broke — which would unfairly inflate EDA's error rate relative to the sync
  conditions and undermine the RQ3 narrative. Size this timeout from the
  worst-case queue depth at C=100 (≈ C × mean single-request pipeline time),
  not copied from the sync conditions; finalize the exact value once
  `pipeline.total_ms` is measured empirically (E1/E2 single-user baseline).
- Poll interval (EDA): 1 s; E2E latency = submit instant → result available

## Federated load profile (A2)

`source_system` values rotate from `source_systems.csv` (uni_a…uni_e) via a
JMeter CSV Data Set Config; query bodies come from `queries.csv` — 144 rows
exported from `../dataset/qa_dataset.json` (the unanswerable stratum is excluded:
those have no gold clauses and are an E2 construct, not a load-shape one).

Both CSV readers set `fileEncoding=UTF-8` explicitly. Two queries contain a
typographic apostrophe, and JMeter defaults to the platform encoding — GBK on
this project's Windows install — which would corrupt the JSON body.

Regenerate `queries.csv` after any change to the QA set:

```python
import json, csv, io
rows = [q for q in json.load(open('dataset/qa_dataset.json', encoding='utf-8'))
        if q['hop_type'] != 'unanswerable']
with io.open('loadtest/queries.csv', 'w', encoding='utf-8', newline='') as f:
    w = csv.writer(f, quoting=csv.QUOTE_ALL)
    w.writerow(['audit_query'])
    for q in rows:
        w.writerow([' '.join(q['query_text'].split())])
```

## Recorded metrics

- Per request: HTTP status, response time; EDA additionally submit→result E2E
- Aggregate per level: p50/p95/p99, throughput (req/min), error rate
  (4xx/5xx/timeout, taxonomy per thesis 4.5.3)
- Gateway `GET /api/v1/metrics` sampled every 5 s during the run
  → queue-depth curve → backlog recovery time := time from load-stop until
  `queue_depth` returns to 0
- 3 repetitions per (condition × level); report mean ± std

## Running `compliance_gateway.jmx`

Three Thread Groups, one per condition; only EDA is enabled in the committed
file. Every knob is a JMeter property, so a sweep needs no edits to the plan:

```bash
jmeter -n -t loadtest/compliance_gateway.jmx \
       -JHOST=localhost -JPORT=8080 \
       -JEDA_THREADS=10 -JRAMP=10 -JDURATION=120 \
       -l results/eda-c10.jtl
```

`HOST` (localhost), `PORT` (8080), `RAMP` and `DURATION` all take `-J` overrides.

**The condition is chosen by which thread count you set**, not by enabling Thread
Groups in the GUI. Each group reads its own property and all three default to 0,
and a group with 0 threads starts nothing:

| Condition | Property |
|---|---|
| EDA | `-JEDA_THREADS=<C>` |
| sync-unbounded | `-JSYNC_THREADS=<C>` |
| sync-throttled | `-JTHROTTLE_THREADS=<C>` |

Naming none of them produces a run with zero samples — visibly wrong, rather
than quietly measuring a condition you did not intend. There is no longer a
`THREADS` property; a command still passing `-JTHREADS` gets that empty run.

`run_e3.sh` drives the full 3 × 5 × 3 matrix over these properties, switching
the inference backend with the condition and checking that a real request
completes before each level. See [SERVER_DEPLOYMENT.md](../SERVER_DEPLOYMENT.md).

**Set `-JPORT` to whatever the gateway actually bound to.** On a shared machine
8080 is often taken and the gateway gets moved with `SERVER_PORT`; a plan still
aimed at 8080 does not fail fast — every sampler is recorded as a connection
error and the summary only appears at the end, so a full level of GPU time is
spent before the mistake shows. One `curl` against the submit endpoint,
expecting 202, costs nothing and settles it.

**Run one condition at a time.** All three drive the same single-GPU backend, so
two at once measures their interference rather than either mode. Restart the
inference consumer between conditions as well, or a residual backlog carries
into the next run's queue-depth curve.

### Two details in the plan that are load-bearing

**The submit response seeds the loop variable.** The 202 body is
`{request_id, status:"QUEUED"}`, and the JSON extractor reads *both* fields. The
`status` capture is not decoration: it resets `poll_status` at the start of every
thread iteration. Without it the leftover `DONE` from the previous iteration
makes the While condition false immediately, and the second request onward is
submitted but never polled — a silent under-count of E2E latency.

**Completion is detected on `$.status`, not `$.decision`.** A pending poll
returns `{status:"PENDING", request_id}`; a completed one has no `status` key at
all, so the extractor's default value `DONE` is what ends the loop. Keying on
`$.decision` looks equivalent and is not: that key is absent while pending, and a
JMeter extractor that fails to match leaves the previous value in place, so the
loop would never exit. Worse, the gateway answers `PENDING` for *any* unknown ID,
so a broken extractor polls a nonexistent request forever without ever raising an
error — at high thread counts that saturates the load generator, not the system
under test.

The plan has no overall poll timeout, deliberately: see the EDA poll-timeout
caveat above. A request still legitimately queued must not be scored as a
failure.

## Rehearsing the plan without the SLM

`stub_inference.py` serves the same contract as `inference-service/sync_api.py`
with a fixed delay instead of a pipeline. Nothing in the gateway changes — it
already calls `gateway.inference.sync-url`, so pointing that at this process is
the whole switch:

```bash
python loadtest/stub_inference.py --delay 3      # happy path, seconds per cycle
python loadtest/stub_inference.py --delay 250    # trips the 240 s acquire timeout
```

Debug the plan against the stub first, then switch to the real backend to
collect figures. `GET /stats` reports peak concurrent requests, which is the
direct evidence for whether the throttle held — measured on this project:
`peak_in_flight=2` unbounded versus `1` throttled at the same offered load.

Both designed failure modes were confirmed this way at `--delay 250`:

| Condition | Result | Cause |
|---|---|---|
| sync-unbounded | **504** at 240 s | backend slower than `inference.timeout-ms` |
| sync-throttled | **503** `throttle-queue-timeout` | permit not acquired within `acquire-timeout-ms` |

## Sizing the profile before the real run

The concurrency levels above assume a per-request cost that has not been measured
on the target machine. Measure it first — a single request through
`POST /api/v1/audit/sync` against the real backend — and derive the levels from
it, because the EDA drain time is roughly `threads × single-request time` and
that sets the wall-clock cost of the whole experiment. At 8 s per request C=100
drains in ~13 min; at 100 s it takes over two hours, for one level of one
condition, before repetitions.

**Memory, not CPU, is the binding constraint at the low end.** Each inference
process loads its own copy of the 1.3 GB embedding model, so the consumer and
`sync_api` together cost 4–6 GB before Kafka, Neo4j, the gateway, Ollama and
JMeter's own 1 GB heap. On an 11.6 GB laptop that pushed a single inference from
160 s to over 26 min through paging. Running one condition at a time is what
keeps this tractable; check free commit charge before starting a level.
