# Load testing plan — E3, RQ3 (thesis 4.5.3)

Compares three integration modes under identical load, same pipeline behind all:

| Condition | Endpoint | Expected failure mode |
|---|---|---|
| EDA | `POST /api/v1/audit` + poll `GET /api/v1/audit/{id}` | none (backlog grows, drains after) |
| sync-unbounded | `POST /api/v1/audit/sync` | thread saturation, timeouts (504) |
| sync-throttled | `POST /api/v1/audit/sync-throttled` | held connections, queue timeouts (503) |

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
JMeter CSV Data Set Config; query bodies sampled from `../dataset/qa_dataset.json`
(export a CSV of query_text values once the dataset is generated).

## Recorded metrics

- Per request: HTTP status, response time; EDA additionally submit→result E2E
- Aggregate per level: p50/p95/p99, throughput (req/min), error rate
  (4xx/5xx/timeout, taxonomy per thesis 4.5.3)
- Gateway `GET /api/v1/metrics` sampled every 5 s during the run
  → queue-depth curve → backlog recovery time := time from load-stop until
  queueDepth returns to 0
- 3 repetitions per (condition × level); report mean ± std

## JMeter plan structure (to be built as compliance_gateway.jmx)

```
Test Plan
├── CSV Data Set Config (source_systems.csv)
├── CSV Data Set Config (queries.csv)
├── Thread Group "EDA" (N threads)
│   ├── HTTP POST /api/v1/audit  → JSON Extractor (requestId)
│   └── While Controller (status == PENDING, timeout 300s)
│       ├── HTTP GET /api/v1/audit/${requestId}
│       └── Constant Timer 1000 ms
├── Thread Group "sync-unbounded": HTTP POST /api/v1/audit/sync
├── Thread Group "sync-throttled": HTTP POST /api/v1/audit/sync-throttled
└── Listeners: Aggregate Report + Simple Data Writer (CSV for analysis)
```

Run one thread group at a time (others disabled); restart the inference
consumer between conditions so no residual backlog contaminates the next run.
