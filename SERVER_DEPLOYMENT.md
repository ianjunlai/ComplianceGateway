# Running on a shared GPU server, over SSH, without Docker

No root and no Docker needed. conda supplies Python and the JDK; Neo4j and Ollama are
tarballs you unpack in your home directory and start.

**Part 1 gets you to results.** Kafka, Maven, the Spring gateway and JMeter are needed
*only* for the load experiment (E3) — `run_eval.py` calls the pipeline directly, with no
broker and no gateway anywhere in the chain. Skip Part 2 until E1/E2 has finished.

> Neither pip nor conda can install the Neo4j database or the Kafka broker — both are JVM
> applications, not Python packages. The `neo4j` and `confluent-kafka` entries already in
> `requirements.txt` are the *client* libraries. The servers come from tarballs; everything
> else comes from conda and pip.

---

# Part 1 — E1/E2

## 1. Copy the two things git does not carry

`inference-service/artifacts/` and `inference-service/.env` are gitignored. Run these on
your **laptop**, after cloning on the server:

```bash
# on the server
git clone https://github.com/ianjunlai/ComplianceGateway.git

# on the laptop
scp -r inference-service/artifacts user@server:~/ComplianceGateway/inference-service/
scp    inference-service/.env      user@server:~/ComplianceGateway/inference-service/
```

`artifacts/extraction_cache.json` is the stored result of a 430,704-token, ~90-minute
extraction pass. Without it step 5 silently re-runs all 345 extractions — re-spending the
budget *and* building a graph that differs from the one the reported results came from.

Then edit the copied `.env` on the server. Two values need correcting:

```
SLM_MODEL=llama3.1:8b-instruct-q4_K_M      # the laptop stand-in was llama3.2:1b
EXTRACTION_MODEL=qwen-plus                 # what the GDPR cache was built with
```

`EXTRACTION_MODEL` matters more than it looks. The extraction cache is keyed by
provider, model and prompt profile, and the GDPR cache was written under `qwen-plus`.
Leaving the laptop's `qwen-plus-2025-07-28` — which belongs to the public-benchmark run —
makes step 5 stop with a cache-mismatch error rather than silently re-extracting 345
passages. That guard is deliberate, but it is easier to set the value correctly now than
to debug the error later.

## 2. Pick a GPU

```bash
nvidia-smi        # find a card with no other tenants
export CUDA_VISIBLE_DEVICES=1
```

Export it in **every** shell you use, Python included — not just for Ollama. The embedding
model is constructed without a device argument in
[pipeline/embeddings.py](inference-service/pipeline/embeddings.py), so
sentence-transformers auto-selects `cuda:0`, which on a shared box is whichever card
someone else is already on.

## 3. conda environment

```bash
conda create -n cg python=3.11 -y
conda activate cg
conda install -c conda-forge openjdk=21 -y      # Neo4j needs a JRE; the tarball has none
export JAVA_HOME=$CONDA_PREFIX

cd ~/ComplianceGateway/inference-service
pip install -r requirements.txt                 # pulls torch, ~2.5 GB
```

## 4. Neo4j

```bash
cd ~
curl -O https://dist.neo4j.org/neo4j-community-5.24.0-unix.tar.gz
tar xf neo4j-community-5.24.0-unix.tar.gz
cd neo4j-community-5.24.0
bin/neo4j-admin dbms set-initial-password compliance123
bin/neo4j start
```

`bin/neo4j status` should report it running. Default ports are 7687 (bolt) and 7474
(browser); if either is taken, see the appendix.

## 5. Ollama

The tarball bundles its own CUDA runtime — no root, no systemd unit.

Ollama ships Linux builds as zstd archives (`.tar.zst`, ~1.4 GB); the older
`ollama-linux-amd64.tgz` no longer exists and `https://ollama.com/download/...tgz`
now redirects to a 404.

```bash
mkdir -p ~/ollama && cd ~/ollama
curl -L -o ollama.tar.zst \
  https://github.com/ollama/ollama/releases/latest/download/ollama-linux-amd64.tar.zst
tar --zstd -xf ollama.tar.zst
```

**`tar --zstd` needs GNU tar 1.31 or newer, and Ubuntu 20.04 ships 1.30.** Check with
`tar --version`. If it is older, decompress separately — and if `zstd` is missing too,
conda has it, so this still needs no root:

```bash
conda install -c conda-forge zstd -y      # only if `which zstd` finds nothing
zstd -dc ollama.tar.zst | tar -xf -
```

Then:

```bash
export OLLAMA_MODELS=$HOME/ollama/models        # ~5 GB, watch your home quota
~/ollama/bin/ollama serve &
~/ollama/bin/ollama pull llama3.1:8b-instruct-q4_K_M
```

## 6. Build the graph — and stop here if it does not match

```bash
cd ~/ComplianceGateway/inference-service
python -m ingestion.build_indexes
```

The log **must** say `345/345 chunks served from cache` and make no API calls. If it starts
calling the API, stop it: `artifacts/` did not transfer.

Then check the graph and the retrieval layer:

```bash
python evaluation/ablation/compare_all_strategies.py
#   expect R@10  vector_rag 0.537 / hybrid 0.537 / light_rag 0.567 / hippo_rag 0.179
```

Takes a couple of minutes and calls no SLM. In Cypher, the counts should be 345 `Chunk`,
1,673 `Entity`, 3,684 `RELATES`, 4,394 `MENTIONED_IN`, and `SHOW INDEXES` should list
`chunk_vec`, `entity_vec` and `edge_vec` all ONLINE.

**A mismatch means the graph is not the one the reported numbers came from**, so nothing
measured after it would be comparable to the thesis. Fix it before spending GPU time.

## 7. Run E1/E2

Under `tmux`, so a dropped SSH session does not kill a multi-hour job. One strategy per
process — the active strategy is fixed when the process starts.

```bash
tmux new -s eval
conda activate cg && export CUDA_VISIBLE_DEVICES=1
cd ~/ComplianceGateway/inference-service

python -m evaluation.run_eval --strategy zero_shot  --judge --run-id server1
python -m evaluation.run_eval --strategy vector_rag --judge --run-id server1
python -m evaluation.run_eval --strategy hybrid     --judge --run-id server1
python -m evaluation.run_eval --strategy light_rag  --judge --run-id server1
python -m evaluation.run_eval --strategy hippo_rag  --judge --run-id server1
```

Detach with `Ctrl-b d`, reattach with `tmux attach -t eval`. Expect roughly 1.5–2 hours for
all five on an A100.

Results land in the repo-root `results/` regardless of the working directory: a `.jsonl`
appended after every query, and a `.json` summary at the end. If a run dies, rerun the
**same** command with `--resume` appended — failed queries are retried, scored ones are
not.

`--judge` adds one qwen-max call per query (800 across all five). It can be dropped now and
added later against the same `--run-id`.

**Part 1 ends here.** You have the E1/E2 results.

---

# Part 2 — E3, the load experiment

Only needed for the load test. Everything below is additional to Part 1.

## 8. Kafka, Maven, JMeter

```bash
conda install -c conda-forge maven -y

cd ~
curl -O https://archive.apache.org/dist/kafka/3.7.0/kafka_2.13-3.7.0.tgz
tar xf kafka_2.13-3.7.0.tgz && cd kafka_2.13-3.7.0
```

Two edits in `config/kraft/server.properties`:

```
log.dirs=/home/<you>/kafka-logs
auto.create.topics.enable=false
```

`log.dirs` defaults to `/tmp/kraft-combined-logs`. That works for a single sitting, but
`/tmp` is cleared on reboot — taking the cluster metadata with it, so the next start needs
`kafka-storage.sh format` again — it is shared with every other user of the machine, and it
is sometimes a small tmpfs that a multi-hour run can fill.

`auto.create.topics.enable` is **not present in the shipped file**; the broker default is
`true`. Adding it is still valid. The consequence is smaller than it looks — the same file
sets `num.partitions=1`, so an auto-created topic would get one partition anyway, which is
what the gateway asks for. Check with `grep -E "^num.partitions" config/kraft/server.properties`.
The reason to add the line regardless is diagnostic: with auto-creation off, a mistyped
topic name fails loudly instead of silently producing into a new empty topic that nothing
consumes.

```bash
bin/kafka-storage.sh format -t $(bin/kafka-storage.sh random-uuid) -c config/kraft/server.properties
bin/kafka-server-start.sh -daemon config/kraft/server.properties

cd ~ && curl -O https://archive.apache.org/dist/jmeter/binaries/apache-jmeter-5.6.3.tgz
tar xf apache-jmeter-5.6.3.tgz
```

## 9. Start the services

**One inference backend at a time.** The consumer and `sync_api` drive the same GPU, so a
measurement taken with both up describes their interference rather than either integration
mode.

```bash
# gateway — all three conditions need it
cd ~/ComplianceGateway/gateway-service && mvn spring-boot:run

# then EITHER, for the EDA condition:
cd ~/ComplianceGateway/inference-service && python consumer_main.py
# OR, for both synchronous conditions:
uvicorn sync_api:app --port 8000
```

### "Port 8080 was already in use"

First find out whose it is. On a shared machine the likeliest answer is your own gateway
from an earlier attempt, still running:

```bash
ss -ltnp | grep :8080
```

If the process is yours, kill it — that is the whole fix, and it keeps every default intact.

If the port genuinely belongs to another tenant, **move the gateway with an environment
variable rather than by editing `application.yml`**. Spring's relaxed binding maps
`SERVER_PORT` onto `server.port`, so no file in the repository changes:

```bash
SERVER_PORT=28080 \
GATEWAY_INFERENCE_SYNC_URL=http://localhost:8000/infer \
mvn spring-boot:run
```

Two other places name the port, and both have a way to avoid an edit:

| | How to follow the move |
|---|---|
| JMeter plan | `-JPORT=28080` on the command line; the `.jmx` stays as it is |
| `dashboard/app.js` | hardcodes `localhost:8080` — leave it, and point the **left** side of the SSH tunnel at 8080: `ssh -L 8080:localhost:28080 …` |

The tunnel trick is worth understanding rather than copying: the browser runs on your
laptop and only ever sees local port 8080, so as long as the tunnel's local end is 8080 the
dashboard needs no change no matter which port the server listens on. Editing `app.js`
instead would put a machine-specific port into version control.

## 10. Run the load test

Rehearse against the stub first — a fixed delay instead of a pipeline, so extractor and
timeout mistakes surface in seconds rather than after an hour of GPU time:

```bash
python loadtest/stub_inference.py --delay 3
python loadtest/stub_inference.py --delay 250    # trips the 240 s acquire timeout
```

Then measure one real request and size the concurrency levels from it — EDA drain time is
roughly `threads × single-request time`, which sets the wall-clock cost of the whole
experiment.

```bash
~/apache-jmeter-5.6.3/bin/jmeter -n -t ~/ComplianceGateway/loadtest/compliance_gateway.jmx \
  -JHOST=localhost -JPORT=8080 \
  -JEDA_THREADS=10 -JRAMP=10 -JDURATION=120 \
  -l ~/ComplianceGateway/results/eda-c10.jtl
```

**Pass `-JPORT` explicitly, and match it to whatever the gateway actually bound to.** If
the port moved (see §9), a plan pointing at 8080 does not stop — JMeter records every
sampler as a connection failure and reports it only in the summary, so a whole level's
worth of GPU time is spent before the mistake is visible. Confirm with one request before
starting a level:

```bash
PORT=28080          # whatever the gateway bound to; keep it in the shell
curl -s -o /dev/null -w '%{http_code}\n' -X POST http://localhost:$PORT/api/v1/audit \
  -H 'Content-Type: application/json' \
  -d '{"source_system":"uni_a","audit_query":"test"}'
```

Expect **202**. A **501** or any other unexpected code means you are talking to whatever
else holds that port — on a shared machine that is another tenant's service, not a broken
gateway. Set `PORT` once in the shell and reuse it in the `curl` and in `-JPORT`, rather
than typing the number twice and getting one of them wrong.

### Driving the whole matrix

The condition is selected by giving exactly one Thread Group a non-zero thread count; all
three default to 0, so nothing in the `.jmx` changes between runs and the GUI is not
involved:

```bash
jmeter … -JEDA_THREADS=25        # EDA
jmeter … -JSYNC_THREADS=25       # sync-unbounded
jmeter … -JTHROTTLE_THREADS=25   # sync-throttled
```

`loadtest/run_e3.sh` drives all 45 runs over those properties, and does the parts that are
easy to get wrong by hand:

```bash
cd ~/ComplianceGateway
PORT=28080 ./loadtest/run_e3.sh --dry-run     # print the plan first
PORT=28080 ./loadtest/run_e3.sh               # ~3.5-4 h; run it under tmux
```

- switches the inference backend with the condition — `consumer_main.py` for EDA,
  `sync_api` for both synchronous ones — and never leaves two running
- **submits one real request and waits for a decision before each level.** This is the
  check that would have caught the wasted run described above; JMeter's own error rate
  cannot, because a poll returning `PENDING` is a 200 and a completely dead pipeline
  reports 0% errors
- waits for `queue_depth` to reach 0 between levels, so a backlog never leaks into the next
  curve
- repetition-major ordering, so drift over a four-hour session lands on all three
  conditions equally rather than on whichever ran last
- one timestamped `.jtl` per run, plus `results/e3/manifest.csv` recording samples,
  free GPU memory and gateway counters before and after each level
- skips any run whose `.jtl` exists, so an interrupted matrix resumes where it stopped

Full profile and failure-mode taxonomy: [loadtest/README.md](loadtest/README.md).

### How long the whole thing takes

Wall clock is dominated by one number you have to measure first: `T`, the mean
single-request pipeline time. For each `(condition, level, repetition)` a run costs about
`ramp + duration + C × T` — the tail is the queue draining after load stops.

Across 5 levels (C = 1, 10, 25, 50, 100, summing to 186 threads), 3 conditions and 3
repetitions:

    total ≈ 9 × (5 × 130 s + 186 × T)  =  1.6 h + 0.47 h per second of T

| `T` | EDA + sync, 3 reps | with the sync timeout tails |
|---|---|---|
| 3 s | ~3.0 h | ~4.5 h |
| 6 s | ~4.4 h | ~6 h |
| 10 s | ~6.3 h | ~8 h |

The second column matters more than it looks. At C = 25 and above the two synchronous
conditions are *supposed* to fail, and they fail by timing out: 240 s for sync-unbounded,
up to 480 s for sync-throttled, which a run must wait out before it can end. That tail is
roughly a third of the total and is not wasted — it is the RQ3 result.

To cut it: drop to 2 repetitions, or drop C = 50 (the curve is defined by 1 / 10 / 25 /
100). Do not shorten `DURATION` below 120 s — the queue-depth curve needs a steady state to
be visible in.

## 11. Watch the dashboard from your laptop

Nothing needs changing. `app.js` targets `http://localhost:8080/api/v1`
([dashboard/app.js:5](dashboard/app.js)) and the gateway allows any origin, so the browser
runs on your laptop and `localhost` resolves through an SSH tunnel.

```bash
# on the laptop — the right of each pair is the port the gateway bound to on the server
ssh -L 8080:localhost:8080 -L 7474:localhost:7474 -L 7687:localhost:7687 user@server

# if the gateway was moved to 28080 (see §9), only the right side changes:
ssh -L 8080:localhost:28080 -L 7474:localhost:7474 -L 7687:localhost:7687 user@server

# in a second local terminal
cd dashboard && python -m http.server 5173
# open http://localhost:5173
```

The **local** port — the left of each pair — must be **8080 exactly**, because `app.js`
hardcodes it; `-L 18080:...` silently breaks the page. The **remote** port is whatever the
gateway is listening on, and the two need not match. Make sure nothing on your laptop is
already holding 8080.

Neo4j Browser at `http://localhost:7474` works through the same tunnel, but only if 7687 is
forwarded too: the browser page opens its own bolt connection directly.

---

# Appendix

## If a port is already taken

`ss -ltnp | grep -E '7474|7687|9092|8080|8000|11434'` shows what is occupied and, with `-p`,
by whom. Check the owner before remapping anything: a port held by your own earlier run is
fixed by killing that process, and every default stays intact.

Everything is configurable without touching code — `NEO4J_URI`, `KAFKA_BOOTSTRAP` and
`OLLAMA_HOST` in `.env`; `SERVER_PORT`, `SPRING_KAFKA_BOOTSTRAP_SERVERS` and
`GATEWAY_INFERENCE_SYNC_URL` as environment variables for the gateway;
`server.bolt.listen_address` and `server.http.listen_address` in `conf/neo4j.conf`. Prefer
these to editing the files: a port number chosen for one shared machine does not belong in
version control. For the gateway specifically see
["Port 8080 was already in use"](#port-8080-was-already-in-use) above, including how the
SSH tunnel keeps `dashboard/app.js` untouched.

One case is not obvious. **Moving Kafka's port means changing `advertised.listeners`, not
just `listeners`** — a broker tells clients where to reconnect, so if the advertised value
still names the old port the client connects successfully, gets redirected, and fails
against nothing. Set both in `config/kraft/server.properties`.

## Getting results back

`results/` is gitignored, so committing from it needs `-f`. The simplest route leaves no
GitHub credential on a machine other people use:

```bash
# on the laptop
scp -r user@server:~/ComplianceGateway/results/ ./results/
git add -f results/ && git commit -m "results from the GPU server" && git push
```

Raw `.jtl` files reach tens of megabytes at C=100 — `gzip` them first; GitHub warns above
50 MB per file. If you would rather push directly from the server, create an SSH key there
and add it as a repository **deploy key** with write access, then delete it when the
experiments are done.

## Things that actually go wrong

**The build makes API calls instead of reporting `345/345 chunks served from cache`** —
`artifacts/` did not transfer. Stop immediately; continuing costs ~430k tokens and produces
a different graph.

**`compare_all_strategies.py` prints something other than 0.537 / 0.537 / 0.567 / 0.179** —
the graph differs from the measured one. Check the chunk and entity counts before going
further. Figures from before August 2026 read 0.196 for light_rag and 0.171 for hippo_rag;
those predate the retrieval fixes and are not a target to reproduce.

**Inference is far slower than ~6 s per query, or the GPU runs out of memory** —
`CUDA_VISIBLE_DEVICES` was not exported in that shell, so `cuda:0` landed on a card
somebody else is using.

**A long run dies when the SSH session drops** — it was not under `tmux`. Restart with the
same `--run-id` plus `--resume`.
