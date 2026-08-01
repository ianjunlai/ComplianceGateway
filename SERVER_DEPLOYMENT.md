# Running the experiments on a shared GPU server

A single continuous path — `git clone` on the server through to results pushed back to
GitHub — for a machine with **no Docker permission and no root**, which is the normal
situation on a shared university GPU box.

`docker compose up -d` cannot work there. It needs the daemon socket, and membership in
the `docker` group is equivalent to root, so administrators do not hand it out. Everything
below avoids it. Nothing here requires a code change: every endpoint the services talk to
is already read from an environment variable.

[REPRODUCIBILITY.md](REPRODUCIBILITY.md) remains the reference for *what* the experiments
are and what the parameters mean. This document is only about *where* they run.

---

## 0. Why a GPU, and which one

Ollama's official builds support CUDA and ROCm only. On a machine with integrated
graphics it falls back to CPU silently, and decode of `llama3.1:8b-instruct-q4_K_M` becomes
memory-bandwidth bound — roughly 10 tok/s on dual-channel DDR5, so about 90 s per RAG
query against about 6 s on an A100.

For E1/E2 that only turns a 1.5-hour job into an overnight one. **For E3 it invalidates the
experiment**: the gateway's `inference.timeout-ms` is 240 s, so at 90 s per request with no
throughput gain from CPU concurrency, the sync-unbounded condition returns 100 % 504 from
C=10 upward instead of degrading gradually, and the graded curve RQ3 rests on never
appears.

On a shared box, pick the idle card and pin everything to it:

```bash
nvidia-smi        # note which device has no other tenants
```

Every command below assumes `CUDA_VISIBLE_DEVICES=1`. Change the number to match.

---

## 1. What `git clone` does not give you

Two paths are gitignored and must be copied by hand **before** anything is built:

| Path | Size | Why it matters |
|---|---|---|
| `inference-service/artifacts/` | 2.2 MB | `extraction_cache.json` is the stored result of a 430,704-token, ~90-minute extraction pass. Without it the graph build silently re-runs all 345 extractions — re-spending the budget *and* producing a graph that differs from the one the reported results were measured on. |
| `inference-service/.env` | 1 KB | API keys and model selection. Copy the file; do not retype the keys. |

From the laptop:

```bash
git clone https://github.com/ianjunlai/ComplianceGateway.git    # on the server
scp -r inference-service/artifacts user@server:~/ComplianceGateway/inference-service/
scp inference-service/.env          user@server:~/ComplianceGateway/inference-service/
```

Then edit the copied `.env` on the server: `SLM_MODEL` is currently `llama3.2:1b`, a
laptop stand-in. Set it back to:

```
SLM_MODEL=llama3.1:8b-instruct-q4_K_M
```

---

## 2. Survey the server before installing anything

Three commands decide whether section 3 takes twenty minutes or two hours.

```bash
which podman podman-compose nerdctl apptainer singularity
ss -ltn | grep -E '7474|7687|9092|8080|8000|11434'
nvidia-smi
```

**On the container runtimes.** `podman` and `nerdctl` are rootless and both read the
existing `docker-compose.yml` unchanged — that is the easy path. If `podman` is present
but `podman-compose` is not, it installs into your own environment with
`pip install podman-compose`; no root needed. `apptainer` cannot do compose but can run the
same images individually.

**On the ports.** Anything `ss -ltn` lists is taken by another tenant. Remap rather than
contend — see the table in §3.

---

## 3. Bring up Kafka and Neo4j

### 3a. With a rootless container runtime (preferred)

```bash
cd ~/ComplianceGateway
podman-compose up -d          # or: nerdctl compose up -d
podman ps                     # cg-kafka and cg-neo4j should both be Up
```

If `ss -ltn` showed collisions, remap with this table. All alternatives are above 1024, so
rootless binding is fine:

| Service | Default | Suggested alternative |
|---|---|---|
| Neo4j HTTP (browser) | 7474 | 27474 |
| Neo4j Bolt | 7687 | 27687 |
| Kafka (external) | 9092 | **29092** |
| Gateway | 8080 | 28080 |
| sync inference API | 8000 | 28000 |
| Ollama | 11434 | 21434 |

> **Do not use 19092 for Kafka.** It is already the broker's INTERNAL listener in
> [docker-compose.yml](docker-compose.yml).

**Remapping Kafka is not just a port mapping.** A broker tells clients where to reconnect,
so if the advertised listener still says 9092 the client connects to 29092, is redirected
to 9092, and fails with nothing listening there. Change all three together:

```yaml
    ports:
      - "29092:29092"
    environment:
      KAFKA_LISTENERS: INTERNAL://:19092,CONTROLLER://:9093,EXTERNAL://:29092
      KAFKA_ADVERTISED_LISTENERS: INTERNAL://kafka:19092,EXTERNAL://localhost:29092
```

Neo4j needs only the port mapping changed (`27474:7474`, `27687:7687`), since the container
keeps its internal ports.

### 3b. From tarballs, if no container runtime exists

All of this installs into `$HOME` with no root.

```bash
conda create -n cg python=3.11 && conda activate cg
conda install -c conda-forge openjdk=21 maven     # Neo4j 5.24 and the gateway both need 21
export JAVA_HOME=$CONDA_PREFIX
```

**Neo4j 5.24 community:**

```bash
curl -O https://dist.neo4j.org/neo4j-community-5.24.0-unix.tar.gz
tar xf neo4j-community-5.24.0-unix.tar.gz && cd neo4j-community-5.24.0
```

Edit `conf/neo4j.conf` if the ports moved:

```
server.http.listen_address=:27474
server.bolt.listen_address=:27687
# The browser UI opens its own bolt connection from wherever it is displayed.
# Through an SSH tunnel that is your laptop, so advertise the LOCAL port:
server.bolt.advertised_address=localhost:7687
```

```bash
bin/neo4j-admin dbms set-initial-password compliance123
bin/neo4j start
```

**Kafka 3.7.0, KRaft single node:**

```bash
curl -O https://archive.apache.org/dist/kafka/3.7.0/kafka_2.13-3.7.0.tgz
tar xf kafka_2.13-3.7.0.tgz && cd kafka_2.13-3.7.0
```

In `config/kraft/server.properties`:

```
listeners=PLAINTEXT://:29092,CONTROLLER://:9093
advertised.listeners=PLAINTEXT://localhost:29092
log.dirs=/home/<you>/kafka-logs
auto.create.topics.enable=false
```

The last line matters: the gateway creates its three topics explicitly with
`partitions=1`, and silent auto-creation would give them the broker default instead,
changing the ordering guarantees the EDA condition depends on.

```bash
bin/kafka-storage.sh format -t $(bin/kafka-storage.sh random-uuid) -c config/kraft/server.properties
bin/kafka-server-start.sh -daemon config/kraft/server.properties
```

**Ollama** — the tarball bundles its own CUDA runtime, so no root and no systemd unit:

```bash
mkdir -p ~/ollama && curl -L https://ollama.com/download/ollama-linux-amd64.tgz | tar zx -C ~/ollama
export OLLAMA_MODELS=$HOME/ollama/models       # ~5 GB; check your home quota
CUDA_VISIBLE_DEVICES=1 OLLAMA_HOST=http://127.0.0.1:21434 ~/ollama/bin/ollama serve &
OLLAMA_HOST=http://127.0.0.1:21434 ~/ollama/bin/ollama pull llama3.1:8b-instruct-q4_K_M
```

`OLLAMA_HOST` does double duty — bind address for `ollama serve`, base URL for the Python
client in [config.py:34](inference-service/config.py) — and one `http://127.0.0.1:<port>`
value satisfies both. If your Ollama version rejects the URL form when serving, use the
bare `127.0.0.1:21434` for `serve` only.

**JMeter** (needed in §7):

```bash
curl -O https://archive.apache.org/dist/jmeter/binaries/apache-jmeter-5.6.3.tgz
tar xf apache-jmeter-5.6.3.tgz
```

---

## 4. Python environment and GPU pinning

```bash
conda activate cg
cd ~/ComplianceGateway/inference-service
pip install -r requirements.txt      # pulls torch, ~2.5 GB
```

Point the config at whatever ports survived, in `inference-service/.env`:

```
NEO4J_URI=bolt://localhost:27687
KAFKA_BOOTSTRAP=localhost:29092
OLLAMA_HOST=http://127.0.0.1:21434
```

**Set `CUDA_VISIBLE_DEVICES` on the Python processes too, not only on Ollama.** The
embedding model is constructed without a device argument in
[pipeline/embeddings.py](inference-service/pipeline/embeddings.py), so
sentence-transformers auto-selects `cuda:0` — on a shared box that is whichever card the
other tenants are already on. Export it once per shell:

```bash
export CUDA_VISIBLE_DEVICES=1
```

---

## 5. Rebuild the graph — and stop here if it does not match

```bash
cd ~/ComplianceGateway/inference-service
python -m ingestion.build_indexes
```

The log **must** report `345/345 chunks served from cache` and make no API calls. If it
starts calling the API, stop it: `artifacts/` did not transfer, and letting it continue
both re-spends the extraction budget and builds a different graph.

Then confirm the graph and the retrieval layer:

```cypher
MATCH (c:Chunk)  RETURN count(c);          // 345
MATCH (e:Entity) RETURN count(e);          // 1673
MATCH ()-[r:RELATES]->()      RETURN count(r);   // 3684
MATCH ()-[r:MENTIONED_IN]->() RETURN count(r);   // 4394
SHOW INDEXES;   // chunk_vec, entity_vec, edge_vec all ONLINE
```

```bash
python evaluation/ablation/compare_all_strategies.py
#   expect  vector_rag 0.537 / hybrid 0.537 / light_rag 0.196 / hippo_rag 0.171
```

This takes a couple of minutes and calls no SLM. **A mismatch means the graph is not the
one the reported retrieval numbers came from**, so nothing measured after it would be
comparable to the thesis. Fix it before spending GPU time.

---

## 6. E1/E2 — retrieval and reasoning quality

Long jobs on a shared server need to survive a dropped session, so run them under `tmux`.
One strategy per process: the active strategy is fixed when the process starts.

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

Expect roughly 1.5–2 hours for all five on an A100.

Output lands in the repo-root `results/` regardless of the working directory
([run_eval.py:30](inference-service/evaluation/run_eval.py)): a `.jsonl` appended after
every query, and a `.json` with the summary at the end. If a run dies, rerun the **same**
command with `--resume` appended — previously failed queries are retried, already-scored
ones are not.

`--judge` is one qwen-max call per query, 800 across all five strategies. It can be dropped
now and added later against the same `--run-id`.

Delete `results/hybrid-smoke5.json` if it came across; it predates the hybrid ranking fix.

---

## 7. E3 — load

Start only the services the current condition needs. **One condition at a time.** Unlike a
laptop, memory is not the constraint here — GPU contention is: the consumer and `sync_api`
drive the same card, and a measurement taken with both up describes their interference
rather than either integration mode.

```bash
# gateway (all three conditions)
cd ~/ComplianceGateway/gateway-service
SERVER_PORT=28080 \
SPRING_KAFKA_BOOTSTRAP_SERVERS=localhost:29092 \
GATEWAY_INFERENCE_SYNC_URL=http://localhost:28000/infer \
mvn spring-boot:run

# then EITHER, for the EDA condition:
cd ~/ComplianceGateway/inference-service && python consumer_main.py
# OR, for both synchronous conditions:
uvicorn sync_api:app --port 28000
```

Rehearse the JMeter plan against the stub first — fixed delay instead of a pipeline, so
extractor and timeout mistakes surface in seconds rather than after an hour of GPU time:

```bash
python loadtest/stub_inference.py --delay 3 --port 28000
python loadtest/stub_inference.py --delay 250 --port 28000   # trips the 240 s acquire timeout
```

Then measure one real request and size the concurrency levels from it — EDA drain time is
roughly `threads × single-request time`, and that sets the wall-clock cost of the whole
experiment:

```bash
~/apache-jmeter-5.6.3/bin/jmeter -n -t ~/ComplianceGateway/loadtest/compliance_gateway.jmx \
  -JHOST=localhost -JPORT=28080 -JTHREADS=10 -JRAMP=10 -JDURATION=120 \
  -l ~/ComplianceGateway/results/eda-c10.jtl
```

Enable the matching Thread Group in the GUI and disable the others to switch conditions.
Restart the consumer between conditions so no residual backlog contaminates the next
queue-depth curve. Full profile and failure-mode taxonomy: [loadtest/README.md](loadtest/README.md).

**Record `nvidia-smi` before and after every level.** A neighbour landing on your card
mid-level invalidates that level's latency figures — re-run it. Disclose the shared-hardware
caveat in the write-up regardless; E1 and E2 are unaffected, since retrieval and
faithfulness scores do not depend on timing.

---

## 8. Watching the dashboard from your laptop

The server is headless, but nothing needs to change to see the live monitor. `app.js`
targets `http://localhost:8080/api/v1` ([dashboard/app.js:5](dashboard/app.js)) and the
gateway allows any origin
([WebConfig.java](gateway-service/src/main/java/edu/compliance/gateway/config/WebConfig.java)),
so an SSH tunnel is the whole mechanism: the browser runs on your laptop, and `localhost`
resolves through the tunnel to the server.

From the **laptop**:

```bash
ssh -L 8080:localhost:28080 -L 7474:localhost:27474 -L 7687:localhost:27687 user@server
```

The left-hand number is the local port and must be **8080 exactly** — `app.js` hardcodes
it, so forwarding to 18080 silently breaks the page. Make sure nothing local is already
using 8080 (a gateway left running from earlier testing is the usual culprit).

Then, in a second local terminal, serve the dashboard from your own clone:

```bash
cd dashboard && python -m http.server 5173
# open http://localhost:5173
```

The badge reads `connected` once the gateway is up. Queue depth, per-stage latencies and
recent audits are live — this is the screenshot for the thesis, taken while E3 runs.

**Neo4j Browser** at `http://localhost:7474` works through the same tunnel, but needs
7687 forwarded as well: the browser page opens its own bolt connection directly. If the
connect dialog fails, type the bolt URL manually as `bolt://localhost:7687` rather than
accepting the server's advertised address.

---

## 9. Getting results back to GitHub

`results/` is gitignored, so committing anything from it needs `-f`:

```bash
git add -f results/*.json results/*.jsonl
```

Raw `.jtl` files from JMeter can reach tens of megabytes at C=100. Gzip them, or commit
only the aggregated CSVs — GitHub warns above 50 MB per file and rejects above 100 MB.

```bash
gzip results/*.jtl && git add -f results/*.jtl.gz
```

Two routes, pick one.

### Route A — pull to the laptop, push from there (no credentials on the shared box)

```bash
# from the laptop
scp -r user@server:~/ComplianceGateway/results/ ./results/
git add -f results/ && git commit -m "E1/E2/E3 results from the GPU server" && git push
```

Nothing that can authenticate to GitHub ever exists on a machine other people use. This is
the safer default, at the cost of one extra copy.

### Route B — a deploy key on the server

Scoped to this one repository and revocable from the repo settings at any time.

```bash
# on the server
ssh-keygen -t ed25519 -C "server-compliancegateway" -f ~/.ssh/id_ed25519_cg -N ""
cat ~/.ssh/id_ed25519_cg.pub
```

Add that public key at **GitHub → the repository → Settings → Deploy keys → Add deploy
key**, and tick **Allow write access**. Then:

```bash
cat >> ~/.ssh/config <<'EOF'
Host github-cg
  HostName github.com
  User git
  IdentityFile ~/.ssh/id_ed25519_cg
  IdentitiesOnly yes
EOF
chmod 600 ~/.ssh/config

git remote set-url origin git@github-cg:ianjunlai/ComplianceGateway.git
git add -f results/ && git commit -m "E1/E2/E3 results from the GPU server" && git push
```

Delete the deploy key from the repository settings when the experiments are finished.

Do not use a password or a personal access token stored in a credential helper here: both
write a reusable secret in plaintext on a machine you do not control.

---

## 10. When something goes wrong

| Symptom | Cause | Fix |
|---|---|---|
| Build log shows API calls instead of `345/345 chunks served from cache` | `artifacts/` did not transfer | Stop immediately. Re-`scp` it and rebuild — otherwise you pay ~430k tokens and get a different graph |
| `compare_all_strategies.py` prints numbers other than 0.537 / 0.537 / 0.196 / 0.171 | The graph differs from the measured one | Do not proceed. Check the chunk/entity/relationship counts in §5 first |
| Producer connects, then `_ALL_BROKERS_DOWN` or a timeout | Kafka advertised listener still names the old port | `KAFKA_ADVERTISED_LISTENERS` / `advertised.listeners` must name the port clients actually dial (§3) |
| Out-of-memory on the GPU, or inference far slower than ~6 s | `cuda:0` landed on the busy card | `export CUDA_VISIBLE_DEVICES=<idle device>` in **every** shell, Python included, not just Ollama's |
| Dashboard badge stuck on `gateway unreachable` | Tunnel is not on local port 8080, or something local already holds it | `ssh -L 8080:...`; `app.js` hardcodes 8080 |
| Neo4j Browser loads but will not connect | 7687 not forwarded, or a wrong advertised bolt address | Forward 7687 as well; type `bolt://localhost:7687` manually in the connect dialog |
| `permission denied ... docker daemon socket` | Docker is present but you are not in the `docker` group | Use §3a or §3b; the group is not going to be granted |
| Long run dies when the SSH session drops | Not running under a multiplexer | `tmux new -s eval`; restart with the same `--run-id` plus `--resume` |
