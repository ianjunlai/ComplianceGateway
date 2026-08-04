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

```bash
mkdir -p ~/ollama
curl -L https://ollama.com/download/ollama-linux-amd64.tgz | tar zx -C ~/ollama
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

In `config/kraft/server.properties` set `log.dirs` to a path in your home, and:

```
auto.create.topics.enable=false
```

That line matters: the gateway creates its three topics explicitly with `partitions=1`, and
silent auto-creation would give them the broker default instead, changing the ordering
guarantees the EDA condition depends on.

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
  -JTHREADS=10 -JRAMP=10 -JDURATION=120 \
  -l ~/ComplianceGateway/results/eda-c10.jtl
```

One condition per run; enable the matching Thread Group and disable the others. Restart the
consumer between conditions so no residual backlog contaminates the next queue-depth curve.
Record `nvidia-smi` before and after each level — a neighbour landing on your card
invalidates that level's latency figures. Full profile and failure-mode taxonomy:
[loadtest/README.md](loadtest/README.md).

## 11. Watch the dashboard from your laptop

Nothing needs changing. `app.js` targets `http://localhost:8080/api/v1`
([dashboard/app.js:5](dashboard/app.js)) and the gateway allows any origin, so the browser
runs on your laptop and `localhost` resolves through an SSH tunnel.

```bash
# on the laptop
ssh -L 8080:localhost:8080 -L 7474:localhost:7474 -L 7687:localhost:7687 user@server

# in a second local terminal
cd dashboard && python -m http.server 5173
# open http://localhost:5173
```

The local port must be **8080 exactly** — `app.js` hardcodes it, so `-L 18080:...` silently
breaks the page. Make sure nothing local is already holding 8080.

Neo4j Browser at `http://localhost:7474` works through the same tunnel, but only if 7687 is
forwarded too: the browser page opens its own bolt connection directly.

---

# Appendix

## If a port is already taken

`ss -ltn | grep -E '7474|7687|9092|8080|8000|11434'` shows what is occupied. Everything is
configurable without touching code — `NEO4J_URI`, `KAFKA_BOOTSTRAP` and `OLLAMA_HOST` in
`.env`; `SERVER_PORT`, `SPRING_KAFKA_BOOTSTRAP_SERVERS` and `GATEWAY_INFERENCE_SYNC_URL` as
environment variables for the gateway; `server.bolt.listen_address` and
`server.http.listen_address` in `conf/neo4j.conf`.

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
