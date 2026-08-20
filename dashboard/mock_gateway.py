"""Mock gateway for exercising the client page without the full stack.

Replays a real response recorded from the hybrid run, so the page is tested
against the exact shape the gateway produces.

    python dashboard/mock_gateway.py 8080
"""
import json, re, sys, threading, time, uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QA = {q["query_id"]: q for q in json.loads(
    (ROOT / "dataset" / "qa_v2.json").read_text(encoding="utf-8"))}
ROWS = {r["query_id"]: r for r in json.loads(
    (ROOT / "results" / "hybrid-full0812.json").read_text(encoding="utf-8"))["rows"]}

PENDING, DONE = {}, {}
DELAY = 3.0
COUNT = {"submitted": 0, "completed": 0, "errors": 0}
RECENT = []


def match(query, source):
    """Find the recorded run for this question.

    The institution matters: a request from Limerick is answered from Irish and
    EU law only, so replaying a Cambridge run would show UK provisions that the
    real system would never have retrieved for that caller.
    """
    q = query.strip().lower()[:60]
    for qid, item in QA.items():
        if item["query_text"].strip().lower()[:60] == q:
            return qid
    same = [qid for qid, item in QA.items() if item["source_system"] == source]
    return same[0] if same else next(iter(QA))


ALLOWED = {"cambridge": {"EU", "UK"}, "tcd": {"EU", "IE"},
           "ul": {"EU", "IE"}, "goettingen": {"EU", "DE"}}
JURIS = {c["chunk_id"]: c.get("jurisdiction") for c in json.loads(
    (ROOT / "dataset" / "corpus" / "full_corpus.json").read_text(encoding="utf-8"))}


def in_scope(ids, source):
    """The live system filters candidates by jurisdiction before ranking; a
    replay has to do the same or it misrepresents what the caller would see."""
    allow = ALLOWED.get(source)
    return [i for i in ids if allow is None or JURIS.get(i) in allow]


def finish(rid, qid, source):
    time.sleep(DELAY)
    r = ROWS[qid]
    DONE[rid] = {
        "request_id": rid, "source_system": source,
        "decision": r["prediction"], "reasoning": r["reasoning"],
        "retrieved_chunk_ids": in_scope(r.get("retrieved_chunk_ids", []), source),
        "strategy": "hybrid", "stage_timings_ms": r.get("stage_timings_ms", {}),
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    PENDING.pop(rid, None)
    COUNT["completed"] += 1
    RECENT.insert(0, dict(DONE[rid], e2e_ms=int(DELAY * 1000)))
    del RECENT[20:]


class H(BaseHTTPRequestHandler):
    def _send(self, code, body):
        raw = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_OPTIONS(self):
        self._send(204, {})

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n) or b"{}")
        rid = str(uuid.uuid4())
        PENDING[rid] = True
        COUNT["submitted"] += 1
        source = body.get("source_system", "unknown")
        threading.Thread(target=finish, args=(
            rid, match(body.get("audit_query", ""), source), source),
            daemon=True).start()
        self._send(202, {"request_id": rid, "status": "QUEUED"})

    def do_GET(self):
        m = re.match(r"/api/v1/audit/([\w-]+)", self.path)
        if m:
            rid = m.group(1)
            return self._send(200, DONE.get(rid, {"request_id": rid, "status": "PENDING"}))
        if self.path.startswith("/api/v1/metrics"):
            return self._send(200, dict(COUNT, queue_depth=len(PENDING), recent=RECENT))
        self._send(404, {"error": "not found"})

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
    print(f"mock gateway on http://localhost:{port}/api/v1  (delay {DELAY}s)")
    ThreadingHTTPServer(("0.0.0.0", port), H).serve_forever()
