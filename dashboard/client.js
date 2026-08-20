/* Compliance Audit Portal — the user-facing client.
 *
 * Submits to the event-driven endpoint and polls for the result, which is what
 * a real caller does: POST /audit returns 202 with a request id, and the
 * decision arrives once the single consumer reaches it. */

const DEFAULT_GATEWAY = "http://localhost:8080/api/v1";
const POLL_MS = 500;
const TIMEOUT_MS = 240000;          // matches the gateway's own limit
const HISTORY_KEY = "cg-client-history";

// Examples are per institution: a request from Limerick must be answerable
// from Irish and EU law, so a Cambridge question would be checked against a
// body of law that does not bind it.
const LABELS = { cross_tier: "Cross-tier", single: "Single-tier",
                 unanswerable: "Not covered" };
let EXAMPLES = {};

// Which law binds which institution — the same mapping the retrieval layer uses.
const SCOPE = {
  cambridge: "EU + UK law",
  tcd: "EU + Irish law",
  ul: "EU + Irish law",
  goettingen: "EU + German law",
};

const $ = id => document.getElementById(id);
let clauses = {};

/* ---------- gateway address, remembered per browser ---------------------- */

function gateway() {
  return $("gateway").value.trim().replace(/\/+$/, "") || DEFAULT_GATEWAY;
}

$("gateway").value = localStorage.getItem("cg-gateway") || DEFAULT_GATEWAY;
$("gateway").addEventListener("change", () =>
  localStorage.setItem("cg-gateway", gateway()));

/* ---------- identity ------------------------------------------------------ */

function showScope() {
  $("scope-hint").textContent = "requests are checked against " + SCOPE[$("source-system").value];
}
$("source-system").addEventListener("change", () => { showScope(); showExamples(); });
showScope();

/* ---------- examples ------------------------------------------------------ */

function showExamples() {
  const set = EXAMPLES[$("source-system").value] || [];
  document.querySelectorAll(".example").forEach(b => {
    const ex = set[+b.dataset.ex];
    b.hidden = !ex;
    if (ex) { b.textContent = LABELS[ex.kind] || ex.kind; b.dataset.text = ex.text; }
  });
}

document.querySelectorAll(".example").forEach(b =>
  b.addEventListener("click", () => { $("query").value = b.dataset.text || ""; }));

/* ---------- submit and poll ---------------------------------------------- */

$("submit").addEventListener("click", submit);

async function submit() {
  const query = $("query").value.trim();
  if (!query) { setStatus("Enter a request first.", "warn"); return; }

  $("submit").disabled = true;
  $("result-card").hidden = true;
  const started = Date.now();

  try {
    setStatus("Submitting…", "busy");
    const res = await fetch(`${gateway()}/audit`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        source_system: $("source-system").value,
        audit_query: query,
      }),
    });
    if (!res.ok) throw new Error(`gateway returned ${res.status}`);
    const { request_id } = await res.json();

    setStatus(`Queued as ${request_id.slice(0, 8)} — waiting for the auditor…`, "busy");
    const result = await poll(request_id, started);

    render(result);
    remember(query, result, Date.now() - started);
    setStatus(`Answered in ${((Date.now() - started) / 1000).toFixed(1)} s`, "ok");
  } catch (e) {
    setStatus(e.message, "err");
  } finally {
    $("submit").disabled = false;
  }
}

async function poll(id, started) {
  while (Date.now() - started < TIMEOUT_MS) {
    await new Promise(r => setTimeout(r, POLL_MS));
    const res = await fetch(`${gateway()}/audit/${id}`);
    if (!res.ok) continue;
    const body = await res.json();
    if (body.status !== "PENDING") return body;
    setStatus(`Queued as ${id.slice(0, 8)} — waiting ${
      Math.round((Date.now() - started) / 1000)} s`, "busy");
  }
  throw new Error("No answer within 240 s — the queue may be saturated.");
}

/* ---------- rendering ----------------------------------------------------- */

function render(r) {
  $("result-card").hidden = false;

  const d = (r.decision || "ERROR").toUpperCase();
  $("decision").textContent = d;
  $("decision").className = "decision-badge " + d.toLowerCase();

  // Clause ids appear in the reasoning as [uk-dpa-s8]; make them stand out.
  $("reasoning").innerHTML = escapeHtml(r.reasoning || r.error || "")
    .replace(/\[([a-z0-9][\w.\-]*)\]/gi, '<span class="cite">[$1]</span>');

  renderChunks(r.retrieved_chunk_ids || []);
}

function renderChunks(ids) {
  const box = $("chunks");
  $("clause-text").hidden = true;
  if (!ids.length) { box.innerHTML = '<span class="hint">None retrieved.</span>'; return; }
  box.innerHTML = ids.map(id =>
    `<button class="chip" data-id="${escapeHtml(id)}">${escapeHtml(id)}</button>`).join("");
  box.querySelectorAll(".chip").forEach(c =>
    c.addEventListener("click", () => showClause(c.dataset.id)));
}

function showClause(id) {
  const c = clauses[id];
  const pre = $("clause-text");
  pre.hidden = false;
  pre.textContent = c
    ? `${id}   [${c.r} · ${c.j}]\n\n${c.t}`
    : `${id}\n\n(text not available in this build)`;
}

/* ---------- local history (this browser only) ----------------------------- */

function remember(query, r, wallMs) {
  const hist = load();
  hist.unshift({
    at: new Date().toISOString(),
    query,
    decision: r.decision || "ERROR",
    ms: wallMs,
  });
  localStorage.setItem(HISTORY_KEY, JSON.stringify(hist.slice(0, 20)));
  renderHistory();
}

const load = () => {
  try { return JSON.parse(localStorage.getItem(HISTORY_KEY)) || []; }
  catch { return []; }
};

function renderHistory() {
  const hist = load();
  $("history-empty").hidden = hist.length > 0;
  document.querySelector("#history tbody").innerHTML = hist.map(h => `
    <tr>
      <td class="mono">${h.at.slice(11, 19)}</td>
      <td class="q">${escapeHtml(h.query)}</td>
      <td><span class="decision ${h.decision.toLowerCase()}">${h.decision}</span></td>
      <td class="mono">${(h.ms / 1000).toFixed(1)} s</td>
    </tr>`).join("");
}

/* ---------- helpers ------------------------------------------------------- */

function setStatus(msg, kind) {
  const s = $("status");
  s.textContent = msg;
  s.className = "status " + (kind || "");
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, c =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

fetch("clauses.json")
  .then(r => r.json())
  .then(j => { clauses = j; })
  .catch(() => { /* chips still work, just without the text */ });

fetch("examples.json")
  .then(r => r.json())
  .then(j => { EXAMPLES = j; showExamples(); })
  .catch(() => { document.querySelectorAll(".example").forEach(b => b.hidden = true); });

renderHistory();
