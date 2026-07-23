/* ComplianceGateway live monitor.
 * Polls the gateway metrics endpoint and renders queue depth, counters and
 * per-stage latencies of recent audits. No build step, no dependencies. */

const GATEWAY = "http://localhost:8080/api/v1";
const POLL_MS = 2000;
const HISTORY_POINTS = 150; // 5 min at 2 s

const history = [];

async function poll() {
  const status = document.getElementById("conn-status");
  try {
    const res = await fetch(`${GATEWAY}/metrics`);
    const m = await res.json();
    status.textContent = "connected";
    status.className = "badge online";
    render(m);
  } catch (e) {
    status.textContent = "gateway unreachable";
    status.className = "badge offline";
  }
}

function render(m) {
  setText("queue-depth", m.queueDepth);
  setText("submitted", m.submitted);
  setText("completed", m.completed);
  setText("errors", m.errors);

  const e2es = (m.recent || []).map(r => r.e2eMs).filter(v => v != null);
  setText("avg-e2e", e2es.length
    ? `${Math.round(e2es.reduce((a, b) => a + b, 0) / e2es.length)} ms` : "–");

  history.push(m.queueDepth);
  if (history.length > HISTORY_POINTS) history.shift();
  renderChart();
  renderTable(m.recent || []);
}

function renderChart() {
  const chart = document.getElementById("queue-chart");
  const max = Math.max(1, ...history);
  chart.innerHTML = history
    .map(v => `<div class="bar" style="height:${(v / max) * 100}%" title="${v}"></div>`)
    .join("");
}

function renderTable(recent) {
  const tbody = document.querySelector("#recent-table tbody");
  tbody.innerHTML = recent.slice(0, 20).map(r => {
    const t = r.stageTimingsMs || {};
    return `<tr>
      <td class="mono">${short(r.requestId)}</td>
      <td>${r.sourceSystem ?? ""}</td>
      <td><span class="decision ${(r.decision || "").toLowerCase()}">${r.decision ?? ""}</span></td>
      <td>${r.strategy ?? ""}</td>
      <td>${ms(t.queue_wait_ms)}</td>
      <td>${ms(t.ner_ms)}</td>
      <td>${ms(t.retrieval_ms)}</td>
      <td>${ms(t.generation_ms)}</td>
      <td>${ms(r.e2eMs)}</td>
    </tr>`;
  }).join("");
}

const setText = (id, v) => { document.getElementById(id).textContent = v ?? "–"; };
const short = id => (id || "").slice(0, 8);
const ms = v => (v == null ? "–" : `${v} ms`);

poll();
setInterval(poll, POLL_MS);
