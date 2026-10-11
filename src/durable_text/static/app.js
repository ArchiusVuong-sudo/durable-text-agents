"use strict";

const CHIPS = [
  "hi, my dishwasher is leaking",
  "water all over the kitchen floor",
  "tomorrow morning works",
  "actually afternoon is better",
  "any update on my repair?",
];
const KEYWORDS = ["STOP", "START", "HELP"];
const $ = (id) => document.getElementById(id);
let state = null;
let lastMsgCount = -1;
let fetching = false;
let again = false;
let lastJson = "";
// Items already on screen: only unseen ones get the "new" class (and its fade-in),
// so a re-render does not replay the animation on every bubble and card.
const seen = { msgs: new Set(), turns: new Set() };
let firstRender = true;
function isNew(set, key) {
  if (set.has(key)) return false;
  set.add(key);
  return !firstRender;
}

function esc(v) {
  return String(v ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}
function plural(n, word) { return `${n} ${word}${n === 1 ? "" : "s"}`; }
function secs(ms) { return ms == null ? "" : (ms / 1000).toFixed(1) + " s"; }
function toast(text, isErr) {
  const t = $("toast");
  t.textContent = text;
  t.className = "toast" + (isErr ? " err" : "");
  clearTimeout(toast._t);
  toast._t = setTimeout(() => t.classList.add("hidden"), 2600);
}

async function api(path, body) {
  const res = await fetch(path, {
    method: "POST",
    headers: body ? { "Content-Type": "application/json" } : {},
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail || detail; } catch (_) { /* not json */ }
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return res.json();
}

async function sendText(body) {
  const text = body.trim();
  if (!text || !state || !state.thread) return;
  const form = new URLSearchParams({
    From: state.thread.customer.phone,
    To: state.thread.business_number,
    Body: text,
    MessageSid: "msg_" + crypto.randomUUID().replace(/-/g, "").slice(0, 24),
  });
  try {
    const res = await fetch("/sms/inbound", { method: "POST", body: form });
    if (!res.ok) throw new Error("webhook returned " + res.status);
  } catch (e) {
    toast("Could not send: " + e.message, true);
  }
  refresh();
}

async function action(fn, okText) {
  try {
    await fn();
    if (okText) toast(okText);
  } catch (e) {
    toast(e.message, true);
  }
  refresh();
}

async function refresh() {
  if (fetching) { again = true; return; }
  fetching = true;
  try {
    const res = await fetch("/api/state");
    if (!res.ok) throw new Error(res.statusText);
    const text = await res.text();
    if (text === lastJson) return; // nothing changed since the last render
    lastJson = text;
    state = JSON.parse(text);
    render(state);
    firstRender = false;
  } catch (e) {
    toast("Server unreachable, retrying", true);
  } finally {
    fetching = false;
    if (again) { again = false; refresh(); }
  }
}

/* ---------- render ---------- */

function render(s) {
  liveWorkers = new Set(s.workers.filter((w) => w.status === "running").map((w) => w.pid));
  const mode = $("mode");
  if (s.mode === "live") {
    mode.className = "mode live";
    mode.textContent = "Live · " + s.model;
  } else {
    mode.className = "mode replay";
    mode.textContent = "Replay mode";
  }
  mode.title = s.mode_reason + (s.fixtures != null ? ` · ${s.fixtures} recorded LLM calls` : "");
  const t = s.totals;
  $("totals").textContent = s.mode === "live"
    ? `${plural(t.turns, "committed turn")} · ${plural(t.llm_calls, "LLM call")} · $${t.cost_usd.toFixed(3)} est.`
    : `${plural(t.turns, "committed turn")} · ${plural(t.llm_calls, "LLM call")} replayed`;
  renderPhone(s);
  renderTimeline(s);
  renderWorker(s);
  renderClock(s);
  renderApprovals(s);
  renderSession(s);
  renderTimers(s);
  const w = s.webhooks[0];
  $("lastwebhook").textContent = w ? `${w.sid} · ${w.outcome}` : "no webhooks yet";
  $("replay").disabled = !w;
}

function renderPhone(s) {
  const th = s.thread;
  $("optout").classList.toggle("hidden", !(th && th.opted_out));
  const box = $("thread");
  if (!s.messages.length) {
    box.innerHTML = `<div class="empty-thread">No messages yet.<br>Tap a suggested text below to start as Maya Chen.</div>`;
  } else {
    box.innerHTML = s.messages.map(bubble).join("") +
      (s.typing && !(th && th.opted_out) ? `<div class="typing" aria-label="agent is typing"><i></i><i></i><i></i></div>` : "");
  }
  if (s.messages.length !== lastMsgCount || s.typing) box.scrollTop = box.scrollHeight;
  lastMsgCount = s.messages.length;
}

function bubble(m) {
  const tags = [];
  if (m.direction === "out") {
    tags.push(`<span>${esc(m.label)}</span>`);
    if (m.kind === "template") tags.push(`<span class="tag indigo">no LLM call</span>`);
    if (m.status === "delivered") tags.push(`<span>delivered</span>`);
    else if (m.status === "queued") tags.push(`<span>sending…</span>`);
    else if (m.status === "suppressed") tags.push(`<span class="tag red">not sent: opted out</span>`);
  } else {
    if (m.kind === "keyword") tags.push(`<span class="tag indigo">keyword</span>`);
    if (m.kind === "ignored") tags.push(`<span class="tag red">opted out, no reply</span>`);
  }
  const fresh = isNew(seen.msgs, `${m.id}:${m.sid}`) ? " new" : "";
  return `<div class="bubble-wrap ${m.direction} ${m.status}${fresh}">
    <div class="bubble">${esc(m.body)}</div>
    <div class="meta">${tags.join("<span>·</span>")}</div></div>`;
}

function renderChips() {
  $("chips").innerHTML =
    CHIPS.map((c) => `<button class="chip" data-text="${esc(c)}">${esc(c)}</button>`).join("") +
    KEYWORDS.map((k) => `<button class="chip kw" data-text="${k}">${k}</button>`).join("");
}

function renderTimeline(s) {
  const items = [];
  for (const t of s.turns) items.push({ at: t.started_ms, html: turnCard(t) });
  for (const w of s.webhooks) {
    if (w.outcome === "accepted") continue;
    const text = {
      duplicate: `duplicate MessageSid ignored · ${esc(w.sid)}`,
      opted_out: `text from an opted-out number, no reply`,
      unknown_number: `text from an unknown number ignored`,
    }[w.outcome] || `${esc(w.outcome.replace("keyword:", "").toUpperCase())} keyword answered from a template · no LLM call`;
    items.push({ at: w.created_ms, html: `<div class="channel-row"><span class="tag amber">channel</span><span>${text}</span><span class="line"></span></div>` });
  }
  items.sort((a, b) => b.at - a.at);
  $("timeline").innerHTML = items.length ? items.map((i) => i.html).join("") :
    `<div class="empty"><b>No turns yet</b>Send a text from the phone. Each burst of texts becomes one agent turn, rebuilt from the last checkpoint and committed in one transaction.</div>`;
}

const STATUS = {
  running: (t) => `<span class="pill indigo"><span class="spin"></span><span data-start="${t.started_ms}">running</span></span>`,
  committed: () => `<span class="pill green">committed</span>`,
  superseded: () => `<span class="pill amber">superseded, nothing sent</span>`,
  interrupted: () => `<span class="pill red">interrupted</span>`,
  failed: () => `<span class="pill red">failed</span>`,
};

let liveWorkers = new Set();

function turnCard(t) {
  const orphaned = t.status === "running" && !liveWorkers.has(t.pid);
  const title = t.kind === "turn" ? `Inbound turn #${t.id}` : t.kind === "wake" ? `Wake turn #${t.id}` : `Delegation #${t.id}`;
  const kindLabel = t.kind === "turn" ? "text" : t.kind === "wake" ? "wake" : "delegate";
  const dur = t.ended_ms ? secs(t.ended_ms - t.started_ms) : "";
  let sub;
  if (t.kind === "dispatch") {
    sub = `${esc(t.input.from)} → <b>${esc(t.agent_before)}</b> · headless AgentSession · worker pid <b>${t.pid}</b>`;
  } else {
    const cp = t.checkpoint_from ? `resumed from checkpoint <b>v${t.checkpoint_from}</b>` : `new session (checkpoint v0)`;
    sub = `${cp} · worker pid <b>${t.pid}</b> · agent <b>${esc(t.agent_before)}</b>`;
  }
  if (t.attempt > 1) sub += ` · attempt <b>${t.attempt}</b>`;
  const fresh = isNew(seen.turns, `${t.id}:${t.started_ms}`) ? " new" : "";
  return `<article class="card ${t.status}${fresh}">
    <div class="card-head">
      <span class="kind ${t.kind}">${kindLabel}</span>
      <span class="card-title">${title}</span>
      <div class="card-right">${orphaned ? `<span class="pill red">worker ${t.pid} gone</span>` : (STATUS[t.status] || STATUS.failed)(t)}${dur ? `<span>${dur}</span>` : ""}</div>
    </div>
    <div class="subline">${sub}</div>
    ${inputBlock(t)}
    <ul class="steps">${t.trace.map(step).join("") || emptySteps(t)}</ul>
    ${foot(t, orphaned)}
  </article>`;
}

function emptySteps(t) {
  const note = {
    running: "waiting for the model",
    superseded: "cancelled during its first model call",
    interrupted: "stopped during its first model call",
  }[t.status];
  return note ? `<li class="step"><span class="ic out">…</span><span class="muted">${note}</span></li>` : "";
}

function inputBlock(t) {
  if (t.kind === "turn") {
    const texts = t.input.texts || [];
    const label = texts.length > 1 ? `<b>${texts.length} texts coalesced</b> · ` : "";
    return `<div class="input">${label}${texts.map((x) => `<span class="q">“${esc(x)}”</span>`).join(" + ")}</div>`;
  }
  if (t.kind === "wake") {
    const ev = t.input.event || {};
    const d = ev.data || {};
    const extra = d.technician_name ? ` · ${esc(d.technician_name)}, ${esc(d.window)}` : "";
    return `<div class="input">no inbound text · event <span class="code">${esc(ev.type)}</span>${extra}</div>`;
  }
  const task = t.input.task || {};
  return `<div class="input">task <span class="code">${esc(t.input.delegation_id)}</span>: ${esc(task.skill)} technician, ${esc(task.zone)} zone, ${esc(task.window)}</div>`;
}

function args(a) {
  if (a == null || typeof a !== "object") return esc(a);
  return Object.entries(a).map(([k, v]) => `${esc(k)}: ${esc(JSON.stringify(v))}`).join(", ");
}

function step(x) {
  switch (x.t) {
    case "call": return `<li class="step"><span class="ic call">ƒ</span><span class="code">${esc(x.name)}(${args(x.args)})</span></li>`;
    case "output": return x.error
      ? `<li class="step"><span class="ic err">!</span><span class="err-text">${esc(x.name)}: ${esc(x.output)}</span></li>`
      : x.output ? `<li class="step"><span class="ic out">↩</span><span class="out-text" title="${esc(x.output)}">${esc(x.output)}</span></li>` : "";
    case "handoff": return `<li class="step"><span class="ic hand">⇄</span><span>AgentHandoff <b>${esc(x.from)}</b> → <b>${esc(x.to)}</b></span></li>`;
    case "message": return `<li class="step"><span class="ic msg">T</span><span>“${esc(x.text)}”</span></li>`;
    case "hold": return `<li class="step"><span class="ic eff">+</span><span>hold slot <span class="code">${esc(x.slot)}</span></span></li>`;
    case "delegation": return `<li class="step"><span class="ic eff">+</span><span>delegate to <b>${esc(x.to)}</b> · <span class="code">${esc(x.id)}</span>, its own AgentSession</span></li>`;
    case "timer": return `<li class="step"><span class="ic eff">+</span><span>schedule timer <span class="code">${esc(x.kind)}</span> · ${esc(x.due.replace("T", " "))}</span></li>`;
    case "proposal": return `<li class="step"><span class="ic eff">+</span><span>proposal: <b>${esc(x.technician_name)}</b>, ${esc(x.window)} · waits for a dispatcher</span></li>`;
    default: return "";
  }
}

function foot(t, orphaned) {
  const calls = t.llm.map((c) => {
    const cls = c.source === "live" ? "live" : c.source === "replay" ? "replay" : c.source === "replay-fallback" ? "fallback" : "";
    const ms = c.source === "replay" ? c.recorded_ms : c.duration_ms;
    const tok = c.input_tokens ? ` · ${c.input_tokens}+${c.output_tokens} tok` : "";
    return `<span class="call-chip ${cls}" title="${esc(c.model)}">${esc(c.source)} ${secs(ms)}${tok}</span>`;
  }).join("");
  let left = "";
  const texts = t.trace.filter((x) => x.t === "message").length;
  if (t.status === "committed") {
    left = t.kind === "dispatch" ? "result committed · waits for approval" :
      `checkpoint <b>v${t.checkpoint_to}</b> committed · ${texts} ${texts === 1 ? "text" : "texts"} to the outbox`;
  } else if (t.status === "superseded") left = "a newer text arrived · rerun with every unread text";
  else if (t.status === "interrupted") left = esc(t.error || "worker stopped") + " · the job was run again";
  else if (orphaned) left = "its worker was killed · the job lease is reclaimed by the next worker";
  else if (t.status === "failed") left = `<span class="err-text">${esc(t.error)}</span>`;
  else left = "uncommitted · nothing is sent until the turn commits";
  const cost = t.cost_usd && t.llm.some((c) => c.source === "live") ? ` · $${t.cost_usd.toFixed(4)}` : "";
  return `<div class="card-foot"><span>${left}</span><span class="calls">${calls}${cost ? `<span class="call-chip">${cost.slice(3)}</span>` : ""}</span></div>`;
}

function renderWorker(s) {
  const cur = s.workers.find((w) => w.status === "running" && w.pid === s.worker_pid);
  const rows = s.workers.slice(0, 4).map((w) => {
    const on = w.status === "running";
    const label = on ? "running" : w.status;
    return `<div class="pidrow"><span class="dot ${on ? "on" : w.status === "killed" ? "dead" : ""}"></span>
      <span class="mono">pid ${w.pid}</span><span class="muted">${esc(label)}</span></div>`;
  }).join("");
  const jobs = s.jobs.filter((j) => j.status === "running").map((j) =>
    `<div class="pidrow"><span class="tag">${esc(j.kind)}</span><span class="muted small">job #${j.id} · attempt ${j.attempts} · lease pid ${j.lease_owner ?? "-"}${j.cancel_requested ? " · cancel requested" : ""}</span></div>`).join("");
  $("worker").innerHTML = (rows || `<div class="none">starting…</div>`) +
    `<div class="label" style="margin-top:8px">Jobs in flight</div>` + (jobs || `<div class="none">idle</div>`);
  $("kill").disabled = !cur;
}

function renderClock(s) {
  const next = s.timers.find((t) => t.status === "pending");
  $("clock").innerHTML = `<span class="big">${esc(s.clock.human)}</span>
    <span class="muted small">${next ? `next timer ${esc(next.due_human)}` : "no pending timers"}</span>`;
  $("next").disabled = !next;
}

function renderApprovals(s) {
  if (!s.delegations.length) {
    $("approvals").innerHTML = `<div class="none">No delegations yet. The scheduler delegates the technician pick to a headless dispatch agent.</div>`;
    return;
  }
  $("approvals").innerHTML = s.delegations.map((d) => {
    const head = `<div class="proposal-head"><span><span class="code">${esc(d.id)}</span> · ${esc(d.from)} → ${esc(d.to)}</span>${dpill(d)}</div>`;
    if (d.status === "needs_approval" || d.status === "approved") {
      const r = d.result || {};
      return `<div class="proposal ${d.status === "needs_approval" ? "waiting" : ""}">${head}
        <div class="who" style="margin-top:6px">${esc(r.technician_name)} · ${esc(r.window)}</div>
        <div class="why">${esc(r.reason)}</div>
        ${d.status === "needs_approval" ? `<button class="btn primary small" data-approve="${esc(d.id)}">Approve</button>` : ""}</div>`;
    }
    const msg = d.status === "failed" ? "dispatch could not find a technician" :
      `dispatch agent is picking a technician${d.attempts > 1 ? ` · attempt ${d.attempts}` : ""}`;
    return `<div class="proposal">${head}<div class="why" style="margin:6px 0 0">${msg}</div></div>`;
  }).join("");
}

function dpill(d) {
  return {
    queued: `<span class="pill gray">queued</span>`,
    running: `<span class="pill indigo"><span class="spin"></span>running</span>`,
    needs_approval: `<span class="pill amber">needs approval</span>`,
    approved: `<span class="pill green">approved</span>`,
    failed: `<span class="pill red">failed</span>`,
  }[d.status] || `<span class="pill gray">${esc(d.status)}</span>`;
}

function renderSession(s) {
  const th = s.thread;
  const cp = s.checkpoint;
  const v = (cp && cp.values) || {};
  const rows = [
    ["Active agent", th ? `<span class="code">${esc(th.active_agent)}</span>` : "-"],
    ["Checkpoint", cp ? `v${cp.version} · ${cp.items} chat items` : "v0 · empty"],
    ["SMS consent", th && th.opted_out ? `<span class="tag red">opted_out</span>` : `<span class="tag green">subscribed</span>`],
    ["Issue", esc(v.issue || "-")],
    ["Held window", esc(v.held_window || "-")],
    ["Dispatch", esc(v.dispatch_status || "-") + (v.technician ? ` · ${esc(v.technician)}` : "")],
  ];
  $("session").innerHTML = `<dl class="kv">${rows.map(([k, val]) => `<dt>${k}</dt><dd>${val}</dd>`).join("")}</dl>`;
}

function renderTimers(s) {
  if (!s.timers.length) {
    $("timers").innerHTML = `<div class="none">No timers yet. An approved visit schedules a reminder and a follow-up on the simulated clock.</div>`;
    return;
  }
  const names = { visit_reminder: "Visit reminder", visit_followup: "Follow-up" };
  $("timers").innerHTML = s.timers.map((t) => {
    const pill = t.status === "pending" ? `<span class="pill gray">pending</span>` :
      t.status === "fired" ? `<span class="pill green">fired</span>` :
      `<span class="pill red">suppressed: ${esc(t.note || "")}</span>`;
    return `<div class="timer"><span><b>${esc(names[t.kind] || t.kind)}</b> <span class="muted">· ${esc(t.due_human)}</span></span>${pill}</div>`;
  }).join("");
}

/* ---------- wiring ---------- */

renderChips();
$("chips").addEventListener("click", (e) => {
  const b = e.target.closest("[data-text]");
  if (b) sendText(b.dataset.text);
});
$("composer").addEventListener("submit", (e) => {
  e.preventDefault();
  const input = $("body");
  sendText(input.value);
  input.value = "";
});
$("approvals").addEventListener("click", (e) => {
  const b = e.target.closest("[data-approve]");
  if (b) { b.disabled = true; action(() => api(`/api/delegations/${b.dataset.approve}/approve`), "Approved, the thread wakes up"); }
});
$("kill").addEventListener("click", () => action(async () => {
  const r = await api("/api/worker/kill");
  toast(`Sent SIGKILL to worker pid ${r.killed}`);
}));
$("adv").addEventListener("click", () => action(() => api("/api/clock/advance", { hours: 3 })));
$("next").addEventListener("click", () => action(() => api("/api/clock/next-timer")));
$("replay").addEventListener("click", () => action(async () => {
  const r = await api("/api/dev/replay-webhook");
  toast(r.outcome === "duplicate" ? "Carrier retry: duplicate MessageSid ignored" : r.outcome);
}));
$("reset").addEventListener("click", () => action(() => api("/api/reset"), "Demo reset"));

setInterval(() => {
  for (const el of document.querySelectorAll("[data-start]")) {
    el.textContent = "running " + secs(Date.now() - Number(el.dataset.start));
  }
}, 250);

const es = new EventSource("/api/stream");
es.addEventListener("change", refresh);
es.addEventListener("hello", refresh);
setInterval(refresh, 4000);
refresh();
