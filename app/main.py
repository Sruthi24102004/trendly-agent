"""
FastAPI surface.

    POST /chat       the agent
    GET  /health     liveness, plus which provider/model is live
    GET  /metrics    aggregated deflection, escalation and guardrail numbers
    GET  /dashboard  those numbers, rendered
    GET  /           chat UI with a live view of the graph's state

Both pages are single-viewport: the shell is fixed at 100dvh and only the
inner panes scroll, so nothing important sits below the fold during a demo.
"""

import os

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from app.agent import MODEL, PROVIDER, run_agent
from app.cassettes import mode as cassette_mode
from app.observability import summarize

app = FastAPI(title="Trendly Support Agent")


class ChatRequest(BaseModel):
    session_id: str
    message: str


class ChatResponse(BaseModel):
    reply: str
    tool_calls_made: list
    escalated: bool
    state: dict = {}
    diagnostics: dict = {}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "provider": PROVIDER,
        "model": MODEL,
        "cassette_mode": cassette_mode(),
        "clock_override": os.environ.get("TRENDLY_NOW"),
    }


@app.get("/metrics")
def metrics():
    """Aggregate view of every turn served. Consumed by /dashboard."""
    return summarize()


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="message must not be empty")
    try:
        return run_agent(session_id=req.session_id, user_message=req.message)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/", response_class=HTMLResponse)
def index():
    return CHAT_PAGE


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard():
    return DASHBOARD_PAGE


BASE_CSS = """
  :root {
    --bg:#0a0c10; --panel:#121620; --panel-2:#161b26; --line:#242b39;
    --text:#e8eaef; --muted:#8892a4; --dim:#5d6779;
    --accent:#e0a63c; --accent-dim:#8a6620;
    --ok:#4ade80; --warn:#e0a63c; --bad:#f87171; --info:#7dd3fc;
    --radius:12px;
  }
  * { box-sizing:border-box; }
  html, body { height:100%; }
  body {
    margin:0; background:var(--bg); color:var(--text);
    font-family:ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif;
    font-size:14px; line-height:1.5; overflow:hidden;
  }
  .mono { font-family:ui-monospace,SFMono-Regular,Menlo,monospace; }
  a { color:var(--accent); text-decoration:none; }
  a:hover { text-decoration:underline; }
  ::-webkit-scrollbar { width:9px; height:9px; }
  ::-webkit-scrollbar-thumb { background:#232a38; border-radius:6px; }
  ::-webkit-scrollbar-thumb:hover { background:#2f3849; }
  ::-webkit-scrollbar-track { background:transparent; }

  .shell { height:100dvh; display:grid; grid-template-rows:auto 1fr;
           max-width:1240px; margin:0 auto; padding:16px 20px 18px; gap:14px; }
  header { display:flex; align-items:center; gap:12px; flex-wrap:wrap; }
  .logo { width:26px; height:26px; border-radius:7px; flex:none;
          background:linear-gradient(140deg,var(--accent),var(--accent-dim));
          display:grid; place-items:center; color:#1a1206; font-weight:700; font-size:13px; }
  h1 { font-size:16px; font-weight:600; margin:0; letter-spacing:-.01em; }
  .tagline { font-size:12px; color:var(--dim); margin-top:1px; }
  .spacer { flex:1; }
  .badge { font-size:11px; padding:4px 9px; border-radius:999px;
           background:var(--panel-2); border:1px solid var(--line); color:var(--muted); }
  .badge .dot { display:inline-block; width:6px; height:6px; border-radius:50%;
                background:var(--ok); margin-right:6px; vertical-align:middle; }
  .card { background:var(--panel); border:1px solid var(--line); border-radius:var(--radius); }
  .btn { cursor:pointer; font-family:inherit; }
  .btn:hover { color:var(--accent); border-color:var(--accent-dim); }
"""

CHAT_PAGE = """<!DOCTYPE html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Trendly Support</title><style>""" + BASE_CSS + """
  .main { display:grid; grid-template-columns:minmax(0,1fr) 300px; gap:14px;
          min-height:0; }
  .chatcol { display:grid; grid-template-rows:1fr auto auto; gap:10px; min-height:0; }

  #chat { overflow-y:auto; padding:18px; display:flex; flex-direction:column; gap:14px; }
  .turn { display:flex; flex-direction:column; gap:5px; max-width:82%; }
  .turn.me { align-self:flex-end; align-items:flex-end; }
  .who { font-size:10px; letter-spacing:.11em; text-transform:uppercase; color:var(--dim); }
  .bubble { padding:11px 14px; border-radius:13px; white-space:pre-wrap;
            word-break:break-word; }
  .me .bubble { background:linear-gradient(160deg,var(--accent),#c9902f);
                color:#1a1206; border-bottom-right-radius:4px; font-weight:500; }
  .bot .bubble { background:var(--panel-2); border:1px solid var(--line);
                 border-bottom-left-radius:4px; }
  .err .bubble { background:#3a1616; border:1px solid #5c2626; color:#fecaca; }

  .trace { display:flex; flex-wrap:wrap; gap:5px; }
  .chip { font-size:10.5px; padding:2.5px 8px; border-radius:6px; background:#1a2130;
          border:1px solid var(--line); color:var(--muted); }
  .chip.out { color:var(--info); border-color:#24384a; }
  .chip.block { color:var(--warn); border-color:#4a3a1c; }
  .chip.viol { color:var(--bad); border-color:#4d2424; }
  .chip.esc { color:var(--bad); border-color:#4d2424; }
  .chip .ms { color:var(--dim); }

  .note { align-self:center; font-size:11.5px; color:var(--dim); font-style:italic;
          padding:5px 12px; border-radius:999px; background:#151a25;
          border:1px dashed var(--line); }
  .dots span { display:inline-block; width:5px; height:5px; margin-right:3px;
    border-radius:50%; background:var(--dim); animation:b 1.2s infinite; }
  .dots span:nth-child(2){animation-delay:.18s} .dots span:nth-child(3){animation-delay:.36s}
  @keyframes b { 0%,60%,100%{opacity:.25;transform:translateY(0)} 30%{opacity:1;transform:translateY(-3px)} }

  .composer { display:flex; gap:9px; padding:9px; align-items:center; }
  input { flex:1; padding:11px 13px; border-radius:9px; border:1px solid transparent;
          background:var(--panel-2); color:var(--text); font-size:14px; outline:none; }
  input:focus { border-color:var(--accent-dim); }
  .send { padding:11px 18px; border:0; border-radius:9px; cursor:pointer;
          background:var(--accent); color:#1a1206; font-weight:600; font-size:13.5px; }
  .send:disabled { opacity:.4; cursor:default; }

  .presets { display:flex; gap:6px; overflow-x:auto; padding-bottom:2px; }
  .presets button { flex:none; padding:5px 11px; font-size:11.5px; cursor:pointer;
    border-radius:999px; border:1px solid var(--line); background:var(--panel);
    color:var(--muted); white-space:nowrap; }
  .presets button:hover { color:var(--accent); border-color:var(--accent-dim); }

  .side { overflow-y:auto; padding:14px; }
  .side h3 { font-size:10px; letter-spacing:.11em; text-transform:uppercase;
             color:var(--dim); margin:18px 0 8px; font-weight:600; }
  .side h3:first-child { margin-top:0; }
  .kv { display:flex; justify-content:space-between; gap:10px; font-size:12px;
        padding:5px 0; border-bottom:1px solid #1b2130; }
  .kv b { font-weight:400; color:var(--muted); }
  .kv span { text-align:right; word-break:break-all; font-size:11.5px; }
  .tag { display:inline-block; font-size:11px; padding:3px 8px; border-radius:6px;
         background:#1a2130; border:1px solid var(--line); color:var(--muted);
         margin:0 4px 4px 0; }
  .tag.none { color:var(--dim); font-style:italic; border-style:dashed; }
  .tag.ok { color:var(--ok); border-color:#25462f; }
  .tag.no { color:var(--bad); border-color:#4d2424; }
  .tag.ex { color:var(--warn); border-color:#4a3a1c; }

  @media (max-width:900px) {
    .main { grid-template-columns:1fr; grid-template-rows:1fr auto; }
    .side { max-height:190px; }
  }
</style></head><body>
<div class="shell">
  <header>
    <div class="logo">T</div>
    <div>
      <h1>Trendly Support Assistant</h1>
      <div class="tagline">Tool calls, guardrail blocks and reply validation shown under each reply</div>
    </div>
    <div class="spacer"></div>
    <div class="badge" id="model-badge"><span class="dot"></span>connecting…</div>
    <button class="badge btn" id="new-session" title="Clears conversation state">New session</button>
    <a class="badge" href="/dashboard">Ops dashboard →</a>
  </header>

  <div class="main">
    <div class="chatcol">
      <div class="card" id="chat"></div>
      <div class="presets" id="presets"></div>
      <div class="card composer">
        <input id="input" placeholder="Ask about an order, a return, or our policy…" autofocus />
        <button class="send" id="send" onclick="send()">Send</button>
      </div>
    </div>

    <div class="card side">
      <h3>Session state</h3>
      <div class="kv"><b>session</b><span class="mono" id="s-id">—</span></div>
      <div class="kv"><b>bound customer</b><span class="mono" id="s-cust">not bound</span></div>
      <h3>Orders looked up</h3>
      <div id="s-orders"><span class="tag none">none</span></div>
      <h3>Eligibility decided</h3>
      <div id="s-elig"><span class="tag none">none</span></div>
      <h3>Last turn</h3>
      <div class="kv"><b>model</b><span class="mono" id="d-model">—</span></div>
      <div class="kv"><b>latency</b><span class="mono" id="d-latency">—</span></div>
      <div class="kv"><b>agent steps</b><span class="mono" id="d-iter">—</span></div>
      <div class="kv"><b>blocked calls</b><span class="mono" id="d-blocked">—</span></div>
      <div class="kv"><b>redrafts</b><span class="mono" id="d-retries">—</span></div>
      <div class="kv"><b>escalation</b><span class="mono" id="d-esc">—</span></div>
    </div>
  </div>
</div>

<script>
let sessionId, boundCustomer = null;
const chat = document.getElementById("chat");
const input = document.getElementById("input");
const button = document.getElementById("send");

fetch("/health").then(r => r.json()).then(h => {
  document.getElementById("model-badge").innerHTML =
    '<span class="dot"></span>' + h.model +
    (h.cassette_mode && h.cassette_mode !== "off" ? " &middot; " + h.cassette_mode : "");
}).catch(() => {});

// Each preset records the customer who owns the order it references. A
// session binds to a customer on its first lookup and then refuses orders
// belonging to anyone else — correct in production, where one chat is one
// customer, but it means clicking presets across four customers in a single
// session blocks everything after the first. So a preset that crosses a
// customer boundary starts a fresh session and says so, rather than
// weakening the guardrail to make the demo convenient.
const PRESETS = [
  ["Happy path",     "I'd like to return my kurta from order TR-4530, wrong size.", "C-101"],
  ["Final sale",     "I want to return my shirt from TR-4528, wrong size.", "C-103"],
  ["Jewellery",      "Can I return the earrings from order TR-4527?", "C-102"],
  ["Out of window",  "I want to return the jacket from TR-4523, it doesn't fit.", "C-102"],
  ["Lost parcel",    "Order TR-4526 never arrived, what do I do?", "C-101"],
  ["Delayed",        "Where is my order TR-4525? It's really late.", "C-103"],
  ["Partial",        "Order TR-4524 only had one thing in the box.", "C-100"],
  ["Cancelled",      "Can I return the scarf from TR-4529?", "C-100"],
  ["Damaged",        "The earrings from TR-4527 turned up cracked and broken.", "C-102"],
  ["Ambiguous item", "I'd like to return the leather jacket from order TR-4524.", "C-100"],
  ["Cross-customer", "Who placed order TR-4522 and what's in it?", null],
  ["Discount",       "Can you give me a 20% discount code?", null],
  ["Policy",         "How long does a refund take once you receive my return?", null]
];
const presets = document.getElementById("presets");
PRESETS.forEach(function (p) {
  const b = document.createElement("button");
  b.textContent = p[0];
  b.onclick = function () {
    if (p[2] && boundCustomer && p[2] !== boundCustomer) {
      newSession("That order belongs to a different customer — starting a new session.");
    }
    input.value = p[1];
    send();
  };
  presets.appendChild(b);
});

function newSession(note) {
  sessionId = "demo-" + Math.random().toString(36).slice(2, 10);
  boundCustomer = null;
  document.getElementById("s-id").textContent = sessionId;
  document.getElementById("s-cust").textContent = "not bound";
  tags(document.getElementById("s-orders"), []);
  tags(document.getElementById("s-elig"), []);
  ["d-model", "d-latency", "d-iter", "d-blocked", "d-retries", "d-esc"]
    .forEach(function (id) { document.getElementById(id).textContent = "—"; });
  if (note) {
    const n = el("div", "note", note);
    chat.appendChild(n);
    chat.scrollTop = chat.scrollHeight;
  }
}

function el(tag, cls, text) {
  const d = document.createElement(tag);
  if (cls) d.className = cls;
  if (text !== undefined) d.textContent = text;
  return d;
}

function addTurn(who, text, cls) {
  const wrap = el("div", "turn " + cls);
  wrap.appendChild(el("div", "who", who));
  wrap.appendChild(el("div", "bubble", text));
  chat.appendChild(wrap);
  chat.scrollTop = chat.scrollHeight;
  return wrap;
}

function addTrace(wrap, data) {
  const calls = data.tool_calls_made || [];
  const d = data.diagnostics || {};
  const viols = d.validation_violations || [];
  if (!calls.length && !viols.length && !data.escalated) return;

  const row = el("div", "trace");
  calls.forEach(function (c) {
    const r = c.result || {};
    if (r.blocked) { row.appendChild(el("span", "chip block", c.tool + " · blocked")); return; }
    row.appendChild(el("span", r.outcome ? "chip out" : "chip",
      c.tool + (r.outcome ? " · " + r.outcome : "")));
  });
  if (viols.length) row.appendChild(el("span", "chip viol", "redrafted · " + viols.join(", ")));
  if (data.escalated) row.appendChild(el("span", "chip esc", "escalated" + (d.escalation_reason ? " · " + d.escalation_reason : "")));
  if (d.latency_ms) row.appendChild(el("span", "chip", Math.round(d.latency_ms) + " ms"));
  wrap.appendChild(row);
  chat.scrollTop = chat.scrollHeight;
}

function tags(el_, items, clsOf) {
  el_.innerHTML = "";
  if (!items.length) { el_.appendChild(el("span", "tag none", "none")); return; }
  items.forEach(function (t) {
    el_.appendChild(el("span", "tag " + (clsOf ? clsOf(t) : ""), t));
  });
}

function updateSide(data) {
  const s = data.state || {}, d = data.diagnostics || {};
  boundCustomer = s.session_customer_id || null;
  document.getElementById("s-cust").textContent = s.session_customer_id || "not bound";
  tags(document.getElementById("s-orders"), s.looked_up_orders || []);

  const eo = s.eligibility_outcomes || {};
  const rows = Object.keys(eo).map(function (k) { return k.split("|")[1] + " · " + eo[k]; });
  tags(document.getElementById("s-elig"), rows, function (t) {
    if (t.indexOf("eligible_refund") > -1) return "ok";
    if (t.indexOf("exchange_only") > -1) return "ex";
    return "no";
  });

  document.getElementById("d-model").textContent =
    (d.model_used || "—") + (d.fallback_used ? " (fallback)" : "");
  document.getElementById("d-latency").textContent =
    d.latency_ms ? Math.round(d.latency_ms) + " ms" : "—";
  document.getElementById("d-iter").textContent = s.iterations_this_turn;
  document.getElementById("d-blocked").textContent = d.blocked_calls || 0;
  document.getElementById("d-retries").textContent = s.validation_retries || 0;
  document.getElementById("d-esc").textContent = d.escalation_reason || "none";
}

async function send() {
  const message = input.value.trim();
  if (!message) return;
  addTurn("You", message, "me");
  input.value = "";
  button.disabled = true;

  const pending = el("div", "turn bot");
  pending.appendChild(el("div", "who", "Trendly"));
  const bub = el("div", "bubble");
  bub.innerHTML = '<span class="dots"><span></span><span></span><span></span></span>';
  pending.appendChild(bub);
  chat.appendChild(pending);
  chat.scrollTop = chat.scrollHeight;

  try {
    const res = await fetch("/chat", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sessionId, message })
    });
    pending.remove();
    if (!res.ok) {
      const e = await res.json().catch(function () { return { detail: res.statusText }; });
      addTurn("Error", "Server error (" + res.status + "): " + (e.detail || "unknown"), "bot err");
      return;
    }
    const data = await res.json();
    const wrap = addTurn("Trendly", data.reply, "bot");
    addTrace(wrap, data);
    updateSide(data);
  } catch (err) {
    pending.remove();
    addTurn("Error", "Request failed: " + err.message, "bot err");
  } finally {
    button.disabled = false;
    input.focus();
  }
}
input.addEventListener("keydown", function (e) { if (e.key === "Enter") send(); });
document.getElementById("new-session").onclick = function () {
  newSession("New session started.");
  input.focus();
};
newSession();
</script></body></html>
"""

DASHBOARD_PAGE = """<!DOCTYPE html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Trendly Agent — Ops</title><style>""" + BASE_CSS + """
  .body { overflow-y:auto; min-height:0; padding-right:4px; }
  .grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(178px,1fr));
          gap:11px; margin-bottom:13px; }
  .metric { padding:13px 15px; }
  .metric .label { font-size:10px; letter-spacing:.11em; text-transform:uppercase;
                   color:var(--dim); font-weight:600; }
  .metric .value { font-size:27px; font-weight:650; margin-top:4px; letter-spacing:-.02em; }
  .metric .foot { font-size:11px; color:var(--dim); margin-top:2px; }
  .value.ok { color:var(--ok); } .value.warn { color:var(--warn); }
  .value.accent { color:var(--accent); }

  .cols { display:grid; grid-template-columns:repeat(auto-fit,minmax(300px,1fr)); gap:11px; }
  .panel { padding:14px; }
  .panel h2 { font-size:10px; letter-spacing:.11em; text-transform:uppercase;
              color:var(--dim); margin:0 0 11px; font-weight:600; }
  .row { margin-bottom:9px; }
  .row:last-child { margin-bottom:0; }
  .rowtop { display:flex; justify-content:space-between; font-size:12.5px; margin-bottom:4px; }
  .rowtop span:last-child { color:var(--muted); font-variant-numeric:tabular-nums; }
  .track { height:5px; border-radius:3px; background:#1a2130; overflow:hidden; }
  .fill { height:100%; border-radius:3px;
          background:linear-gradient(90deg,var(--accent-dim),var(--accent)); }
  .fill.bad { background:linear-gradient(90deg,#7a2c2c,var(--bad)); }
  .fill.info { background:linear-gradient(90deg,#26536b,var(--info)); }
  .empty { color:var(--dim); font-size:12.5px; font-style:italic; }
</style></head><body>
<div class="shell">
  <header>
    <div class="logo">T</div>
    <div>
      <h1>Trendly Agent — Operations</h1>
      <div class="tagline">Live from the turn-level event log, refreshed every 5s</div>
    </div>
    <div class="spacer"></div>
    <a class="badge" href="/metrics">Raw JSON</a>
    <a class="badge" href="/">← Back to chat</a>
  </header>
  <div class="body" id="body"><div class="card panel"><div class="empty">Loading…</div></div></div>
</div>
<script>
function metric(label, value, foot, cls) {
  return '<div class="card metric"><div class="label">' + label + '</div>' +
    '<div class="value ' + (cls || "") + '">' + value + '</div>' +
    '<div class="foot">' + (foot || "&nbsp;") + '</div></div>';
}
function panel(title, obj, emptyText, fillCls) {
  const keys = Object.keys(obj || {});
  let inner;
  if (!keys.length) {
    inner = '<div class="empty">' + emptyText + '</div>';
  } else {
    const max = Math.max.apply(null, keys.map(function (k) { return obj[k]; })) || 1;
    inner = keys.map(function (k) {
      const pct = Math.round((obj[k] / max) * 100);
      return '<div class="row"><div class="rowtop"><span>' + k + '</span><span>' +
        obj[k] + '</span></div><div class="track"><div class="fill ' +
        (fillCls || "") + '" style="width:' + pct + '%"></div></div></div>';
    }).join("");
  }
  return '<div class="card panel"><h2>' + title + '</h2>' + inner + '</div>';
}
async function load() {
  let m;
  try { m = await (await fetch("/metrics")).json(); }
  catch (e) { return; }
  const el = document.getElementById("body");
  if (!m.turns) {
    el.innerHTML = '<div class="card panel"><div class="empty">' +
      'No turns recorded yet — send a message from the chat page.</div></div>';
    return;
  }
  const pct = Math.round(m.deflection_rate * 100);
  el.innerHTML =
    '<div class="grid">' +
      metric("Deflection rate", pct + "%",
        m.handled_without_human + " of " + m.turns + " turns closed without a human",
        pct >= 70 ? "ok" : "warn") +
      metric("Turns", m.turns, m.sessions + " session" + (m.sessions === 1 ? "" : "s"), "accent") +
      metric("Escalations", m.escalated,
        m.escalations.policy_mandated + " policy-mandated · " +
        m.escalations.agent_limitation + " agent limit") +
      metric("Guardrail blocks", m.guardrails.blocked_tool_calls,
        m.guardrails.reply_validation_retries + " replies redrafted") +
      metric("Latency p50", Math.round(m.latency_ms.p50) + " ms",
        "p95 " + Math.round(m.latency_ms.p95) + " ms · max " + Math.round(m.latency_ms.max) + " ms") +
    '</div>' +
    '<div class="cols">' +
      panel("Escalations by reason", m.escalations.by_reason,
        "None — every turn was handled.", "bad") +
      panel("Eligibility outcomes", m.eligibility_outcomes, "No eligibility checks yet.") +
      panel("Tool usage", m.tools, "No tools called yet.", "info") +
      panel("Validation violations caught", m.guardrails.validation_violations,
        "No replies failed validation.", "bad") +
      panel("Turns by model", m.models.turns_by_model, "—") +
    '</div>';
}
load(); setInterval(load, 5000);
</script></body></html>
"""
