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
import secrets

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from app.agent import MODEL, PROVIDER, get_session_history, graph, run_agent
from app.tools import customer_profile
from app.cassettes import mode as cassette_mode
from app.observability import list_sessions, summarize

app = FastAPI(title="Trendly Support Agent")

# ---------- Operator routes ----------
# /metrics, /sessions, /dashboard and /history expose every conversation the
# agent has served, across all customers. That is an operator view, not a
# customer one, so it is gated:
#
#   ADMIN_TOKEN set    -> a matching token is required from anywhere
#   ADMIN_TOKEN unset  -> local requests only; anything else 404s
#
# 404 rather than 403 on purpose: an unauthenticated caller shouldn't learn
# that an admin surface exists. The customer chat page links to none of it.
ADMIN_TOKEN = os.environ.get("ADMIN_TOKEN", "").strip()
ADMIN_COOKIE = "trendly_admin"
LOCAL_HOSTS = {"127.0.0.1", "::1", "localhost", "testclient"}


def _is_admin(request: Request) -> bool:
    if ADMIN_TOKEN:
        supplied = (
            request.headers.get("x-admin-token")
            or request.query_params.get("token")
            or request.cookies.get(ADMIN_COOKIE)
            or ""
        )
        return secrets.compare_digest(supplied, ADMIN_TOKEN)
    host = request.client.host if request.client else ""
    return host in LOCAL_HOSTS


def require_admin(request: Request) -> None:
    if not _is_admin(request):
        raise HTTPException(status_code=404, detail="Not found")


def _admin_page(request: Request, html: str) -> HTMLResponse:
    """Serve an operator page, remembering a token passed as ?token=... so the
    page's own fetch calls to /metrics and /sessions are authorised too."""
    response = HTMLResponse(html)
    token = request.query_params.get("token")
    if ADMIN_TOKEN and token and secrets.compare_digest(token, ADMIN_TOKEN):
        response.set_cookie(
            ADMIN_COOKIE, token, httponly=True, samesite="strict", max_age=86400
        )
    return response


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
def health(request: Request):
    """Public liveness probe. Which model and configuration is running is
    operator information, so it is only included for an admin caller."""
    if not _is_admin(request):
        return {"status": "ok"}
    return {
        "status": "ok",
        "provider": PROVIDER,
        "model": MODEL,
        "cassette_mode": cassette_mode(),
        "clock_override": os.environ.get("TRENDLY_NOW"),
        "admin_auth": "token" if ADMIN_TOKEN else "localhost-only",
    }


@app.get("/metrics", dependencies=[Depends(require_admin)])
def metrics():
    """Aggregate view of every turn served. Consumed by /dashboard."""
    return summarize()


@app.get("/sessions", dependencies=[Depends(require_admin)])
def sessions_index(limit: int = 200):
    """Every conversation the agent has served, newest first."""
    return {"sessions": list_sessions(limit)}


@app.get("/history", response_class=HTMLResponse,
         dependencies=[Depends(require_admin)])
def history(request: Request):
    return _admin_page(request, HISTORY_PAGE)


@app.get("/session/{session_id}")
def session(session_id: str):
    """Replay a conversation from the checkpointer so the customer's own page
    can rehydrate after a refresh.

    Deliberately not admin-gated — the customer needs it — so it is protected
    only by the unguessability of the session id. That is weak: anyone holding
    an id can read that conversation. Real deployment ties the thread to an
    authenticated user instead, which is the same gap the cross-customer
    binding papers over. Noted in SOLUTION.md."""
    try:
        return get_session_history(session_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/demo/contacts", dependencies=[Depends(require_admin)])
def demo_contacts():
    """
    The seeded accounts, for switching between them quickly during
    evaluation. Admin-gated like the other operator routes: locally it just
    works, and on a deployed host without a token it 404s, so a real customer
    can never enumerate the customer list. The chat page hides the switcher
    when this returns anything other than 200.
    """
    from app.tools import _load_data

    return {
        "contacts": [
            {
                "customer_id": c["customer_id"],
                "name": c["name"],
                "first_name": c["name"].split()[0],
                "email": c["email"],
            }
            for c in _load_data()["customers"]
        ]
    }


@app.get("/session/{session_id}/customer")
def session_customer(session_id: str):
    """
    The signed-in customer's own profile and orders, for the panel beside the
    chat. Returns nothing until the session has been verified, and only ever
    the customer bound to this session — contact details come back masked,
    since the point is to confirm which account is open, not to display it.
    """
    try:
        snapshot = graph.get_state({"configurable": {"thread_id": session_id}})
        values = snapshot.values if snapshot else {}
        customer_id = values.get("session_customer_id")
        if not customer_id:
            return {"verified": False}
        profile = customer_profile(customer_id)
        return {"verified": True, **profile} if profile else {"verified": False}
    except Exception:
        return {"verified": False}


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


@app.get("/dashboard", response_class=HTMLResponse,
         dependencies=[Depends(require_admin)])
def dashboard(request: Request):
    return _admin_page(request, DASHBOARD_PAGE)


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
  .main { display:grid; grid-template-columns:minmax(0,1fr) 316px; gap:14px; min-height:0; }
  .chatcol { display:grid; grid-template-rows:1fr auto auto; gap:10px; min-height:0; }

  #chat { overflow-y:auto; padding:20px; display:flex; flex-direction:column; gap:16px;
          scroll-behavior:smooth; }
  .turn { display:flex; flex-direction:column; gap:5px; max-width:80%;
          animation:rise .28s cubic-bezier(.2,.8,.3,1); }
  .turn.me { align-self:flex-end; align-items:flex-end; }
  @keyframes rise { from { opacity:0; transform:translateY(7px); } to { opacity:1; transform:none; } }
  .who { font-size:10px; letter-spacing:.11em; text-transform:uppercase; color:var(--dim); }
  .bubble { padding:12px 15px; border-radius:14px; white-space:pre-wrap;
            word-break:break-word; line-height:1.55; }
  .me .bubble { background:linear-gradient(160deg,var(--accent),#c9902f);
                color:#1a1206; border-bottom-right-radius:5px; font-weight:500; }
  .bot .bubble { background:var(--panel-2); border:1px solid var(--line);
                 border-bottom-left-radius:5px; }
  .err .bubble { background:#3a1616; border:1px solid #5c2626; color:#fecaca; }
  .note { align-self:center; font-size:11.5px; color:var(--dim); font-style:italic;
          padding:6px 14px; border-radius:999px; background:#151a25;
          border:1px dashed var(--line); }

  .hero { margin:auto; text-align:center; max-width:400px; padding:20px; }
  .hero .mark { width:46px; height:46px; border-radius:13px; margin:0 auto 14px;
    background:linear-gradient(140deg,var(--accent),var(--accent-dim));
    display:grid; place-items:center; color:#1a1206; font-weight:700; font-size:20px; }
  .hero h2 { font-size:17px; margin:0 0 6px; font-weight:600; }
  .hero p { font-size:13px; color:var(--muted); margin:0; line-height:1.6; }

  .dots span { display:inline-block; width:5px; height:5px; margin-right:3px;
    border-radius:50%; background:var(--dim); animation:b 1.2s infinite; }
  .dots span:nth-child(2){animation-delay:.18s} .dots span:nth-child(3){animation-delay:.36s}
  @keyframes b { 0%,60%,100%{opacity:.25;transform:translateY(0)} 30%{opacity:1;transform:translateY(-3px)} }

  .composer { display:flex; gap:9px; padding:9px; align-items:center;
              transition:border-color .18s; }
  .composer:focus-within { border-color:var(--accent-dim); }
  input[type=text] { flex:1; padding:11px 13px; border-radius:9px; border:1px solid transparent;
          background:var(--panel-2); color:var(--text); font-size:14px; outline:none; }
  .send { padding:11px 18px; border:0; border-radius:9px; cursor:pointer;
          background:var(--accent); color:#1a1206; font-weight:600; font-size:13.5px;
          transition:transform .12s, opacity .12s; }
  .send:hover:not(:disabled) { transform:translateY(-1px); }
  .send:disabled { opacity:.4; cursor:default; }

  .switcher { display:none; gap:6px; align-items:center; overflow-x:auto;
    padding:8px 10px; }
  .switcher.on { display:flex; }
  .switcher .lbl { flex:none; font-size:10px; letter-spacing:.11em;
    text-transform:uppercase; color:var(--dim); font-weight:600; margin-right:2px; }
  .switcher button { flex:none; padding:5px 11px; font-size:11.5px; cursor:pointer;
    border-radius:999px; border:1px solid var(--line); background:var(--panel-2);
    color:var(--muted); white-space:nowrap; transition:all .15s; }
  .switcher button:hover { color:var(--accent); border-color:var(--accent-dim); }
  .switcher button.on { color:var(--accent); border-color:var(--accent-dim); }

  .quick { display:flex; gap:6px; overflow-x:auto; padding-bottom:2px; }
  .quick button { flex:none; padding:6px 12px; font-size:11.5px; cursor:pointer;
    border-radius:999px; border:1px solid var(--line); background:var(--panel);
    color:var(--muted); white-space:nowrap; transition:all .15s; }
  .quick button:hover { color:var(--accent); border-color:var(--accent-dim);
    transform:translateY(-1px); }

  .side { overflow-y:auto; padding:15px; }
  .side h3 { font-size:10px; letter-spacing:.11em; text-transform:uppercase;
             color:var(--dim); margin:18px 0 9px; font-weight:600; }
  .side h3:first-child { margin-top:0; }
  .locked { text-align:center; padding:26px 12px; color:var(--dim); }
  .locked .ico { font-size:22px; opacity:.5; }
  .locked p { font-size:12px; line-height:1.6; margin:10px 0 0; }

  .who-card { display:flex; align-items:center; gap:11px; padding:11px;
    border-radius:10px; background:var(--panel-2); border:1px solid var(--line); }
  .avatar { width:36px; height:36px; border-radius:50%; flex:none; display:grid;
    place-items:center; font-weight:600; font-size:13px; color:#1a1206;
    background:linear-gradient(140deg,var(--accent),var(--accent-dim)); }
  .who-card .nm { font-size:13px; font-weight:600; }
  .who-card .vf { font-size:10.5px; color:var(--ok); margin-top:1px; }
  .contact { font-size:11.5px; color:var(--muted); padding:7px 0;
    border-bottom:1px solid #1b2130; display:flex; justify-content:space-between; gap:8px; }
  .contact span:last-child { font-family:ui-monospace,monospace; color:var(--dim); }

  .order { padding:10px 11px; border-radius:9px; border:1px solid var(--line);
    background:var(--panel-2); margin-bottom:7px; cursor:pointer; transition:all .15s; }
  .order:hover { border-color:var(--accent-dim); transform:translateY(-1px); }
  .order .top { display:flex; justify-content:space-between; align-items:center; gap:8px; }
  .order .oid { font-size:12px; font-family:ui-monospace,monospace; font-weight:600; }
  .order .items { font-size:11px; color:var(--muted); margin-top:4px; line-height:1.45; }
  .st { font-size:9.5px; padding:2px 7px; border-radius:999px; white-space:nowrap;
        text-transform:uppercase; letter-spacing:.05em; font-weight:600; }
  .st.delivered { color:var(--ok); background:#12281c; }
  .st.transit { color:var(--info); background:#122430; }
  .st.late { color:var(--warn); background:#2a2113; }
  .st.bad { color:var(--bad); background:#2c1618; }
  .st.off { color:var(--dim); background:#1a1f2b; }

  @media (max-width:900px) {
    .main { grid-template-columns:1fr; grid-template-rows:1fr auto; }
    .side { max-height:190px; }
  }
</style></head><body>
<div class="shell">
  <header>
    <div class="logo">T</div>
    <div>
      <h1>Trendly Support</h1>
      <div class="tagline">Orders, returns, exchanges and delivery — 9 AM to 9 PM IST, every day</div>
    </div>
    <div class="spacer"></div>
    <div class="badge" id="status-badge"><span class="dot"></span>Online</div>
    <button class="badge btn" id="switch-account" title="Sign in as a different account">Switch account</button>
    <button class="badge btn" id="end-session" title="Clears this conversation">End chat</button>
  </header>

  <div class="main">
    <div class="chatcol">
      <div class="card" id="chat"></div>
      <div class="card switcher" id="switcher"></div>
      <div class="quick" id="quick"></div>
      <div class="card composer">
        <input type="text" id="input" placeholder="Type your message…" autofocus />
        <button class="send" id="send" onclick="send()">Send</button>
      </div>
    </div>
    <div class="card side" id="side"></div>
  </div>
</div>

<script>
const SESSION_KEY = "trendly.session";
let sessionId, verified = false, boundCustomer = null;
const chat = document.getElementById("chat");
const input = document.getElementById("input");
const button = document.getElementById("send");
const side = document.getElementById("side");

const QUICK_BEFORE = [
  ["Where's my order?", "Where's my order?"],
  ["Return an item", "I'd like to return something."],
  ["Exchange a size", "I need to exchange an item for a different size."],
  ["Refund status", "When will my refund come through?"],
  ["Delivery times", "How long does delivery usually take?"],
  ["Return policy", "What's your return policy?"],
  ["Talk to a person", "Can I speak to a human agent please?"]
];
const QUICK_AFTER = [
  ["Track an order", "Can you tell me where my order is?"],
  ["Return an item", "I'd like to return an item from one of my orders."],
  ["Exchange a size", "I'd like to exchange an item for a different size."],
  ["Refund status", "When will my refund come through?"],
  ["Damaged item", "Something arrived damaged."],
  ["Return policy", "What's your return policy?"],
  ["Talk to a person", "Can I speak to a human agent please?"]
];

function el(tag, cls, text) {
  const d = document.createElement(tag);
  if (cls) d.className = cls;
  if (text !== undefined) d.textContent = text;
  return d;
}

function renderQuick() {
  const box = document.getElementById("quick");
  box.innerHTML = "";
  (verified ? QUICK_AFTER : QUICK_BEFORE).forEach(function (q) {
    const b = el("button", "", q[0]);
    b.onclick = function () { input.value = q[1]; send(); };
    box.appendChild(b);
  });
}

// Evaluation affordance: one click to start a fresh session signed in as any
// seeded account. Hidden entirely when /demo/contacts isn't reachable, which
// is the case on a deployed host without an admin token.
let contacts = [];

async function loadSwitcher() {
  const bar = document.getElementById("switcher");
  try {
    const res = await fetch("/demo/contacts");
    if (!res.ok) return;
    contacts = (await res.json()).contacts || [];
  } catch (e) { return; }
  if (!contacts.length) return;

  bar.classList.add("on");
  renderSwitcher();
}

function renderSwitcher() {
  const bar = document.getElementById("switcher");
  bar.innerHTML = "";
  bar.appendChild(el("div", "lbl", "Sign in as"));
  contacts.forEach(function (c) {
    const b = el("button", boundCustomer === c.customer_id ? "on" : "",
                 c.first_name + " · " + c.customer_id);
    b.title = c.email;
    b.onclick = function () { signInAs(c); };
    bar.appendChild(b);
  });
}

function signInAs(contact) {
  newSession("Starting a new conversation as " + contact.name + ".");
  input.value = "Hi, my email is " + contact.email;
  send();
}

function welcome() {
  const hero = el("div", "hero");
  hero.appendChild(el("div", "mark", "T"));
  hero.appendChild(el("h2", "", "Hi, how can we help?"));
  hero.appendChild(el("p", "",
    "Ask about an order, a return or our policies. To look up anything on " +
    "your account we'll check your email or phone number first."));
  chat.appendChild(hero);
}

function addTurn(who, text, cls) {
  const hero = chat.querySelector(".hero");
  if (hero) hero.remove();
  const wrap = el("div", "turn " + cls);
  wrap.appendChild(el("div", "who", who));
  wrap.appendChild(el("div", "bubble", text));
  chat.appendChild(wrap);
  chat.scrollTop = chat.scrollHeight;
  return wrap;
}

function statusClass(s) {
  if (s === "delivered") return "delivered";
  if (s === "in_transit" || s === "partially_shipped") return "transit";
  if (s === "delayed") return "late";
  if (s === "lost_in_transit") return "bad";
  return "off";
}
function statusLabel(s) { return (s || "").replace(/_/g, " "); }

function lockedPanel() {
  side.innerHTML = "";
  const box = el("div", "locked");
  box.appendChild(el("div", "ico", "🔒"));
  box.appendChild(el("p", "",
    "Your account details and orders will appear here once we've confirmed " +
    "your email address or phone number."));
  side.appendChild(box);
}

function renderCustomer(p) {
  side.innerHTML = "";
  side.appendChild(el("h3", "", "Your account"));

  const card = el("div", "who-card");
  const initials = p.name.split(" ").map(function (n) { return n[0]; }).join("").slice(0, 2);
  card.appendChild(el("div", "avatar", initials));
  const who = el("div");
  who.appendChild(el("div", "nm", p.name));
  who.appendChild(el("div", "vf", "✓ Verified"));
  card.appendChild(who);
  side.appendChild(card);

  const email = el("div", "contact");
  email.appendChild(el("span", "", "Email"));
  email.appendChild(el("span", "", p.email_masked));
  side.appendChild(email);
  const phone = el("div", "contact");
  phone.appendChild(el("span", "", "Phone"));
  phone.appendChild(el("span", "", p.phone_masked));
  side.appendChild(phone);

  side.appendChild(el("h3", "", "Your orders (" + p.orders.length + ")"));
  p.orders.forEach(function (o) {
    const row = el("div", "order");
    const top = el("div", "top");
    top.appendChild(el("div", "oid", o.order_id));
    top.appendChild(el("div", "st " + statusClass(o.status), statusLabel(o.status)));
    row.appendChild(top);
    row.appendChild(el("div", "items", o.items.join(", ")));
    row.onclick = function () {
      input.value = "Tell me about order " + o.order_id;
      send();
    };
    side.appendChild(row);
  });
}

async function refreshSide() {
  try {
    const p = await (await fetch("/session/" + encodeURIComponent(sessionId) + "/customer")).json();
    if (p.verified) {
      const was = verified;
      verified = true;
      boundCustomer = p.customer_id;
      renderCustomer(p);
      if (!was) renderQuick();
    } else {
      verified = false;
      boundCustomer = null;
      lockedPanel();
    }
    if (contacts.length) renderSwitcher();
  } catch (e) { lockedPanel(); }
}

function newSession(note) {
  sessionId = "s-" + Math.random().toString(36).slice(2, 10);
  verified = false;
  boundCustomer = null;
  try { sessionStorage.setItem(SESSION_KEY, sessionId); } catch (e) {}
  chat.innerHTML = "";
  if (note) chat.appendChild(el("div", "note", note));
  welcome();
  lockedPanel();
  renderQuick();
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
      addTurn("Trendly", "Sorry — something went wrong on our end (" + res.status +
        "). Please try again in a moment.", "bot err");
      return;
    }
    const data = await res.json();
    addTurn("Trendly", data.reply, "bot");
    refreshSide();
  } catch (err) {
    pending.remove();
    addTurn("Trendly", "I couldn't reach our systems just then. Please try again.", "bot err");
  } finally {
    button.disabled = false;
    input.focus();
  }
}

document.getElementById("switch-account").onclick = function () {
  if (contacts.length) {
    // The switcher strip is already on screen; nudge toward it.
    newSession("Pick an account below to sign in as.");
    input.focus();
    return;
  }
  newSession("Starting a new conversation.");
  addTurn("Trendly", "Sure — what's the email address or phone number on the " +
    "account you'd like to use?", "bot");
  input.focus();
};

document.getElementById("end-session").onclick = function () {
  if (!chat.querySelector(".turn")) return;
  addTurn("Trendly", "Thanks for chatting with us. This conversation is now " +
    "closed — start a new one any time and we'll pick things up fresh.", "bot");
  button.disabled = true;
  input.disabled = true;
  setTimeout(function () {
    newSession("Previous chat ended.");
    button.disabled = false;
    input.disabled = false;
    input.focus();
  }, 2600);
};

input.addEventListener("keydown", function (e) { if (e.key === "Enter") send(); });

async function restore() {
  let saved = null;
  try { saved = sessionStorage.getItem(SESSION_KEY); } catch (e) {}
  if (!saved) { newSession(); return; }
  sessionId = saved;
  renderQuick();
  try {
    const data = await (await fetch("/session/" + encodeURIComponent(saved))).json();
    if (data.exists) {
      (data.turns || []).forEach(function (t) {
        addTurn(t.role === "user" ? "You" : "Trendly", t.content,
                t.role === "user" ? "me" : "bot");
      });
      chat.appendChild(el("div", "note", "Conversation restored."));
    } else { welcome(); }
  } catch (e) { welcome(); }
  refreshSide();
}
loadSwitcher();
restore();
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
    <a class="badge" href="/history">History</a>
    <a class="badge" href="/metrics">Raw JSON</a>
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


HISTORY_PAGE = """<!DOCTYPE html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Trendly Agent — History</title><style>""" + BASE_CSS + """
  .main { display:grid; grid-template-columns:330px minmax(0,1fr); gap:14px; min-height:0; }
  .list { overflow-y:auto; padding:8px; }
  .filters { display:flex; gap:6px; padding:6px 6px 10px; flex-wrap:wrap; }
  .filters button { flex:none; padding:4px 10px; font-size:11px; cursor:pointer;
    border-radius:999px; border:1px solid var(--line); background:var(--panel-2);
    color:var(--muted); }
  .filters button.on { color:var(--accent); border-color:var(--accent-dim); }
  .grouphead { font-size:10px; letter-spacing:.11em; text-transform:uppercase;
    color:var(--dim); padding:12px 8px 5px; font-weight:600; }
  .item { padding:9px 10px; border-radius:9px; cursor:pointer; border:1px solid transparent; }
  .item:hover { background:var(--panel-2); }
  .item.on { background:var(--panel-2); border-color:var(--accent-dim); }
  .item .id { font-size:12px; font-family:ui-monospace,monospace; }
  .item .meta { font-size:11px; color:var(--dim); margin-top:3px;
    display:flex; gap:8px; flex-wrap:wrap; }
  .item .meta .esc { color:var(--bad); }
  .item .meta .blk { color:var(--warn); }
  .transcript { overflow-y:auto; padding:18px; display:flex;
    flex-direction:column; gap:14px; }
  .turn { display:flex; flex-direction:column; gap:5px; max-width:84%; }
  .turn.me { align-self:flex-end; align-items:flex-end; }
  .who { font-size:10px; letter-spacing:.11em; text-transform:uppercase; color:var(--dim); }
  .bubble { padding:11px 14px; border-radius:13px; white-space:pre-wrap;
    word-break:break-word; }
  .me .bubble { background:linear-gradient(160deg,var(--accent),#c9902f);
    color:#1a1206; border-bottom-right-radius:4px; font-weight:500; }
  .bot .bubble { background:var(--panel-2); border:1px solid var(--line);
    border-bottom-left-radius:4px; }
  .trace { display:flex; flex-wrap:wrap; gap:5px; }
  .chip { font-size:10.5px; padding:2.5px 8px; border-radius:6px; background:#1a2130;
    border:1px solid var(--line); color:var(--muted); }
  .chip.out { color:var(--info); border-color:#24384a; }
  .chip.block { color:var(--warn); border-color:#4a3a1c; }
  .empty { color:var(--dim); font-size:13px; font-style:italic; padding:18px; }
  .sumbar { font-size:11.5px; color:var(--dim); padding:9px 14px;
    border-bottom:1px solid var(--line); display:flex; gap:14px; flex-wrap:wrap; }
  .pane { display:grid; grid-template-rows:auto 1fr; min-height:0; }
  @media (max-width:880px) { .main { grid-template-columns:1fr; } .list { max-height:220px; } }
</style></head><body>
<div class="shell">
  <header>
    <div class="logo">T</div>
    <div>
      <h1>Conversation history</h1>
      <div class="tagline">Every session the agent has served — transcripts replayed from the checkpointer</div>
    </div>
    <div class="spacer"></div>
    <a class="badge" href="/dashboard">Ops dashboard</a>
  </header>
  <div class="main">
    <div class="card list">
      <div class="filters" id="filters"></div>
      <div id="items"><div class="empty">Loading…</div></div>
    </div>
    <div class="card pane">
      <div class="sumbar" id="sumbar">Select a conversation</div>
      <div class="transcript" id="transcript"><div class="empty">Nothing selected.</div></div>
    </div>
  </div>
</div>
<script>
let sessions = [], filter = null, selected = null;

function el(tag, cls, text) {
  const d = document.createElement(tag);
  if (cls) d.className = cls;
  if (text !== undefined) d.textContent = text;
  return d;
}
function ago(iso) {
  if (!iso) return "";
  const mins = Math.floor((Date.now() - new Date(iso).getTime()) / 60000);
  if (mins < 1) return "just now";
  if (mins < 60) return mins + "m ago";
  const hrs = Math.floor(mins / 60);
  if (hrs < 24) return hrs + "h ago";
  return Math.floor(hrs / 24) + "d ago";
}

function renderFilters() {
  const customers = [];
  sessions.forEach(function (s) {
    const c = s.customer_id || "unbound";
    if (customers.indexOf(c) === -1) customers.push(c);
  });
  customers.sort();
  const bar = document.getElementById("filters");
  bar.innerHTML = "";
  const all = el("button", filter === null ? "on" : "", "All (" + sessions.length + ")");
  all.onclick = function () { filter = null; renderList(); };
  bar.appendChild(all);
  customers.forEach(function (c) {
    const n = sessions.filter(function (s) { return (s.customer_id || "unbound") === c; }).length;
    const b = el("button", filter === c ? "on" : "", c + " (" + n + ")");
    b.onclick = function () { filter = c; renderList(); };
    bar.appendChild(b);
  });
}

function renderList() {
  renderFilters();
  const box = document.getElementById("items");
  box.innerHTML = "";
  const rows = sessions.filter(function (s) {
    return filter === null || (s.customer_id || "unbound") === filter;
  });
  if (!rows.length) { box.appendChild(el("div", "empty", "No conversations yet.")); return; }

  let lastGroup = null;
  rows.forEach(function (s) {
    const group = s.customer_id || "No customer bound";
    if (group !== lastGroup) { box.appendChild(el("div", "grouphead", group)); lastGroup = group; }

    const item = el("div", "item" + (selected === s.session_id ? " on" : ""));
    item.appendChild(el("div", "id", s.session_id));
    const meta = el("div", "meta");
    meta.appendChild(el("span", "", s.turns + (s.turns === 1 ? " turn" : " turns")));
    meta.appendChild(el("span", "", ago(s.last_seen)));
    if (s.escalated) meta.appendChild(el("span", "esc", "escalated: " + (s.escalation_reasons[0] || "?")));
    if (s.blocked_calls) meta.appendChild(el("span", "blk", s.blocked_calls + " blocked"));
    if (s.redrafts) meta.appendChild(el("span", "blk", s.redrafts + " redrafted"));
    item.appendChild(meta);
    item.onclick = function () { selected = s.session_id; renderList(); open(s); };
    box.appendChild(item);
  });
}

async function open(s) {
  const bar = document.getElementById("sumbar");
  const pane = document.getElementById("transcript");
  bar.textContent = "Loading " + s.session_id + "…";
  pane.innerHTML = "";
  try {
    const data = await (await fetch("/session/" + encodeURIComponent(s.session_id))).json();
    bar.innerHTML = "";
    [s.session_id, (s.customer_id || "no customer bound"),
     s.turns + " turns", "mean " + Math.round(s.mean_ms) + " ms",
     (s.models || []).join(", ")].forEach(function (t) {
      if (t) bar.appendChild(el("span", "", t));
    });
    if (!data.exists) { pane.appendChild(el("div", "empty",
      "No transcript stored — this session predates message checkpointing, or the store was cleared.")); return; }
    (data.turns || []).forEach(function (t) {
      const wrap = el("div", "turn " + (t.role === "user" ? "me" : "bot"));
      wrap.appendChild(el("div", "who", t.role === "user" ? "Customer" : "Trendly"));
      wrap.appendChild(el("div", "bubble", t.content));
      if ((t.tools || []).length) {
        const row = el("div", "trace");
        t.tools.forEach(function (c) {
          const r = c.result || {};
          if (r.blocked) { row.appendChild(el("span", "chip block", c.tool + " · blocked")); return; }
          row.appendChild(el("span", r.outcome ? "chip out" : "chip",
            c.tool + (r.outcome ? " · " + r.outcome : "")));
        });
        wrap.appendChild(row);
      }
      pane.appendChild(wrap);
    });
  } catch (e) {
    pane.appendChild(el("div", "empty", "Could not load this conversation."));
  }
}

async function load() {
  try {
    const data = await (await fetch("/sessions")).json();
    sessions = data.sessions || [];
  } catch (e) { sessions = []; }
  renderList();
}
load();
</script></body></html>
"""
