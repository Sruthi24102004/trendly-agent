"""
FastAPI app: POST /chat for the agent, GET /health for uptime checks, and a
minimal chat page at / for manual testing and the demo video.
"""

import os

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from app.agent import MODEL, PROVIDER, run_agent

app = FastAPI(title="Trendly Support Agent")


class ChatRequest(BaseModel):
    session_id: str
    message: str


class ChatResponse(BaseModel):
    reply: str
    tool_calls_made: list
    escalated: bool


@app.get("/health")
def health():
    """Cheap liveness probe — no model call, safe for uptime pings."""
    return {
        "status": "ok",
        "provider": PROVIDER,
        "model": MODEL,
        "clock_override": os.environ.get("TRENDLY_NOW"),
    }


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="message must not be empty")
    try:
        return run_agent(session_id=req.session_id, user_message=req.message)
    except Exception as e:
        # Safety net: return clean JSON rather than crashing the ASGI app, so
        # the frontend can show something useful.
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/", response_class=HTMLResponse)
def index():
    return HTML_PAGE


HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
  <title>Trendly Support</title>
  <style>
    body { font-family: system-ui, sans-serif; max-width: 640px; margin: 40px auto; background: #0b0f14; color: #e6e6e6; }
    h2 { color: #fff; margin-bottom: 4px; }
    .sub { color: #6b7280; font-size: 13px; margin-bottom: 16px; }
    #chat { border: 1px solid #2a2f36; border-radius: 8px; height: 440px; overflow-y: auto; padding: 12px; margin-bottom: 12px; background: #11161c; }
    .msg { margin: 8px 0; padding: 8px 12px; border-radius: 8px; max-width: 82%; white-space: pre-wrap; }
    .user { background: #2563eb; color: white; margin-left: auto; }
    .bot { background: #1f2937; }
    .error { background: #7f1d1d; color: #fecaca; }
    .row { display: flex; }
    input { flex: 1; padding: 10px; border-radius: 6px; border: 1px solid #2a2f36; background: #11161c; color: white; }
    button { padding: 10px 16px; margin-left: 8px; border: none; border-radius: 6px; background: #2563eb; color: white; cursor: pointer; }
    button:disabled { opacity: .5; cursor: default; }
    .tag { font-size: 11px; color: #9ca3af; margin: 2px 0 10px 4px; font-family: ui-monospace, monospace; }
    .blocked { color: #f59e0b; }
  </style>
</head>
<body>
  <h2>Trendly Support Assistant</h2>
  <div class="sub">Tool calls are shown under each reply. Blocked calls are guardrail rejections.</div>
  <div id="chat"></div>
  <div class="row">
    <input id="input" placeholder="Ask about an order, return, or policy..." autofocus />
    <button id="send" onclick="send()">Send</button>
  </div>

  <script>
    const sessionId = "demo-" + Math.random().toString(36).slice(2);
    const chat = document.getElementById("chat");
    const input = document.getElementById("input");
    const button = document.getElementById("send");

    function addMessage(text, cls, tagHtml) {
      const div = document.createElement("div");
      div.className = "msg " + cls;
      div.textContent = text;
      chat.appendChild(div);
      if (tagHtml) {
        const t = document.createElement("div");
        t.className = "tag";
        t.innerHTML = tagHtml;
        chat.appendChild(t);
      }
      chat.scrollTop = chat.scrollHeight;
    }

    function describe(calls, escalated) {
      const parts = calls.map(c => {
        const r = c.result || {};
        if (r.blocked) return '<span class="blocked">' + c.tool + ' [blocked]</span>';
        return c.tool + (r.outcome ? " -> " + r.outcome : "");
      });
      if (escalated) parts.push("escalated");
      return parts.join(" &middot; ");
    }

    async function send() {
      const message = input.value.trim();
      if (!message) return;
      addMessage(message, "user");
      input.value = "";
      button.disabled = true;

      try {
        const res = await fetch("/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ session_id: sessionId, message })
        });

        if (!res.ok) {
          const err = await res.json().catch(() => ({ detail: res.statusText }));
          addMessage("Server error (" + res.status + "): " + (err.detail || "unknown"), "bot error");
          return;
        }

        const data = await res.json();
        addMessage(data.reply, "bot", describe(data.tool_calls_made, data.escalated));
      } catch (err) {
        addMessage("Request failed: " + err.message, "bot error");
      } finally {
        button.disabled = false;
        input.focus();
      }
    }

    input.addEventListener("keydown", e => { if (e.key === "Enter") send(); });
  </script>
</body>
</html>
"""
