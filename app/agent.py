"""
LangGraph orchestration.

Guardrails are enforced in code, not just prompted. A blocked call still
returns a ToolMessage explaining why, so the model self-corrects instead of
stalling:

1. check_return_eligibility requires a prior successful lookup_order on the
   same order_id — stops the model inventing an item_id.
2. initiate_return requires a prior check_return_eligibility on the same
   order_id + item_id whose outcome permits action, AND the resolution must
   match that outcome (an exchange_only item cannot be refunded).
3. Session-scoped customer binding: the first order looked up in a session
   binds that session to its customer. Any later lookup of an order belonging
   to a different customer is refused in code. Prompt Rule 4 alone did not
   hold — the model disclosed another customer's order contents when asked
   directly, so this moved into the graph.

Also handles consecutive tool-failure recovery, malformed tool-call retry
with backoff, and max-iteration escalation. State persists per session via
SqliteSaver keyed on session_id.
"""

import json
import os
import re
import sqlite3
import threading
import time
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from app.cassettes import CassetteLLM
from app.observability import log_turn
from app.prompts import SYSTEM_PROMPT
from app.tools import ALL_TOOLS, _find_order
from app.validation import correction_prompt, validate_reply

load_dotenv()

# ---------- Model configuration ----------
# The graph is provider-agnostic: it only needs something that implements
# .bind_tools() and .invoke(). Swapping providers is a config change, not a
# code change, which also makes it cheap to fall back when one is rate-limited.
#
# Defaults are Gemini 3.6 Flash (better tool-calling reliability and ~17%
# fewer output tokens than the previous Flash generation) with 3.5 Flash-Lite
# as the cheap fallback. Set LLM_PROVIDER=groq to switch back.
PROVIDER = os.environ.get("LLM_PROVIDER", "gemini").strip().lower()

_DEFAULTS = {
    # Flash-Lite is the primary: this workload is routing plus tool selection
    # over a one-page policy, which is squarely what it is built for, and its
    # free-tier limits are far higher than 3.6 Flash's 20 requests/day. 3.6
    # Flash sits behind it for anything Lite can't handle.
    #
    # The GA model strings are pinned rather than the "-latest" aliases: those
    # hot-swap and may point at a preview or experimental release, which is not
    # what you want behind a live URL. Override via PRIMARY_MODEL if you do
    # want auto-updating (e.g. gemini-flash-lite-latest).
    "gemini": ("gemini-3.5-flash-lite", "gemini-3.6-flash"),
    "groq": ("llama-3.3-70b-versatile", "llama-3.1-8b-instant"),
}

# Flash-Lite defaults to thinking_level "minimal", which Google explicitly
# warns causes premature tool termination in agentic workflows — precisely the
# failure where the model answers without calling check_return_eligibility.
THINKING_LEVEL = os.environ.get("GEMINI_THINKING_LEVEL", "medium")

# Free tiers cap requests per MINUTE (15/min on Flash-Lite), not per day, so a
# burst fails while the quota is nearly untouched. A minimum spacing between
# model calls converts a wall of 429s into a slightly slower run. Off by
# default (0); scripts that fire many turns back to back set it.
MIN_CALL_INTERVAL_MS = int(os.environ.get("MIN_MODEL_INTERVAL_MS", "0"))
_last_call_at = 0.0
_call_lock = threading.Lock()


def _throttle() -> None:
    global _last_call_at
    if MIN_CALL_INTERVAL_MS <= 0:
        return
    with _call_lock:
        wait = (_last_call_at + MIN_CALL_INTERVAL_MS / 1000) - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        _last_call_at = time.monotonic()


def _suggested_retry_seconds(error: Exception) -> float | None:
    """
    Providers tell us how long to wait — Gemini in a retryDelay field, Groq in
    the message text. Honouring it beats a fixed backoff, which either stalls
    too long or retries far too early.
    """
    text = str(error)
    for pattern in (r"'retryDelay':\s*'(\d+(?:\.\d+)?)s'",
                    r"retry in (\d+(?:\.\d+)?)\s*s"):
        match = re.search(pattern, text)
        if match:
            return float(match.group(1))
    return None
_default_primary, _default_fallback = _DEFAULTS.get(PROVIDER, _DEFAULTS["gemini"])

PRIMARY_MODEL = os.environ.get("PRIMARY_MODEL", _default_primary)
FALLBACK_MODEL = os.environ.get("FALLBACK_MODEL", _default_fallback)
MODEL = PRIMARY_MODEL  # reported by /health
MAX_ITERATIONS = 6
MAX_RETRY_WAIT_S = float(os.environ.get("MAX_RETRY_WAIT_S", "12"))
DB_PATH = os.environ.get("SESSIONS_DB", "sessions.db")


def _build_llm(model_name: str):
    """
    Construct a tool-bound chat model for the configured provider.

    max_retries=0 is deliberate: the SDKs retry internally by default, which
    silently multiplies against agent_node's own retry loop and turns a
    rate-limit into a very long stall. Retry policy lives in one place.
    """
    if PROVIDER == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("Set GEMINI_API_KEY (or GOOGLE_API_KEY) in .env")
        # temperature/top_p/top_k are deprecated on Gemini 3.x — omitted rather
        # than passed, so the model uses its own defaults.
        kwargs = {"model": model_name, "api_key": api_key, "max_retries": 0}
        try:
            client = ChatGoogleGenerativeAI(thinking_level=THINKING_LEVEL, **kwargs)
        except TypeError:
            # Older langchain-google-genai doesn't expose thinking_level. Fall
            # back rather than hard-fail; behaviour degrades, nothing breaks.
            print("[agent] thinking_level unsupported by this SDK version — omitting")
            client = ChatGoogleGenerativeAI(**kwargs)
        return CassetteLLM(client.bind_tools(ALL_TOOLS), model_name)

    if PROVIDER == "groq":
        from langchain_groq import ChatGroq

        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("Set GROQ_API_KEY in .env")
        return CassetteLLM(
            ChatGroq(
                model=model_name, api_key=api_key, temperature=0.1, max_retries=0
            ).bind_tools(ALL_TOOLS),
            model_name,
        )

    raise RuntimeError(f"Unknown LLM_PROVIDER {PROVIDER!r} — use 'gemini' or 'groq'")


primary_llm = _build_llm(PRIMARY_MODEL)
fallback_llm = _build_llm(FALLBACK_MODEL)


def configure_models(provider: str, primary: str, fallback: str) -> None:
    """
    Rebuild the models at runtime. agent_node resolves these as module globals
    on every call, so reassigning them swaps the provider without touching the
    compiled graph. Used by scripts/ab_compare.py to run the same scenarios
    across providers in one process.
    """
    global PROVIDER, PRIMARY_MODEL, FALLBACK_MODEL, MODEL, primary_llm, fallback_llm
    PROVIDER = provider.strip().lower()
    PRIMARY_MODEL, FALLBACK_MODEL, MODEL = primary, fallback, primary
    primary_llm = _build_llm(primary)
    fallback_llm = _build_llm(fallback)

TOOL_MAP = {t.name: t for t in ALL_TOOLS}

# Outcomes from check_return_eligibility that permit initiate_return, mapped
# to the resolutions each one allows.
# Tools whose results ground a factual claim. Once one has run, later turns in
# the same conversation may refer back to what it established.
GROUNDING_TOOL_NAMES = {
    "search_policy", "lookup_order", "check_return_eligibility",
    "apply_delayed_credit", "initiate_return",
}

ALLOWED_RESOLUTIONS = {
    "eligible_refund": {"refund", "exchange"},
    "exchange_only": {"exchange"},
}


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    session_customer_id: str | None       # persists — binds session to a customer
    session_grounded: bool                # persists — a grounding tool has run
    looked_up_orders: list[str]           # persists — order_ids seen via lookup_order
    eligibility_outcomes: dict            # persists — "order|item" -> outcome
    consecutive_failures: int             # per turn
    last_failed_tool: str | None          # per turn
    escalated: bool                       # per turn
    iteration: int                        # per turn
    validation_retries: int               # per turn
    last_violations: list                 # per turn — surfaced in diagnostics
    model_used: str | None                # per turn — which model answered
    fallback_used: bool                   # per turn
    escalation_reason: str | None         # per turn — why a human was needed


def _safe_json(text):
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return text


def _as_text(content) -> str:
    """
    Flatten a message's content to a plain string.

    Providers differ here: Groq returns a string, while Gemini returns a list
    of content blocks — [{"type": "text", "text": ..., "extras": {...}}] —
    carrying thought signatures alongside the visible text. The API contract
    is a string, so the difference is normalised at the boundary rather than
    leaking the provider's shape into the response model.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "\n".join(p for p in parts if p).strip()
    return str(content) if content is not None else ""


def _blocked(message: str, **extra) -> dict:
    """Uniform shape for guardrail rejections, so tests can detect them."""
    return {"blocked": True, "error": f"Blocked: {message}", **extra}


# ---------- NODE: agent ----------

def agent_node(state: AgentState) -> dict:
    messages = state["messages"]
    if not any(isinstance(m, SystemMessage) for m in messages):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages

    iteration = state.get("iteration", 0) + 1

    # Two failure modes to survive here:
    #   1. Groq/Llama occasionally emits a raw "<function=name{...}>" string
    #      instead of a structured tool call, which the API rejects.
    #   2. The free tier's daily token cap returns 429 for the rest of the day.
    # So: retry the primary with backoff, then try the fallback model, and only
    # escalate if both are exhausted. A degraded answer beats no answer, and an
    # honest handoff beats a stack trace.
    last_error = None
    for model_name, model in (
        (PRIMARY_MODEL, primary_llm),
        (FALLBACK_MODEL, fallback_llm),
    ):
        for attempt in range(3 if model is primary_llm else 2):
            try:
                _throttle()
                response = model.invoke(messages)
                is_fallback = model is not primary_llm
                if is_fallback:
                    print(f"[agent_node] answered using fallback model {model_name}")
                return {
                    "messages": [response],
                    "iteration": iteration,
                    "model_used": model_name,
                    "fallback_used": is_fallback,
                }
            except Exception as e:
                last_error = e
                # Rate-limit errors carry a full JSON blob; one line is enough.
                brief = str(e).split("\n")[0][:160]
                print(f"[agent_node] {model_name} attempt {attempt + 1} failed: {brief}")

                suggested = _suggested_retry_seconds(e)
                if suggested is not None and suggested <= MAX_RETRY_WAIT_S:
                    time.sleep(suggested + 0.3)
                elif suggested is not None:
                    # Waiting a minute mid-conversation is worse than trying
                    # the other model, which has a separate quota.
                    break
                else:
                    time.sleep(2 ** attempt)

    escalation = TOOL_MAP["escalate_to_human"].invoke({
        "summary": (
            "The assistant could not generate a response — both the primary "
            f"and fallback models failed. Last error: {last_error!r}. The "
            "customer's request is in the transcript above and has NOT been "
            "actioned; it needs handling from scratch."
        ),
        "reason": "model_unavailable",
    })
    return {
        "messages": [AIMessage(content=escalation["customer_message"])],
        "escalated": True,
        "escalation_reason": "model_unavailable",
        "iteration": iteration,
    }


# ---------- NODE: tools ----------

def tool_node(state: AgentState) -> dict:
    last_message = state["messages"][-1]
    tool_calls = getattr(last_message, "tool_calls", None) or []

    session_customer_id = state.get("session_customer_id")
    session_grounded = bool(state.get("session_grounded"))
    looked_up_orders = set(state.get("looked_up_orders") or [])
    eligibility_outcomes = dict(state.get("eligibility_outcomes") or {})
    consecutive_failures = state.get("consecutive_failures", 0)
    last_failed_tool = state.get("last_failed_tool")
    escalated = state.get("escalated", False)
    escalation_reason = state.get("escalation_reason")

    tool_messages = []

    for tc in tool_calls:
        tool_name = tc["name"]
        args = tc["args"] or {}
        tool_call_id = tc["id"]
        result = None

        # --- Guardrail 3: cross-customer access ---
        if tool_name == "lookup_order":
            order = _find_order(args.get("order_id", ""))
            if (
                order
                and session_customer_id
                and order["customer_id"] != session_customer_id
            ):
                result = _blocked(
                    "this order belongs to a different customer than the one "
                    "in this conversation. Do not confirm it exists, do not "
                    "describe its contents, and do not say who placed it. "
                    "Tell the customer you can only discuss orders on their "
                    "own account.",
                    customer_message=(
                        "I can only look up orders placed on your own account, "
                        "so I'm not able to pull that one up."
                    ),
                )

        # --- Guardrail 1: the order must be established before eligibility ---
        # Originally this bounced the call back with "call lookup_order first".
        # That relies on the model reading a correction and retrying, which it
        # frequently doesn't — it retried the same blocked call and then
        # escalated. The precondition is mechanically satisfiable: the graph
        # has the order ID, so it resolves the order itself and proceeds. The
        # safety property the guardrail existed for (no invented item_ids) is
        # now enforced where it belongs, in _find_item, which returns the real
        # line items instead of guessing.
        if result is None and tool_name == "check_return_eligibility":
            order_key = args.get("order_id", "").strip().upper()
            if order_key not in looked_up_orders:
                order = _find_order(order_key)
                if order is None:
                    pass  # let the tool return its own order_not_found verdict
                elif session_customer_id and order["customer_id"] != session_customer_id:
                    result = _blocked(
                        "this order belongs to a different customer than the "
                        "one in this conversation.",
                        customer_message=(
                            "I can only help with orders placed on your own "
                            "account, so I can't check that one."
                        ),
                    )
                else:
                    looked_up_orders.add(order_key)
                    if session_customer_id is None:
                        session_customer_id = order["customer_id"]

        # --- Guardrail 2: eligibility before action, and matching resolution ---
        if result is None and tool_name == "initiate_return":
            key = f"{args.get('order_id', '').strip().upper()}|{args.get('item_id', '').strip().upper()}"
            outcome = eligibility_outcomes.get(key)
            resolution = (args.get("resolution") or "refund").strip().lower()

            if outcome is None:
                result = _blocked(
                    "call check_return_eligibility for this exact order_id and "
                    "item_id first."
                )
            elif outcome not in ALLOWED_RESOLUTIONS:
                result = _blocked(
                    f"check_return_eligibility returned '{outcome}' for this "
                    "item, which does not permit a return. Explain that "
                    "outcome to the customer instead."
                )
            elif resolution not in ALLOWED_RESOLUTIONS[outcome]:
                result = _blocked(
                    f"this item's outcome is '{outcome}', which allows only: "
                    f"{sorted(ALLOWED_RESOLUTIONS[outcome])}. A final-sale item "
                    "cannot be refunded — offer a size exchange and ask which "
                    "size they need."
                )

        # --- Execute ---
        if result is None:
            tool_obj = TOOL_MAP.get(tool_name)
            if tool_obj is None:
                result = {"error": f"Unknown tool: {tool_name}"}
            else:
                try:
                    result = tool_obj.invoke(args)
                    consecutive_failures = 0
                    if tool_name in GROUNDING_TOOL_NAMES:
                        session_grounded = True

                    if tool_name == "lookup_order" and result.get("found"):
                        order_id = args.get("order_id", "").strip().upper()
                        looked_up_orders.add(order_id)
                        # First order in the session binds it to that customer.
                        if session_customer_id is None:
                            order = _find_order(order_id)
                            if order:
                                session_customer_id = order["customer_id"]

                    if tool_name == "check_return_eligibility":
                        key = (
                            f"{args.get('order_id', '').strip().upper()}|"
                            f"{args.get('item_id', '').strip().upper()}"
                        )
                        eligibility_outcomes[key] = result.get("outcome")

                    if tool_name == "escalate_to_human":
                        escalated = True

                except Exception as e:
                    result = {"error": f"Tool execution failed: {e}"}
                    consecutive_failures = (
                        consecutive_failures + 1 if tool_name == last_failed_tool else 1
                    )
                    last_failed_tool = tool_name

        tool_messages.append(
            ToolMessage(
                content=json.dumps(result, ensure_ascii=False),
                tool_call_id=tool_call_id,
                name=tool_name,
            )
        )

    return {
        "messages": tool_messages,
        "session_customer_id": session_customer_id,
        "session_grounded": session_grounded,
        "looked_up_orders": sorted(looked_up_orders),
        "eligibility_outcomes": eligibility_outcomes,
        "consecutive_failures": consecutive_failures,
        "last_failed_tool": last_failed_tool,
        "escalated": escalated,
        "escalation_reason": escalation_reason,
    }


# ---------- NODE: forced escalation ----------

def force_escalate_node(state: AgentState) -> dict:
    if state.get("consecutive_failures", 0) >= 2:
        summary = (
            f"The '{state.get('last_failed_tool')}' tool failed repeatedly, so "
            "the assistant could not complete the request. See the transcript "
            "above for what the customer asked."
        )
        reason = "repeated_tool_failure"
    else:
        summary = (
            "The assistant reached its reasoning-step limit without resolving "
            "the request. See the transcript above; it may need a person to "
            "untangle what the customer is asking for."
        )
        reason = "max_iterations_exceeded"

    escalation = TOOL_MAP["escalate_to_human"].invoke(
        {"summary": summary, "reason": reason}
    )
    return {
        "messages": [AIMessage(content=escalation["customer_message"])],
        "escalated": True,
        "escalation_reason": reason,
    }


def _turn_tool_results(messages) -> list[dict]:
    """Tool results produced since the customer's most recent message."""
    turn: list[dict] = []
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            # The validator's corrective message is a HumanMessage so that
            # every provider accepts it mid-conversation, but it is not the
            # customer speaking — skip it, or a redraft would see no tools.
            if (m.additional_kwargs or {}).get("validation_correction"):
                continue
            break
        if isinstance(m, ToolMessage):
            turn.append({"tool": m.name, "result": _safe_json(_as_text(m.content))})
    return list(reversed(turn))


# ---------- NODE: validate (checks the drafted reply before it is sent) ----------

def validate_node(state: AgentState) -> dict:
    """
    The guardrails in tool_node constrain what the agent may DO. This
    constrains what it may SAY. Every wrong answer this project produced came
    from the model describing a correct tool verdict incorrectly, so the reply
    is checked against the turn's tool results before the customer sees it.
    """
    messages = state["messages"]
    reply = _as_text(messages[-1].content)
    tool_results = _turn_tool_results(messages)

    violations = validate_reply(
        reply,
        tool_results,
        state.get("session_customer_id"),
        session_grounded=bool(state.get("session_grounded")),
    )
    if not violations:
        return {"last_violations": []}

    retries = state.get("validation_retries", 0)
    codes = [v["code"] for v in violations]
    print(f"[validate_node] rejected reply (attempt {retries + 1}): {codes}")

    if retries >= 1:
        # One corrective attempt already failed. Escalating beats sending a
        # reply we know to be wrong.
        escalation = TOOL_MAP["escalate_to_human"].invoke({
            "summary": (
                "The assistant drafted a reply that failed automated "
                f"validation twice ({', '.join(codes)}). The customer's "
                "request is in the transcript and has NOT been actioned."
            ),
            "reason": "reply_validation_failed",
        })
        return {
            "messages": [AIMessage(content=escalation["customer_message"])],
            "escalated": True,
            "escalation_reason": "reply_validation_failed",
            "last_violations": violations,
            "validation_retries": retries + 1,
        }

    return {
        "messages": [
            HumanMessage(
                content=correction_prompt(violations),
                additional_kwargs={"validation_correction": True},
            )
        ],
        "validation_retries": retries + 1,
        "last_violations": violations,
    }


# ---------- ROUTING ----------

def route_after_agent(state: AgentState) -> str:
    if getattr(state["messages"][-1], "tool_calls", None):
        return "tools"
    return "validate"


def route_after_validation(state: AgentState) -> str:
    """Clean reply -> done. Rejected once -> redraft. Rejected twice -> escalate."""
    if not state.get("last_violations"):
        return END
    if state.get("escalated"):
        return END
    return "agent"


def route_after_tools(state: AgentState) -> str:
    if state.get("consecutive_failures", 0) >= 2:
        return "force_escalate"
    if state.get("iteration", 0) >= MAX_ITERATIONS:
        return "force_escalate"
    return "agent"


# ---------- GRAPH ----------

graph_builder = StateGraph(AgentState)
graph_builder.add_node("agent", agent_node)
graph_builder.add_node("tools", tool_node)
graph_builder.add_node("force_escalate", force_escalate_node)
graph_builder.add_node("validate", validate_node)

graph_builder.set_entry_point("agent")
graph_builder.add_conditional_edges(
    "agent", route_after_agent, {"tools": "tools", "validate": "validate"}
)
graph_builder.add_conditional_edges(
    "validate", route_after_validation, {"agent": "agent", END: END}
)
graph_builder.add_conditional_edges(
    "tools", route_after_tools, {"agent": "agent", "force_escalate": "force_escalate"}
)
graph_builder.add_edge("force_escalate", END)

_conn = sqlite3.connect(DB_PATH, check_same_thread=False)
checkpointer = SqliteSaver(_conn)
graph = graph_builder.compile(checkpointer=checkpointer)


# ---------- ENTRY POINT ----------

def run_agent(session_id: str, user_message: str) -> dict:
    config = {"configurable": {"thread_id": session_id}}
    started = time.perf_counter()

    prior_state = graph.get_state(config)
    prior_count = len(prior_state.values.get("messages", [])) if prior_state.values else 0

    result = graph.invoke(
        {
            "messages": [HumanMessage(content=user_message)],
            # Per-turn counters reset; session_customer_id, looked_up_orders
            # and eligibility_outcomes are deliberately not passed here so the
            # checkpointer preserves them across the conversation.
            "iteration": 0,
            "consecutive_failures": 0,
            "last_failed_tool": None,
            "escalated": False,
            "escalation_reason": None,
            "validation_retries": 0,
            "last_violations": [],
            "model_used": None,
            "fallback_used": False,
        },
        config=config,
    )

    new_messages = result["messages"][prior_count:]
    tool_calls_made = [
        {"tool": m.name, "result": _safe_json(_as_text(m.content))}
        for m in new_messages
        if isinstance(m, ToolMessage)
    ]

    reply = _as_text(result["messages"][-1].content)
    latency_ms = round((time.perf_counter() - started) * 1000, 1)

    blocked_calls = sum(
        1 for c in tool_calls_made
        if isinstance(c["result"], dict) and c["result"].get("blocked")
    )
    outcomes = [
        c["result"]["outcome"] for c in tool_calls_made
        if isinstance(c["result"], dict) and c["result"].get("outcome")
    ]
    violations = [v["code"] for v in (result.get("last_violations") or [])]

    # Everything the graph is holding for this conversation. Surfaced so the
    # UI can show multi-turn state directly rather than asserting it exists.
    state = {
        "session_customer_id": result.get("session_customer_id"),
        "looked_up_orders": result.get("looked_up_orders") or [],
        "eligibility_outcomes": result.get("eligibility_outcomes") or {},
        "iterations_this_turn": result.get("iteration", 0),
        "validation_retries": result.get("validation_retries", 0),
    }
    diagnostics = {
        "model_used": result.get("model_used"),
        "fallback_used": bool(result.get("fallback_used")),
        "latency_ms": latency_ms,
        "blocked_calls": blocked_calls,
        "validation_violations": violations,
        "escalation_reason": result.get("escalation_reason"),
    }

    log_turn({
        "session_id": session_id,
        "provider": PROVIDER,
        "model_used": result.get("model_used"),
        "fallback_used": bool(result.get("fallback_used")),
        "tools": [c["tool"] for c in tool_calls_made],
        "outcomes": outcomes,
        "blocked_calls": blocked_calls,
        "validation_violations": violations,
        "validation_retries": result.get("validation_retries", 0),
        "escalated": bool(result.get("escalated")),
        "escalation_reason": result.get("escalation_reason"),
        "latency_ms": latency_ms,
        "reply_chars": len(reply),
    })

    return {
        "reply": reply,
        "tool_calls_made": tool_calls_made,
        "escalated": bool(result.get("escalated", False)),
        "state": state,
        "diagnostics": diagnostics,
    }
