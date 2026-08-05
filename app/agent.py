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
import sqlite3
import time
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_groq import ChatGroq
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from app.prompts import SYSTEM_PROMPT
from app.tools import ALL_TOOLS, _find_order

load_dotenv()

# llama-3.1-8b-instant does not follow this prompt reliably: it paraphrases
# tool verdicts into their opposite and ignores the data-disclosure rules.
# 70b-versatile is the default; 8b stays as a fallback for when the 70b daily
# token cap is hit, since a degraded answer beats no answer.
PRIMARY_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
FALLBACK_MODEL = os.environ.get("GROQ_FALLBACK_MODEL", "llama-3.1-8b-instant")
MODEL = PRIMARY_MODEL  # reported by /health
MAX_ITERATIONS = 6
DB_PATH = os.environ.get("SESSIONS_DB", "sessions.db")

_api_key = os.environ["GROQ_API_KEY"]
primary_llm = ChatGroq(model=PRIMARY_MODEL, api_key=_api_key, temperature=0.1).bind_tools(ALL_TOOLS)
fallback_llm = ChatGroq(model=FALLBACK_MODEL, api_key=_api_key, temperature=0.1).bind_tools(ALL_TOOLS)

TOOL_MAP = {t.name: t for t in ALL_TOOLS}

# Outcomes from check_return_eligibility that permit initiate_return, mapped
# to the resolutions each one allows.
ALLOWED_RESOLUTIONS = {
    "eligible_refund": {"refund", "exchange"},
    "exchange_only": {"exchange"},
}


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    session_customer_id: str | None       # persists — binds session to a customer
    looked_up_orders: list[str]           # persists — order_ids seen via lookup_order
    eligibility_outcomes: dict            # persists — "order|item" -> outcome
    consecutive_failures: int             # per turn
    last_failed_tool: str | None          # per turn
    escalated: bool                       # per turn
    iteration: int                        # per turn


def _safe_json(text):
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return text


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
                response = model.invoke(messages)
                if model is not primary_llm:
                    print(f"[agent_node] answered using fallback model {model_name}")
                return {"messages": [response], "iteration": iteration}
            except Exception as e:
                last_error = e
                print(f"[agent_node] {model_name} attempt {attempt + 1} failed: {e!r}")
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
        "iteration": iteration,
    }


# ---------- NODE: tools ----------

def tool_node(state: AgentState) -> dict:
    last_message = state["messages"][-1]
    tool_calls = getattr(last_message, "tool_calls", None) or []

    session_customer_id = state.get("session_customer_id")
    looked_up_orders = set(state.get("looked_up_orders") or [])
    eligibility_outcomes = dict(state.get("eligibility_outcomes") or {})
    consecutive_failures = state.get("consecutive_failures", 0)
    last_failed_tool = state.get("last_failed_tool")
    escalated = state.get("escalated", False)

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

        # --- Guardrail 1: lookup before eligibility ---
        if result is None and tool_name == "check_return_eligibility":
            if args.get("order_id", "").strip().upper() not in looked_up_orders:
                result = _blocked(
                    "call lookup_order for this exact order_id first, so you "
                    "use the order's real item_id (SKU). Do not guess one."
                )

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
        "looked_up_orders": sorted(looked_up_orders),
        "eligibility_outcomes": eligibility_outcomes,
        "consecutive_failures": consecutive_failures,
        "last_failed_tool": last_failed_tool,
        "escalated": escalated,
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
    }


# ---------- ROUTING ----------

def route_after_agent(state: AgentState) -> str:
    return "tools" if getattr(state["messages"][-1], "tool_calls", None) else END


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

graph_builder.set_entry_point("agent")
graph_builder.add_conditional_edges("agent", route_after_agent, {"tools": "tools", END: END})
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
        },
        config=config,
    )

    new_messages = result["messages"][prior_count:]
    tool_calls_made = [
        {"tool": m.name, "result": _safe_json(m.content)}
        for m in new_messages
        if isinstance(m, ToolMessage)
    ]

    return {
        "reply": result["messages"][-1].content,
        "tool_calls_made": tool_calls_made,
        "escalated": result.get("escalated", False),
    }
