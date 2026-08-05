"""
Per-turn event log and metric aggregation.

The premise of the assignment is that ~70% of Trendly's 2,000 daily chats are
repetitive. An agent that handles them is only worth anything if someone can
show it moved that number, so every turn is recorded as one JSONL line and
aggregated into a deflection rate, escalation breakdown, and latency profile.

JSONL rather than a database on purpose: append-only, survives a crash
mid-write, trivially greppable, and a real deployment would ship these lines
to whatever the client already uses (Datadog, BigQuery) rather than querying
them here.
"""

import json
import os
import threading
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

LOG_PATH = Path(os.environ.get("EVENT_LOG", "logs/events.jsonl"))
_LOCK = threading.Lock()

# Reasons a turn ended with a human handoff, ordered roughly by how much they
# should worry an ops team: the first three are policy-mandated and healthy,
# the rest indicate the agent couldn't cope.
EXPECTED_ESCALATIONS = {
    "lost_parcel_claim",
    "damaged_within_report_window",
    "cod_bank_details",
    "out_of_policy_request",
}


def log_turn(event: dict) -> None:
    """Append one turn to the event log. Never raises — telemetry must not
    be able to take down a customer conversation."""
    try:
        event = {"ts": datetime.now(timezone.utc).isoformat(), **event}
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(event, ensure_ascii=False)
        with _LOCK, open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception as e:  # pragma: no cover
        print(f"[observability] failed to log turn: {e!r}")


def read_events(limit: int | None = None) -> list[dict]:
    if not LOG_PATH.exists():
        return []
    events = []
    with open(LOG_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # a torn line from a crash shouldn't break /metrics
    return events[-limit:] if limit else events


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(int(round((pct / 100) * (len(ordered) - 1))), len(ordered) - 1)
    return round(ordered[idx], 1)


def summarize(events: list[dict] | None = None) -> dict:
    """Aggregate the event log into the numbers an ops lead would ask for."""
    events = read_events() if events is None else events
    total = len(events)
    if total == 0:
        return {
            "turns": 0,
            "sessions": 0,
            "deflection_rate": None,
            "note": "No turns recorded yet.",
        }

    escalated = [e for e in events if e.get("escalated")]
    handled = total - len(escalated)

    tool_counter: Counter = Counter()
    outcome_counter: Counter = Counter()
    escalation_counter: Counter = Counter()
    violation_counter: Counter = Counter()
    model_counter: Counter = Counter()

    latencies, blocked, fallbacks, revalidations = [], 0, 0, 0

    for e in events:
        tool_counter.update(e.get("tools") or [])
        outcome_counter.update(e.get("outcomes") or [])
        violation_counter.update(e.get("validation_violations") or [])
        if e.get("model_used"):
            model_counter[e["model_used"]] += 1
        if e.get("latency_ms") is not None:
            latencies.append(e["latency_ms"])
        blocked += e.get("blocked_calls", 0)
        fallbacks += 1 if e.get("fallback_used") else 0
        revalidations += e.get("validation_retries", 0)
        if e.get("escalated"):
            escalation_counter[e.get("escalation_reason") or "unspecified"] += 1

    expected = sum(v for k, v in escalation_counter.items() if k in EXPECTED_ESCALATIONS)
    unexpected = sum(escalation_counter.values()) - expected

    return {
        "turns": total,
        "sessions": len({e.get("session_id") for e in events if e.get("session_id")}),
        "handled_without_human": handled,
        "escalated": len(escalated),
        # The headline number: share of turns the agent closed itself.
        "deflection_rate": round(handled / total, 3),
        "escalations": {
            "by_reason": dict(escalation_counter.most_common()),
            "policy_mandated": expected,
            "agent_limitation": unexpected,
        },
        "tools": dict(tool_counter.most_common()),
        "eligibility_outcomes": dict(outcome_counter.most_common()),
        "guardrails": {
            "blocked_tool_calls": blocked,
            "reply_validation_retries": revalidations,
            "validation_violations": dict(violation_counter.most_common()),
        },
        "models": {
            "turns_by_model": dict(model_counter.most_common()),
            "fallback_turns": fallbacks,
        },
        "latency_ms": {
            "p50": _percentile(latencies, 50),
            "p95": _percentile(latencies, 95),
            "max": round(max(latencies), 1) if latencies else 0.0,
        },
    }


def reset() -> None:
    """Clear the log. Used by the A/B harness between configurations."""
    if LOG_PATH.exists():
        LOG_PATH.unlink()
