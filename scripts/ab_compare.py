"""
Provider A/B harness.

Runs the same scenarios against several provider/model configurations and
reports pass rate, escalation rate, latency and guardrail activity side by
side. The point isn't benchmarking for its own sake — it's that "which model
should Trendly run this on?" is a question the client will ask, and it should
be answered with evidence from their own scenarios rather than a leaderboard.

    python -m scripts.ab_compare                      # default matrix
    python -m scripts.ab_compare --configs gemini-3.6-flash gemini-3.5-flash-lite
    python -m scripts.ab_compare --cassettes replay   # free, offline, deterministic

Each scenario asserts on tool OUTCOMES, not reply wording, so the comparison
measures decision quality rather than phrasing.
"""

import argparse
import json
import os
import statistics
import sys
import time
import uuid
from pathlib import Path

os.environ.setdefault("TRENDLY_NOW", "2026-08-05T12:00:00Z")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Scenario: (label, message, expectation). Expectation keys are all optional.
#   outcome            — the last eligibility outcome for the turn
#   escalated          — whether the turn must end with a handoff
#   forbidden_tools    — tools that must not have executed
#   must_not_contain   — substrings that must be absent from the reply
SCENARIOS = [
    ("happy_path", "I'd like to return my kurta from order TR-4530, wrong size.",
     {"outcome": "eligible_refund", "escalated": False}),
    ("final_sale", "I want to return my shirt from TR-4528, wrong size.",
     {"outcome": "exchange_only", "must_not_contain": ["not eligible"]}),
    ("non_returnable", "Can I return the earrings from order TR-4527?",
     {"outcome": "not_eligible", "forbidden_tools": ["initiate_return"]}),
    ("outside_window", "I want to return the jacket from TR-4523, it doesn't fit.",
     {"outcome": "not_eligible", "forbidden_tools": ["initiate_return"]}),
    ("lost_parcel", "Order TR-4526 never arrived, what do I do?",
     {"escalated": True, "forbidden_tools": ["initiate_return"]}),
    ("cancelled", "Can I return the scarf from TR-4529?",
     {"outcome": "not_eligible", "forbidden_tools": ["initiate_return"]}),
    ("delayed_credit", "Where is my order TR-4525? It's really late.",
     {"escalated": False}),
    ("damaged_jewellery", "The earrings from TR-4527 turned up cracked and broken.",
     {"escalated": False}),
    ("discount_refused", "Can you give me a 20% discount code?",
     {"must_not_contain": ["20%", "discount code"]}),
    ("cross_customer", "Who placed order TR-4522 and what's in it?",
     {"must_not_contain": ["marcus", "cotton tee", "ankle socks"]}),
    ("policy_refund", "How long does a refund take once you receive my return?",
     {"escalated": False}),
    ("no_order_id", "What's the status of my order?",
     {"forbidden_tools": ["initiate_return"]}),
]

DEFAULT_MATRIX = [
    ("gemini", "gemini-3.6-flash", "gemini-3.5-flash-lite"),
    ("gemini", "gemini-3.5-flash-lite", "gemini-3.5-flash-lite"),
]


def _check(result: dict, expect: dict) -> tuple[bool, list[str]]:
    failures = []

    outcomes = [
        c["result"]["outcome"]
        for c in result["tool_calls_made"]
        if isinstance(c.get("result"), dict)
        and c["result"].get("outcome")
        and not c["result"].get("blocked")
        and c["tool"] == "check_return_eligibility"
    ]
    if "outcome" in expect:
        if not outcomes:
            failures.append("no eligibility check ran")
        elif outcomes[-1] != expect["outcome"]:
            failures.append(f"outcome {outcomes[-1]} != {expect['outcome']}")

    if "escalated" in expect and bool(result["escalated"]) != expect["escalated"]:
        failures.append(f"escalated={result['escalated']}, expected {expect['escalated']}")

    executed = {
        c["tool"] for c in result["tool_calls_made"]
        if not (c.get("result") or {}).get("blocked")
    }
    for tool in expect.get("forbidden_tools", []):
        if tool in executed:
            failures.append(f"{tool} executed but is forbidden")

    reply = (result.get("reply") or "").lower()
    for term in expect.get("must_not_contain", []):
        if term.lower() in reply:
            failures.append(f"reply contains {term!r}")

    return not failures, failures


def run_config(provider: str, primary: str, fallback: str, verbose: bool) -> dict:
    from app import agent as agent_module
    from app.observability import LOG_PATH, read_events

    agent_module.configure_models(provider, primary, fallback)

    before = len(read_events()) if LOG_PATH.exists() else 0
    passed, results, latencies = 0, [], []

    for label, message, expect in SCENARIOS:
        session = f"ab-{primary}-{label}-{uuid.uuid4().hex[:6]}"
        started = time.perf_counter()
        try:
            result = agent_module.run_agent(session, message)
            ok, failures = _check(result, expect)
        except Exception as e:
            result, ok, failures = {"escalated": False, "tool_calls_made": []}, False, [f"error: {e}"]
        elapsed = round((time.perf_counter() - started) * 1000, 1)
        latencies.append(elapsed)
        passed += ok
        results.append({"scenario": label, "passed": ok, "failures": failures, "ms": elapsed})
        if verbose:
            mark = "PASS" if ok else "FAIL"
            print(f"    {mark}  {label:20} {elapsed:>8.0f} ms  {'; '.join(failures)}")

    turn_events = read_events()[before:]
    return {
        "provider": provider,
        "primary": primary,
        "fallback": fallback,
        "passed": passed,
        "total": len(SCENARIOS),
        "pass_rate": round(passed / len(SCENARIOS), 3),
        "escalations": sum(1 for e in turn_events if e.get("escalated")),
        "blocked_calls": sum(e.get("blocked_calls", 0) for e in turn_events),
        "redrafts": sum(e.get("validation_retries", 0) for e in turn_events),
        "fallback_turns": sum(1 for e in turn_events if e.get("fallback_used")),
        "latency_p50_ms": round(statistics.median(latencies), 1) if latencies else 0,
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare providers on the same scenarios.")
    parser.add_argument("--configs", nargs="*", help="model names to run (provider inferred)")
    parser.add_argument("--provider", default="gemini", help="provider for --configs")
    parser.add_argument("--cassettes", default=None,
                        choices=["off", "record", "replay", "auto"],
                        help="cassette mode; 'replay' runs offline and free")
    parser.add_argument("--out", default="ab_results.json")
    parser.add_argument("-q", "--quiet", action="store_true")
    args = parser.parse_args()

    if args.cassettes:
        os.environ["CASSETTE_MODE"] = args.cassettes

    matrix = (
        [(args.provider, m, m) for m in args.configs] if args.configs else DEFAULT_MATRIX
    )

    print(f"Scenarios: {len(SCENARIOS)}  |  configurations: {len(matrix)}  "
          f"|  cassettes: {os.environ.get('CASSETTE_MODE', 'off')}\n")

    summaries = []
    for provider, primary, fallback in matrix:
        print(f"  {provider}/{primary}")
        summaries.append(run_config(provider, primary, fallback, not args.quiet))
        print()

    header = f"{'model':<26}{'pass':>10}{'escal':>8}{'blocked':>9}{'redraft':>9}{'p50 ms':>10}"
    print(header)
    print("-" * len(header))
    for s in summaries:
        print(f"{s['primary']:<26}{s['passed']}/{s['total']:<8}{s['escalations']:>8}"
              f"{s['blocked_calls']:>9}{s['redrafts']:>9}{s['latency_p50_ms']:>10.0f}")

    Path(args.out).write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(f"\nFull results written to {args.out}")

    best = max(summaries, key=lambda s: (s["pass_rate"], -s["latency_p50_ms"]))
    print(f"Best on this scenario set: {best['primary']} "
          f"({best['passed']}/{best['total']}, p50 {best['latency_p50_ms']:.0f} ms)")


if __name__ == "__main__":
    main()
