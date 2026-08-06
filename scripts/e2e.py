"""
End-to-end check of the whole project, in three stages of increasing cost.

    python -m scripts.e2e              # stages 1 and 2 — free, no model calls
    python -m scripts.e2e --live       # adds stage 3, the conversations
    python -m scripts.e2e --live --only happy_path safety

Stage 1  offline suites — policy logic, guardrails, validation, verification
Stage 2  HTTP surface — routes, admin gating, data exposure, error handling
Stage 3  live conversations — the agent actually talking, paced for free tiers

Stages 1 and 2 need no API key and finish in seconds, so they are what CI runs
and what a reviewer can run without credentials. Stage 3 is the only part that
spends quota, which is why it is opt-in.
"""

import argparse
import importlib
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("TRENDLY_NOW", "2026-08-05T12:00:00Z")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

RESULTS: list[tuple[str, str, bool, str]] = []


def check(stage: str, name: str, passed: bool, detail: str = "") -> bool:
    RESULTS.append((stage, name, passed, detail))
    mark = "PASS" if passed else "FAIL"
    line = f"  {mark}  {name}"
    if detail and not passed:
        line += f"\n        {detail}"
    print(line)
    return passed


def header(text: str) -> None:
    print()
    print("=" * 92)
    print(f"  {text}")
    print("=" * 92)


# ------------------------------------------------------------------ stage 1

OFFLINE_SUITES = [
    "tests/test_tools_unit.py",
    "tests/test_guardrails_unit.py",
    "tests/test_verification.py",
    "tests/test_admin_routes.py",
]


def stage_offline() -> None:
    header("Stage 1 — offline suites (no model, no network)")
    for suite in OFFLINE_SUITES:
        if not Path(suite).exists():
            check("offline", suite, False, "file not found")
            continue
        started = time.perf_counter()
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", suite, "-q", "--no-header", "-p", "no:cacheprovider"],
            capture_output=True, text=True,
        )
        elapsed = time.perf_counter() - started
        tail = [ln for ln in proc.stdout.strip().splitlines() if ln.strip()]
        summary = tail[-1] if tail else "no output"
        check("offline", f"{suite}  ({elapsed:.1f}s)", proc.returncode == 0, summary)


# ------------------------------------------------------------------ stage 2

def _client(token: str | None):
    if token is None:
        os.environ.pop("ADMIN_TOKEN", None)
    else:
        os.environ["ADMIN_TOKEN"] = token
    import app.main
    importlib.reload(app.main)
    from fastapi.testclient import TestClient
    return TestClient(app.main.app), app.main


def stage_http() -> None:
    header("Stage 2 — HTTP surface (no model calls)")

    client, main = _client("e2e-token")

    # --- customer routes stay open ---
    check("http", "GET / serves the chat page", client.get("/").status_code == 200)
    check("http", "GET /health is public", client.get("/health").status_code == 200)
    check("http", "public /health hides configuration",
          client.get("/health").json() == {"status": "ok"})
    check("http", "admin /health reveals model",
          "model" in client.get("/health", headers={"x-admin-token": "e2e-token"}).json())

    # --- operator routes are gated ---
    for route in ["/metrics", "/sessions", "/dashboard", "/history"]:
        check("http", f"{route} 404s without a token",
              client.get(route).status_code == 404)
        check("http", f"{route} opens with the token",
              client.get(route, headers={"x-admin-token": "e2e-token"}).status_code == 200)

    # --- the customer page must not advertise any of it ---
    page = main.CHAT_PAGE
    leaks = [r for r in ["/dashboard", "/history", "/metrics", "/sessions"]
             if f'href="{r}"' in page]
    check("http", "chat page links to no operator route", not leaks, str(leaks))
    check("http", "chat page shows no internal diagnostics",
          "validation_violations" not in page and "agent steps" not in page)

    # --- session endpoints ---
    unknown = client.get("/session/never-existed").json()
    check("http", "unknown session replays as empty", unknown["exists"] is False)
    profile = client.get("/session/never-existed/customer").json()
    check("http", "unverified session exposes no customer", profile == {"verified": False})

    # --- request validation ---
    check("http", "empty message rejected",
          client.post("/chat", json={"session_id": "x", "message": "   "}).status_code == 400)
    check("http", "malformed body rejected",
          client.post("/chat", json={"message": "hi"}).status_code == 422)

    # --- localhost fallback when no token is configured ---
    local, _ = _client(None)
    check("http", "operator routes open locally when no token is set",
          all(local.get(r).status_code == 200
              for r in ["/metrics", "/sessions", "/dashboard", "/history"]))
    check("http", "health reports the auth mode",
          local.get("/health").json().get("admin_auth") == "localhost-only")

    # --- data exposure at the tool boundary ---
    from app.tools import customer_profile, lookup_order
    blob = str(lookup_order.invoke({"order_id": "TR-4522"})).lower()
    check("http", "lookup_order returns no customer identity",
          not any(t in blob for t in ["marcus", "bell", "@example.com", "c-101"]))
    prof = customer_profile("C-100")
    check("http", "profile masks contact details",
          "•" in prof["email_masked"] and "98765" not in prof["phone_masked"])
    check("http", "profile scopes orders to that customer",
          {o["order_id"] for o in prof["orders"]} == {"TR-4521", "TR-4524", "TR-4529"})

    # --- the guardrails exist as code, not just prompt text ---
    import inspect

    from app import agent
    source = inspect.getsource(agent.tool_node)
    check("http", "order tools gated on verification",
          "if tool_name in ORDER_TOOLS and not session_customer_id" in source)
    bindings = [ln.strip() for ln in source.splitlines()
                if "session_customer_id =" in ln and "state.get" not in ln]
    check("http", "only verification can bind a session",
          len(bindings) == 1 and 'result["customer_id"]' in bindings[0], str(bindings))
    check("http", "initiate_return checks the eligibility outcome",
          "ALLOWED_RESOLUTIONS[outcome]" in source)


# ------------------------------------------------------------------ stage 3

def stage_live(only: list[str] | None, interval_ms: int) -> None:
    header("Stage 3 — live conversations")
    os.environ["MIN_MODEL_INTERVAL_MS"] = str(interval_ms)

    from scripts import demo

    names = only or list(demo.CONVERSATIONS)
    style = demo.Style(False)
    totals = {"turns": 0, "escalated": 0, "blocked": 0, "redrafts": 0, "ms": 0.0}

    for name in names:
        if name not in demo.CONVERSATIONS:
            check("live", name, False, "unknown conversation")
            continue
        try:
            result = demo.run_conversation(name, style, False)
            for key in totals:
                totals[key] += result[key]
            check("live", f"{name} ({result['turns']} turns)", True)
        except Exception as e:
            check("live", name, False, repr(e))

    if totals["turns"]:
        handled = totals["turns"] - totals["escalated"]
        print()
        print(f"  turns {totals['turns']} · handled without human {handled} "
              f"({handled / totals['turns']:.0%}) · escalated {totals['escalated']} "
              f"· blocked {totals['blocked']} · redrafts {totals['redrafts']} "
              f"· mean {totals['ms'] / totals['turns']:.0f} ms")


# ------------------------------------------------------------------ report

def report() -> int:
    header("Summary")
    stages = {}
    for stage, _, passed, _ in RESULTS:
        row = stages.setdefault(stage, [0, 0])
        row[0 if passed else 1] += 1

    for stage, (passed, failed) in stages.items():
        status = "OK" if not failed else f"{failed} FAILED"
        print(f"  {stage:10} {passed} passed, {failed} failed   {status}")

    failures = [(s, n, d) for s, n, p, d in RESULTS if not p]
    if failures:
        print()
        print("  Failures:")
        for stage, name, detail in failures:
            print(f"    [{stage}] {name}")
            if detail:
                print(f"        {detail}")
    print()
    return 1 if failures else 0


def main() -> None:
    parser = argparse.ArgumentParser(description="End-to-end check of the whole project.")
    parser.add_argument("--live", action="store_true", help="also run conversations (uses quota)")
    parser.add_argument("--only", nargs="*", help="limit stage 3 to these conversations")
    parser.add_argument("--interval-ms", type=int, default=4500,
                        help="gap between model calls in stage 3")
    parser.add_argument("--skip-offline", action="store_true")
    args = parser.parse_args()

    print()
    print("  Trendly agent — end-to-end check")
    print(f"  clock frozen at {os.environ['TRENDLY_NOW']}")

    if not args.skip_offline:
        stage_offline()
    stage_http()
    if args.live:
        stage_live(args.only, args.interval_ms)
    else:
        print()
        print("  Stage 3 skipped — pass --live to run the conversations.")

    sys.exit(report())


if __name__ == "__main__":
    main()
