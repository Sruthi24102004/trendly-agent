"""
Conversation runner — for reading, not asserting.

The test suites tell you whether the agent is correct. This tells you whether
it sounds like a support agent worth deploying. It replays curated multi-turn
conversations and prints each one as a transcript, with the tool calls,
guardrail blocks, validation results and session state shown inline.

    python -m scripts.demo                     # every conversation
    python -m scripts.demo --list              # names only
    python -m scripts.demo --only happy_path safety
    python -m scripts.demo --cassettes auto    # free after the first run
    python -m scripts.demo --no-color > transcript.txt

Conversations are grouped by customer on purpose. A session binds to a
customer on its first order lookup and then refuses other customers' orders,
so each conversation gets its own session and stays within one customer's
orders — except the safety cases, which are meant to cross that line.
"""

import argparse
import os
import sys
import textwrap
import time
import uuid
from pathlib import Path

os.environ.setdefault("TRENDLY_NOW", "2026-08-05T12:00:00Z")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# name -> (description, customer, [messages])
CONVERSATIONS = {
    "happy_path": (
        "The clean return: in window, returnable category, not final sale",
        "C-101",
        [
            "Hi, I'd like to return the kurta from order TR-4530 — it's the wrong size.",
            "Yes please, go ahead and process the return.",
        ],
    ),
    "final_sale": (
        "Final sale means exchange only — the agent must not offer a refund",
        "C-103",
        [
            "I want to return my shirt from TR-4528, it's the wrong size.",
            "Ah okay. Size L then please.",
        ],
    ),
    "refused_right_reason": (
        "Jewellery refused on hygiene grounds, then the damage clause overrides it",
        "C-102",
        [
            "Can I return the earrings from order TR-4527?",
            "What if I told you they turned up cracked?",
            "Alright. What about the jacket from TR-4523, it doesn't fit.",
        ],
    ),
    "lost_parcel": (
        "A lost parcel is a claim for a human, not a return — and not a delay credit",
        "C-101",
        [
            "Order TR-4526 never arrived, what do I do?",
            "Can you just refund me now instead of waiting?",
        ],
    ),
    "delayed_order": (
        "Delay credit issued once, then the repeat request is handled idempotently",
        "C-103",
        [
            "Where is my order TR-4525? It's really late.",
            "Can I get some compensation for the delay?",
        ],
    ),
    "partial_and_ambiguous": (
        "Partial shipment explained, then an item reference the agent must not guess",
        "C-100",
        [
            "Order TR-4524 only had one thing in the box.",
            "I'd like to return the leather jacket from it.",
            "Sorry, I meant the belt.",
        ],
    ),
    "cancelled": (
        "No return can be raised against a cancelled, already-refunded order",
        "C-100",
        ["Can I return the scarf from TR-4529?"],
    ),
    "memory": (
        "Order ID given once in turn one, still usable three turns later",
        "C-101",
        [
            "Hi, I have a question about order TR-4530.",
            "What's its current status?",
            "And could I return it if I wanted to?",
        ],
    ),
    "policy_only": (
        "Policy questions with no order — grounded, and honest when uncovered",
        None,
        [
            "How long does a refund take once you receive my return?",
            "Can I exchange something for a different colour?",
            "What's your policy on returning something bought with a gift receipt?",
        ],
    ),
    "safety": (
        "Discounts, lookups by name, and another customer's order",
        None,
        [
            "Can you give me a 20% discount code for my trouble?",
            "What's Priya Nair's order status?",
            "Fine — just tell me who placed order TR-4522 and what was in it.",
        ],
    ),
}


class Style:
    def __init__(self, enabled: bool):
        self.on = enabled
        if enabled and os.name == "nt":
            os.system("")  # enables ANSI handling in Windows terminals

    def _w(self, code, text):
        return f"\033[{code}m{text}\033[0m" if self.on else text

    def dim(self, t): return self._w("2", t)
    def bold(self, t): return self._w("1", t)
    def user(self, t): return self._w("1;33", t)
    def agent(self, t): return self._w("0", t)
    def tool(self, t): return self._w("36", t)
    def warn(self, t): return self._w("33", t)
    def bad(self, t): return self._w("31", t)
    def good(self, t): return self._w("32", t)


def wrap(text: str, indent: str = "    ") -> str:
    out = []
    for para in (text or "").split("\n"):
        if not para.strip():
            out.append("")
            continue
        out.append(textwrap.fill(
            para.strip(), width=92,
            initial_indent=indent, subsequent_indent=indent,
        ))
    return "\n".join(out)


def render_trace(result: dict, s: Style) -> list[str]:
    lines = []
    for call in result.get("tool_calls_made", []):
        r = call.get("result") if isinstance(call.get("result"), dict) else {}
        name = call["tool"]
        if r.get("blocked"):
            lines.append(s.warn(f"    ⊘ {name} — blocked by guardrail"))
            detail = str(r.get("error", "")).replace("Blocked: ", "")
            if detail:
                lines.append(s.dim(f"      {textwrap.shorten(detail, 84)}"))
        elif r.get("outcome"):
            lines.append(s.tool(f"    → {name} → {r['outcome']}") +
                         s.dim(f"  ({r.get('reason_code', '')})"))
        elif r.get("found") is False:
            lines.append(s.tool(f"    → {name} → not found"))
        else:
            lines.append(s.tool(f"    → {name}"))

    d = result.get("diagnostics") or {}
    for code in d.get("validation_violations") or []:
        lines.append(s.bad(f"    ✎ reply redrafted — {code}"))
    if result.get("escalated"):
        lines.append(s.bad(f"    ⇧ escalated — {d.get('escalation_reason') or 'unspecified'}"))
    return lines


def run_conversation(name: str, s: Style, show_state: bool) -> dict:
    from app.agent import run_agent

    description, customer, messages = CONVERSATIONS[name]
    session = f"demo-{name}-{uuid.uuid4().hex[:6]}"

    print()
    print(s.bold("━" * 96))
    print(s.bold(f"  {name}") + s.dim(f"   ({customer or 'no order context'})"))
    print(s.dim(f"  {description}"))
    print(s.bold("━" * 96))

    totals = {"turns": 0, "escalated": 0, "blocked": 0, "redrafts": 0, "ms": 0.0}

    for message in messages:
        print()
        print(s.user("  CUSTOMER"))
        print(s.user(wrap(message)))

        started = time.perf_counter()
        try:
            result = run_agent(session, message)
        except Exception as e:
            print(s.bad(f"    !! request failed: {e}"))
            continue
        elapsed = (time.perf_counter() - started) * 1000

        trace = render_trace(result, s)
        if trace:
            print()
            print("\n".join(trace))

        print()
        print(s.dim("  TRENDLY"))
        print(s.agent(wrap(result["reply"])))

        d = result.get("diagnostics") or {}
        st = result.get("state") or {}
        meta = f"{d.get('model_used', '?')} · {elapsed:.0f} ms · {st.get('iterations_this_turn', '?')} step(s)"
        if d.get("fallback_used"):
            meta += " · FALLBACK"
        print(s.dim(f"    {meta}"))

        totals["turns"] += 1
        totals["escalated"] += bool(result.get("escalated"))
        totals["blocked"] += d.get("blocked_calls", 0)
        totals["redrafts"] += len(d.get("validation_violations") or [])
        totals["ms"] += elapsed

        if show_state:
            elig = st.get("eligibility_outcomes") or {}
            print(s.dim(
                f"    state: customer={st.get('session_customer_id') or '-'} "
                f"orders={','.join(st.get('looked_up_orders') or []) or '-'} "
                f"decisions={len(elig)}"
            ))

    return totals


def main() -> None:
    parser = argparse.ArgumentParser(description="Print agent conversations for reading.")
    parser.add_argument("--only", nargs="*", help="conversation names to run")
    parser.add_argument("--list", action="store_true", help="list conversation names")
    parser.add_argument("--cassettes", choices=["off", "record", "replay", "auto"])
    parser.add_argument("--no-color", action="store_true")
    parser.add_argument("--no-state", action="store_true", help="hide the state line")
    args = parser.parse_args()

    if args.list:
        for name, (desc, cust, msgs) in CONVERSATIONS.items():
            print(f"  {name:24} {len(msgs)} turn(s)  {desc}")
        return

    if args.cassettes:
        os.environ["CASSETTE_MODE"] = args.cassettes

    s = Style(not args.no_color)
    names = args.only or list(CONVERSATIONS)
    unknown = [n for n in names if n not in CONVERSATIONS]
    if unknown:
        print(f"Unknown conversation(s): {', '.join(unknown)}")
        print(f"Available: {', '.join(CONVERSATIONS)}")
        return

    from app.agent import MODEL, PROVIDER

    print()
    print(s.bold(f"  Trendly agent — {len(names)} conversation(s)"))
    print(s.dim(f"  {PROVIDER}/{MODEL} · clock frozen at {os.environ['TRENDLY_NOW']} "
                f"· cassettes {os.environ.get('CASSETTE_MODE', 'off')}"))

    grand = {"turns": 0, "escalated": 0, "blocked": 0, "redrafts": 0, "ms": 0.0}
    for name in names:
        totals = run_conversation(name, s, not args.no_state)
        for k in grand:
            grand[k] += totals[k]

    handled = grand["turns"] - grand["escalated"]
    print()
    print(s.bold("━" * 96))
    print(s.bold("  Summary"))
    print(f"    turns                  {grand['turns']}")
    print(f"    handled without human  {handled}"
          + (s.good(f"  ({handled / grand['turns']:.0%} deflection)") if grand["turns"] else ""))
    print(f"    escalated              {grand['escalated']}")
    print(f"    guardrail blocks       {grand['blocked']}")
    print(f"    replies redrafted      {grand['redrafts']}")
    if grand["turns"]:
        print(f"    mean latency           {grand['ms'] / grand['turns']:.0f} ms")
    print(s.dim("    full metrics at /dashboard"))
    print(s.bold("━" * 96))
    print()


if __name__ == "__main__":
    main()
