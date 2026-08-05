"""
Reply validation.

Guardrails so far constrain what the agent may *do*. This constrains what it
may *say*. Every incorrect answer in this project got through not because a
tool returned the wrong verdict, but because the model then described that
verdict wrongly:

  - eligibility returned exchange_only, the reply said "not eligible for return"
  - eligibility returned outside_return_window, the reply said "you can't
    return it because it's already been delivered"
  - a lost parcel got a ₹250 delay credit and an offer to "cancel the order
    for a full refund" — an action with no backing tool

Each check compares the drafted reply against the tool results from the same
turn. Deterministic, no second model call, so it costs nothing and can't
itself hallucinate. On violation the agent gets one corrective retry; if it
fails again the turn escalates rather than sending a wrong answer.
"""

import re

from app.tools import _load_data

# --- Contradictions: phrases a reply must not contain given an outcome ------

REFUSAL_PHRASES = [
    "not eligible",
    "isn't eligible",
    "is not eligible",
    "can't be returned",
    "cannot be returned",
    "unable to accept",
    "not returnable",
    "can't accept this back",
]

APPROVAL_PHRASES = [
    "is eligible for return",
    "you're eligible for a refund",
    "i've raised the return",
    "i've processed the return",
    "your return is confirmed",
]

CONTRADICTIONS = {
    # exchange_only means eligible — for an exchange. Calling it "not eligible"
    # is the single most common way this went wrong.
    "exchange_only": REFUSAL_PHRASES,
    "eligible_refund": REFUSAL_PHRASES,
    "not_eligible": APPROVAL_PHRASES,
}

# --- Claimed actions, and what would have to have happened to justify them ---

ACTION_CLAIMS = {
    "return_created": [
        r"i'?ve (raised|created|processed|initiated|set up) (the |a |your )?return",
        r"your return (has been|is now) (raised|created|processed)",
    ],
    "exchange_created": [
        r"i'?ve (raised|created|processed|arranged) (the |a |an |your )?(size )?exchange",
    ],
    "credit_issued": [
        r"i'?ve (added|issued|applied|credited)[^.]{0,40}(store )?credit",
        r"(store )?credit (has been|is) (added|issued|applied)",
    ],
    "refund_issued": [
        r"i'?ve (issued|processed|started|arranged)[^.]{0,20}(the |a |your )?refund",
    ],
    "order_cancelled": [
        r"i'?ve cancelled",
        r"i can cancel (the|your) order",
        r"cancel (the|your) order for a (full )?refund",
    ],
    "pickup_scheduled": [
        r"i'?ve (scheduled|booked|arranged) (the |a )?(reverse )?pickup",
    ],
}

# --- Policy specifics that must be grounded in a tool result -----------------

POLICY_SPECIFICS = [
    r"₹\s?\d{2,4}",
    r"\b30[- ]days?\b",
    r"\b30 calendar days\b",
    r"\b48[- ]hours?\b",
    r"\b\d\s?[–-]\s?\d\s+business days\b",
    r"\b\d+ business days\b",
]

GROUNDING_TOOLS = {
    "search_policy",
    "check_return_eligibility",
    "apply_delayed_credit",
    "initiate_return",
    "lookup_order",
}

SENSITIVE_REQUESTS = [
    r"\baccount number\b",
    r"\bifsc\b",
    r"\bcvv\b",
    r"\bcard number\b",
    r"\bsort code\b",
    r"\brouting number\b",
]

DISCOUNT_OFFER = r"\b\d{1,2}\s?%\s?(off|discount)\b"

_CUSTOMER_NAMES: dict[str, str] | None = None


def _customer_names() -> dict[str, str]:
    """{customer_id: lowercased name} — used to detect cross-customer leaks."""
    global _CUSTOMER_NAMES
    if _CUSTOMER_NAMES is None:
        _CUSTOMER_NAMES = {
            c["customer_id"]: c["name"].lower() for c in _load_data()["customers"]
        }
    return _CUSTOMER_NAMES


def _executed(tool_results: list[dict], name: str) -> list[dict]:
    return [
        r["result"]
        for r in tool_results
        if r.get("tool") == name and not (r.get("result") or {}).get("blocked")
    ]


def _action_supported(claim: str, tool_results: list[dict]) -> bool:
    returns = _executed(tool_results, "initiate_return")
    credits = _executed(tool_results, "apply_delayed_credit")

    if claim == "return_created":
        return any(r.get("success") and r.get("type") == "refund_return" for r in returns)
    if claim == "exchange_created":
        return any(r.get("success") and r.get("type") == "size_exchange" for r in returns)
    if claim in ("refund_issued", "pickup_scheduled"):
        return any(r.get("success") for r in returns)
    if claim == "credit_issued":
        return any(r.get("outcome") in ("issued", "already_issued") for r in credits)
    if claim == "order_cancelled":
        # There is no cancellation tool. Any such offer is invented authority.
        return False
    return True


def validate_reply(
    reply: str,
    tool_results: list[dict],
    session_customer_id: str | None = None,
    session_grounded: bool = False,
) -> list[dict]:
    """
    Returns a list of violations, each {code, detail}. Empty means the reply
    is consistent with what actually happened this turn.

    `session_grounded` says whether a grounding tool ran anywhere earlier in
    the conversation. The grounding check exists to stop the model inventing
    policy figures from nothing — but a follow-up turn ("what's its status?")
    legitimately answers from context established a turn ago, and flagging
    that produced a needless redraft. Action claims and contradictions are
    still judged strictly per turn: those must be backed by something that
    happened *now*.
    """
    violations: list[dict] = []
    if not reply or not reply.strip():
        return [{"code": "empty_reply", "detail": "The reply was empty."}]

    lowered = reply.lower()
    executed_tools = {
        r["tool"] for r in tool_results if not (r.get("result") or {}).get("blocked")
    }

    # 1. Does the reply contradict the eligibility verdict it is reporting?
    #
    # This is a whole-reply substring search, not per-item — it can't tell
    # which sentence belongs to which line item. With exactly one
    # check_return_eligibility result that's unambiguous. With two or more
    # (a multi-item request with mixed outcomes — e.g. one item eligible,
    # one genuinely "not eligible"), a correct reply legitimately contains
    # phrases from *both* CONTRADICTIONS lists at once, and this check can't
    # tell honest disambiguation from an actual contradiction. Skipping it
    # for the multi-item case trades missing a real contradiction there
    # (rare) for not rejecting correct replies (the failure mode this
    # project's actual bugs came from) — a fixable known limitation, not a
    # silent gap: see SOLUTION.md.
    eligibility_results = _executed(tool_results, "check_return_eligibility")
    if len(eligibility_results) == 1:
        outcome = eligibility_results[0].get("outcome")
        for phrase in CONTRADICTIONS.get(outcome, []):
            if phrase in lowered:
                violations.append({
                    "code": "outcome_contradiction",
                    "detail": (
                        f"Eligibility returned '{outcome}' but the reply says "
                        f"'{phrase}'. Use the tool's customer_message."
                    ),
                })
                break

    # 2. Does it claim an action that never happened?
    for claim, patterns in ACTION_CLAIMS.items():
        if any(re.search(p, lowered) for p in patterns):
            if not _action_supported(claim, tool_results):
                violations.append({
                    "code": "unsupported_action_claim",
                    "detail": (
                        f"The reply claims '{claim}' but no tool result "
                        "supports it. Do not promise actions you have not taken."
                    ),
                })

    # 3. Does it state policy specifics without having consulted policy —
    #    this turn or any earlier one in the conversation?
    if not (executed_tools & GROUNDING_TOOLS) and not session_grounded:
        if any(re.search(p, lowered) for p in POLICY_SPECIFICS):
            violations.append({
                "code": "ungrounded_policy_claim",
                "detail": (
                    "The reply states a figure or timeframe from policy but no "
                    "grounding tool ran this turn. Call search_policy first."
                ),
            })

    # 4. Does it name a customer other than the one in this conversation?
    for customer_id, name in _customer_names().items():
        if customer_id == session_customer_id:
            continue
        first = name.split()[0]
        if name in lowered or (len(first) > 4 and re.search(rf"\b{first}\b", lowered)):
            violations.append({
                "code": "cross_customer_disclosure",
                "detail": "The reply names a customer other than this one.",
            })
            break

    # 5. Is it soliciting details that must never be collected in chat?
    if any(re.search(p, lowered) for p in SENSITIVE_REQUESTS):
        violations.append({
            "code": "sensitive_data_request",
            "detail": (
                "The reply mentions bank/card details. These are collected by "
                "a human over a secure link, never in chat."
            ),
        })

    # 6. Is it inventing a discount?
    if re.search(DISCOUNT_OFFER, lowered):
        violations.append({
            "code": "unauthorised_discount",
            "detail": "The reply offers a discount that policy does not define.",
        })

    return violations


def correction_prompt(violations: list[dict]) -> str:
    """The corrective message fed back to the model for its one retry."""
    lines = "\n".join(f"- {v['code']}: {v['detail']}" for v in violations)
    return (
        "Your draft reply was rejected by an automated check before it reached "
        "the customer. Problems found:\n"
        f"{lines}\n\n"
        "Rewrite the reply so it states exactly what the tool results support — "
        "no more, no less. Base it on each result's customer_message. Do not "
        "apologise for the correction or mention that it happened."
    )
