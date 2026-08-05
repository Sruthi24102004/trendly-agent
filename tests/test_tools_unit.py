"""
Fast, deterministic tests for the decision logic. No LLM, no network — the
whole file runs in well under a second, against a frozen clock so it stays
green regardless of when it's run.

The scenario tests in test_scenarios.py cover whether the *model* uses these
verdicts correctly. This file covers whether the verdicts are correct at all.
Run these first: when a scenario test fails, these tell you instantly whether
the bug is in the policy logic or in the prompt.
"""

import os

import pytest

# Freeze the clock before importing the tools, so every delivery date sits at
# a known distance from "now". 2026-08-05 is the assignment deadline week.
os.environ["TRENDLY_NOW"] = "2026-08-05T12:00:00Z"

from app.tools import (  # noqa: E402
    apply_delayed_credit,
    check_return_eligibility,
    escalate_to_human,
    initiate_return,
    lookup_order,
    search_policy,
    _find_item,
    _find_order,
)


def elig(order_id, item_id, issue_type="change_of_mind"):
    return check_return_eligibility.invoke(
        {"order_id": order_id, "item_id": item_id, "issue_type": issue_type}
    )


# ---------------------------------------------------------------- eligibility

@pytest.mark.parametrize(
    "order_id,item_id,expected_outcome,expected_reason",
    [
        # The clean happy path
        ("TR-4530", "TR-KRT-033", "eligible_refund", "within_policy"),
        # Delivered 2026-06-05 — 61 days ago, outside the 30-day window
        ("TR-4523", "TR-JKT-008", "not_eligible", "outside_return_window"),
        # In window, but jewellery is non-returnable on hygiene grounds
        ("TR-4527", "TR-EAR-042", "not_eligible", "non_returnable_category"),
        # In window and returnable, but final sale -> exchange only
        ("TR-4528", "TR-SHR-009", "exchange_only", "final_sale"),
        # Cancelled and already refunded
        ("TR-4529", "TR-SCF-027", "not_eligible", "order_cancelled"),
        # Carrier marked it lost: a claim, not a return
        ("TR-4526", "TR-BAG-011", "escalate", "lost_parcel_claim"),
        # Still in transit — nothing to return yet
        ("TR-4521", "TR-DRS-014", "not_eligible", "not_delivered"),
        # Socks are non-returnable even though the order is recent
        ("TR-4522", "TR-SOK-031", "not_eligible", "non_returnable_category"),
    ],
)
def test_eligibility_outcomes(order_id, item_id, expected_outcome, expected_reason):
    result = elig(order_id, item_id)
    assert result["outcome"] == expected_outcome, result
    assert result["reason_code"] == expected_reason, result
    assert result["customer_message"], "every verdict must carry a customer_message"


def test_final_sale_message_does_not_say_not_eligible():
    """
    Regression test for the exact failure seen in the first full run: the
    model replied "final sale item, which means it's not eligible for return".
    The message we hand it must make the opposite unmistakable.
    """
    msg = elig("TR-4528", "TR-SHR-009")["customer_message"].lower()
    assert "exchange" in msg
    assert "not eligible" not in msg


def test_damaged_item_overrides_non_returnable_category():
    """
    Policy 6.2: non-returnable categories ARE covered when the item arrives
    damaged. Jewellery delivered 2026-07-23 is outside the 48-hour reporting
    window, so this refuses on window grounds — not on category grounds.
    """
    result = elig("TR-4527", "TR-EAR-042", issue_type="damaged")
    assert result["outcome"] == "not_eligible"
    assert result["reason_code"] == "damage_report_window_expired"


def test_damaged_item_within_48h_escalates(monkeypatch):
    """Same item, reported the day after delivery: replacement or refund."""
    monkeypatch.setenv("TRENDLY_NOW", "2026-07-24T09:00:00Z")
    result = elig("TR-4527", "TR-EAR-042", issue_type="damaged")
    assert result["outcome"] == "escalate"
    assert result["reason_code"] == "damaged_within_report_window"


# Policy 2.5 (footwear must come back in its shoe box, else a ₹300 deduction)
# is implemented but cannot be exercised here: the only footwear order in the
# fixed dataset is TR-4525, which is still in transit and so never reaches the
# eligibility branch. Noted in SOLUTION.md as untested-by-construction.


# ----------------------------------------------------------- item resolution

def test_ambiguous_item_reference_asks_instead_of_guessing():
    """
    Regression test: the old matcher resolved "leather jacket" to a "Woven
    Leather Belt" because they share one word. Wrong item, silently.
    """
    result = elig("TR-4524", "leather jacket")
    assert result["outcome"] == "not_eligible"
    assert result["reason_code"] == "item_not_found"
    assert len(result["items_on_order"]) == 2


@pytest.mark.parametrize(
    "order_id,reference,expected_sku",
    [
        # Resolves: the reference names a product on this order
        ("TR-4522", "socks", "TR-SOK-031"),
        ("TR-4522", "cotton tee", "TR-TSH-002"),
        ("TR-4524", "the belt please", "TR-BLT-005"),
        ("TR-4524", "my jeans", "TR-JNS-021"),
        ("TR-4523", "my jacket", "TR-JKT-008"),
        ("TR-4527", "the earrings", "TR-EAR-042"),
        ("TR-4530", "my kurta", "TR-KRT-033"),
        ("TR-4530", "TR-KRT-033", "TR-KRT-033"),
        # Sole-item fallback: no product word at all, and only one line item
        # on the order, so there is nothing to disambiguate. The model passing
        # the order ID or a positional label used to deadlock the conversation
        # with "which of these did you mean?" on a one-item order.
        ("TR-4527", "TR-4527", "TR-EAR-042"),  # model passed the order ID
        ("TR-4527", "item_1", "TR-EAR-042"),   # positional label
        ("TR-4530", "", "TR-KRT-033"),         # empty reference
        # Refuses: names a product that isn't on this order, or is ambiguous
        ("TR-4524", "leather jacket", None),   # shares "leather" with the belt
        ("TR-4524", "jeans belt", None),       # could be either line item
        ("TR-4530", "a jacket", None),         # contradiction, not absence
        ("TR-4524", "item_1", None),           # no signal, but 2 items -> ask
        ("TR-4522", "that thing", None),       # no product word at all
    ],
)
def test_item_reference_resolution(order_id, reference, expected_sku):
    item, _ = _find_item(_find_order(order_id), reference)
    assert (item["sku"] if item else None) == expected_sku


# ------------------------------------------------------------- data exposure

def test_lookup_order_never_returns_customer_identity():
    """
    The cross-customer leak in the first run disclosed the customer's name.
    lookup_order doesn't need it, so it no longer returns it — the model
    cannot disclose a field it never receives.
    """
    result = lookup_order.invoke({"order_id": "TR-4522"})
    blob = str(result).lower()
    for forbidden in ["marcus", "bell", "@example.com", "+91", "c-101", "_note"]:
        assert forbidden not in blob, f"lookup_order leaked {forbidden!r}"


def test_lookup_order_unknown_id():
    result = lookup_order.invoke({"order_id": "TR-0000000"})
    assert result["found"] is False
    assert "customer_message" in result


# --------------------------------------------------------------- delay credit

def test_delayed_credit_issued_once():
    first = apply_delayed_credit.invoke({"order_id": "TR-4525"})
    assert first["outcome"] == "issued"
    assert first["credit_amount"] == 250

    second = apply_delayed_credit.invoke({"order_id": "TR-4525"})
    assert second["outcome"] == "already_issued", "credit must be idempotent"


def test_delayed_credit_refused_on_lost_parcel():
    """
    Regression: a lost parcel is 30 days past its estimate, so the delay-credit
    arithmetic says "eligible". Policy 1.6 says it is a claim for a human, not
    a delay. The tool issued the credit anyway and the model then offered to
    cancel the order for a refund — an action it has no tool for.
    """
    result = apply_delayed_credit.invoke({"order_id": "TR-4526"})
    assert result["outcome"] == "escalate"
    assert result["reason_code"] == "lost_parcel_claim"
    assert "credit_amount" not in result


def test_delayed_credit_not_offered_on_delivered_order():
    result = apply_delayed_credit.invoke({"order_id": "TR-4530"})
    assert result["outcome"] == "not_eligible"


# -------------------------------------------------------------------- policy

@pytest.mark.parametrize(
    "topic,must_contain",
    [
        ("refund timeline", "business days"),
        ("cash on delivery refund", "Cash on delivery"),
        ("express shipping cost", "Express shipping"),
        ("can I exchange for a different colour", "size exchanges only"),
        ("my dress arrived broken", "48 hours"),
        ("lost parcel", "lost-parcel claim"),
    ],
)
def test_policy_search_finds_the_right_section(topic, must_contain):
    """
    Every one of these returned NOT FOUND under the old substring scorer,
    which the prompt turns into an unnecessary escalation.
    """
    text = search_policy.invoke({"topic": topic})["policy_text"]
    assert must_contain.lower() in text.lower(), f"{topic!r} -> {text[:120]!r}"


def test_uncovered_topic_still_returns_not_found():
    text = search_policy.invoke(
        {"topic": "gift wrapping and personalised engraving service"}
    )["policy_text"]
    assert "NOT FOUND" in text


@pytest.mark.parametrize(
    "topic,must_contain",
    [
        # None of these phrasings appear verbatim in EXPANSIONS — they're
        # paraphrases a real customer would actually type, not the keywords
        # the retriever was hand-tuned against. Covers the retrieval-quality
        # risk flagged in policy_store.py's own docstring: word-level +
        # synonym-map matching, not embeddings.
        ("will I get my money back to my wallet", "Refunds"),
        ("is it free to send something back", "Shipping"),
        ("do returned items need the original packaging", "Returns"),
        ("can I get store credit instead of a refund", "Refunds"),
        ("how do I get a different size sent to me", "Exchanges"),
    ],
)
def test_policy_search_handles_customer_phrasing_not_in_expansions(topic, must_contain):
    text = search_policy.invoke({"topic": topic})["policy_text"]
    assert must_contain in text, f"{topic!r} -> {text[:80]!r}"


def test_ambiguous_delivery_failure_fails_safe_not_wrong_section():
    """
    'Courier never showed up' scores just under MIN_SCORE against every
    section (2 vs. threshold 3) — close enough to 1.6 Lost parcels that a
    looser threshold could easily mis-route it there, telling a customer
    their non-delivery is being handled as a lost-parcel claim when the doc
    doesn't actually say that. Escalating to NOT FOUND (-> human, per the
    system prompt) is the correct fail-safe here, not a retrieval miss to fix.
    """
    text = search_policy.invoke(
        {"topic": "what happens if the courier never shows up"}
    )["policy_text"]
    assert "NOT FOUND" in text


# ---------------------------------------------------------------- escalation

def test_escalation_carries_actionable_context():
    """A ticket a human can act on without redoing the lookup."""
    result = escalate_to_human.invoke({
        "summary": "Carrier marked parcel lost; customer wants a replacement.",
        "reason": "lost_parcel_claim",
        "order_id": "TR-4526",
    })
    ctx = result["order_context"]
    assert ctx["status"] == "lost_in_transit"
    assert ctx["tracking_number"] == "DL5519002244"
    assert ctx["items"][0]["item_id"] == "TR-BAG-011"
    assert result["ticket_id"].startswith("ESC-TR-4526")


# ------------------------------------------------------------------ returns

def test_exchange_requires_a_size():
    result = initiate_return.invoke({
        "order_id": "TR-4528", "item_id": "TR-SHR-009",
        "resolution": "exchange", "reason": "wrong size",
    })
    assert result["success"] is False
    assert result["error"] == "missing_size"


def test_refund_message_matches_payment_method():
    """COD and UPI have different refund routes — 3.1 shouldn't be guessed."""
    upi = initiate_return.invoke({
        "order_id": "TR-4527", "item_id": "TR-EAR-042",
        "resolution": "refund", "reason": "test",
    })
    assert "UPI" in upi["customer_message"]

    cod = initiate_return.invoke({
        "order_id": "TR-4528", "item_id": "TR-SHR-009",
        "resolution": "refund", "reason": "test",
    })
    assert "bank transfer" in cod["customer_message"]
    assert "agent_note" in cod  # never collect bank details in chat


def test_initiate_return_is_idempotent():
    """
    Nothing stopped a second initiate_return call on the same order+item from
    minting a second return_id — a real duplicate reverse pickup, not just a
    confusing reply. Uses TR-4530/TR-KRT-033, untouched by other tests in this
    file, so this doesn't depend on test execution order.
    """
    first = initiate_return.invoke({
        "order_id": "TR-4530", "item_id": "TR-KRT-033",
        "resolution": "refund", "reason": "changed my mind",
    })
    assert first["success"] is True

    second = initiate_return.invoke({
        "order_id": "TR-4530", "item_id": "TR-KRT-033",
        "resolution": "refund", "reason": "changed my mind",
    })
    assert second["success"] is False
    assert second["error"] == "already_initiated"
    assert second["return_id"] == first["return_id"]
