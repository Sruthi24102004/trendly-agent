"""
Fast offline tests for the reply validator, the event log, and the cassette
layer. No model, no network — the whole file runs in under a second.

The validator cases are drawn from actual wrong answers this agent produced
during development, so each one is a regression test rather than a
hypothetical.
"""

import json
import os
import tempfile

import pytest

os.environ.setdefault("TRENDLY_NOW", "2026-08-05T12:00:00Z")

from app.validation import validate_reply, correction_prompt  # noqa: E402


def tool(name, result):
    return {"tool": name, "result": result}


def codes(violations):
    return {v["code"] for v in violations}


# ------------------------------------------------- outcome contradictions

def test_final_sale_described_as_not_eligible_is_rejected():
    """
    Real failure: eligibility returned exchange_only and the agent replied
    "final sale item, which means it's not eligible for return". exchange_only
    means eligible — for an exchange.
    """
    v = validate_reply(
        "That item is a final sale item, which means it's not eligible for return.",
        [tool("check_return_eligibility", {"outcome": "exchange_only"})],
    )
    assert "outcome_contradiction" in codes(v)


def test_correct_final_sale_reply_passes():
    v = validate_reply(
        "That was a final sale item, so I can arrange a size exchange but not a "
        "refund. Which size would you like?",
        [tool("check_return_eligibility", {"outcome": "exchange_only"})],
    )
    assert v == []


def test_legitimate_refusal_passes():
    v = validate_reply(
        "Those earrings are in our jewellery category, which we can't accept "
        "back for hygiene reasons.",
        [tool("check_return_eligibility", {"outcome": "not_eligible"})],
    )
    assert v == []


def test_multi_item_mixed_outcome_reply_not_falsely_flagged():
    """
    Found while checking multi-item handling (not from a live bug report,
    unlike the others in this file): a request naming two items with
    different outcomes produces a correct reply that legitimately contains
    phrases from *both* CONTRADICTIONS lists at once. The contradiction
    check is a whole-reply substring search with no per-item attribution, so
    it can't tell that "not eligible for return" describes the socks, not
    the tee whose result is eligible_refund. Guards against reintroducing
    that false positive.
    """
    v = validate_reply(
        "Your Everyday Cotton Tee is eligible for a refund, so I can raise "
        "that return for you. The Ankle Socks 3-pack, though, is not "
        "eligible for return since socks fall under our non-returnable "
        "hygiene category.",
        [
            tool("check_return_eligibility", {"outcome": "eligible_refund"}),
            tool("check_return_eligibility", {"outcome": "not_eligible"}),
        ],
    )
    assert v == []


# ----------------------------------------------------- unsupported actions

def test_offering_to_cancel_an_order_is_rejected():
    """
    Real failure: on a lost parcel the agent offered to "cancel the order for
    a full refund". There is no cancellation tool — that is invented authority.
    """
    v = validate_reply(
        "I've added a ₹250 store credit. Would you like me to cancel the order "
        "for a full refund?",
        [tool("apply_delayed_credit", {"outcome": "issued", "credit_amount": 250})],
    )
    assert "unsupported_action_claim" in codes(v)


def test_claiming_a_return_that_was_never_created_is_rejected():
    v = validate_reply(
        "All set — I've raised the return for your kurta.",
        [tool("check_return_eligibility", {"outcome": "eligible_refund"})],
    )
    assert "unsupported_action_claim" in codes(v)


def test_claiming_a_return_that_was_created_passes():
    v = validate_reply(
        "All set — I've raised the return for your kurta.",
        [
            tool("check_return_eligibility", {"outcome": "eligible_refund"}),
            tool("initiate_return", {"success": True, "type": "refund_return"}),
        ],
    )
    assert v == []


def test_blocked_tool_call_does_not_support_a_claim():
    """A guardrail rejection is not an action. It must not license the claim."""
    v = validate_reply(
        "I've raised the return for you.",
        [tool("initiate_return", {"blocked": True, "error": "Blocked: ..."})],
    )
    assert "unsupported_action_claim" in codes(v)


# --------------------------------------------------------- policy grounding

@pytest.mark.parametrize(
    "reply",
    [
        "Our return window is 30 days from delivery.",
        "There's a 30-day window on that.",
        "Damage has to be reported within 48 hours.",
        "Express shipping is ₹199.",
        "Refunds take 5-7 business days.",
    ],
)
def test_policy_specifics_without_grounding_are_rejected(reply):
    assert "ungrounded_policy_claim" in codes(validate_reply(reply, []))


def test_policy_specifics_with_grounding_pass():
    v = validate_reply(
        "Our return window is 30 days from delivery.",
        [tool("search_policy", {"topic": "returns"})],
    )
    assert v == []


# ------------------------------------------------------------ data safety

def test_naming_another_customer_is_rejected():
    v = validate_reply(
        "Order TR-4522 was placed by Marcus Bell and contains two cotton tees.",
        [tool("lookup_order", {"found": True})],
        session_customer_id="C-103",
    )
    assert "cross_customer_disclosure" in codes(v)


def test_own_session_has_no_false_positive():
    v = validate_reply(
        "Your order is on its way and should arrive shortly.",
        [tool("lookup_order", {"found": True})],
        session_customer_id="C-101",
    )
    assert v == []


def test_asking_for_bank_details_is_rejected():
    v = validate_reply(
        "Please share your account number and IFSC so I can refund you.",
        [tool("lookup_order", {"found": True})],
    )
    assert "sensitive_data_request" in codes(v)


def test_offering_a_discount_is_rejected():
    assert "unauthorised_discount" in codes(
        validate_reply("I can offer you 20% off as an apology.", [])
    )


def test_empty_reply_is_rejected():
    assert "empty_reply" in codes(validate_reply("", []))


def test_correction_prompt_names_every_violation():
    prompt = correction_prompt([
        {"code": "outcome_contradiction", "detail": "x"},
        {"code": "unsupported_action_claim", "detail": "y"},
    ])
    assert "outcome_contradiction" in prompt
    assert "unsupported_action_claim" in prompt
    # The customer must never learn a correction happened.
    assert "do not" in prompt.lower()


# ------------------------------------------------------------ observability

def test_metrics_aggregate_deflection_and_escalation_split():
    from app import observability

    with tempfile.TemporaryDirectory() as d:
        observability.LOG_PATH = observability.Path(d) / "events.jsonl"
        observability.log_turn({"session_id": "s1", "escalated": False, "latency_ms": 800,
                                "tools": ["lookup_order"]})
        observability.log_turn({"session_id": "s1", "escalated": False, "latency_ms": 900,
                                "tools": ["search_policy"]})
        observability.log_turn({"session_id": "s2", "escalated": True, "latency_ms": 1200,
                                "escalation_reason": "lost_parcel_claim"})
        observability.log_turn({"session_id": "s3", "escalated": True, "latency_ms": 1400,
                                "escalation_reason": "reply_validation_failed"})

        m = observability.summarize()
        assert m["turns"] == 4
        assert m["sessions"] == 3
        assert m["deflection_rate"] == 0.5
        # A lost parcel is policy telling us to escalate; a validation failure
        # is the agent falling short. Ops needs those counted separately.
        assert m["escalations"]["policy_mandated"] == 1
        assert m["escalations"]["agent_limitation"] == 1


def test_metrics_handle_an_empty_log():
    from app import observability

    with tempfile.TemporaryDirectory() as d:
        observability.LOG_PATH = observability.Path(d) / "none.jsonl"
        assert observability.summarize()["turns"] == 0


def test_torn_log_line_does_not_break_metrics():
    from app import observability

    with tempfile.TemporaryDirectory() as d:
        path = observability.Path(d) / "events.jsonl"
        path.write_text(
            json.dumps({"session_id": "s1", "escalated": False}) + "\n{ broken\n",
            encoding="utf-8",
        )
        observability.LOG_PATH = path
        assert observability.summarize()["turns"] == 1
