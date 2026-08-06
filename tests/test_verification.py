"""
Verification gate.

Nothing about an order may be revealed before the caller proves who they are.
This replaces the earlier scheme where a session bound to whichever order was
mentioned first — that stopped a conversation wandering between customers, but
never established who the first one was.

All offline: contact matching and the graph guardrail, no model calls.
"""

import os

import pytest

os.environ.setdefault("TRENDLY_NOW", "2026-08-05T12:00:00Z")

from app.tools import customer_profile, find_customer, verify_customer  # noqa: E402


@pytest.mark.parametrize(
    "contact,expected",
    [
        ("ananya.rao@example.com", "C-100"),
        ("ANANYA.RAO@EXAMPLE.COM", "C-100"),      # case-insensitive
        ("+91-98765-10001", "C-100"),
        ("9876510001", "C-100"),                   # no country code
        ("98765 10001", "C-100"),                  # spaces
        ("marcus.bell@example.com", "C-101"),
        ("+1-415-555-0102", "C-101"),
        ("priya.nair@example.com", "C-102"),
        ("diego.ramos@example.com", "C-103"),
        ("nobody@example.com", None),
        ("123", None),
        ("", None),
    ],
)
def test_contact_matching(contact, expected):
    assert (find_customer(contact) or {}).get("customer_id") == expected


def test_verify_returns_a_greeting_not_a_data_dump():
    result = verify_customer.invoke({"contact": "priya.nair@example.com"})
    assert result["verified"] is True
    assert result["customer_id"] == "C-102"
    assert "Priya" in result["customer_message"]
    # Verification confirms identity; it must not spill the account.
    for leaked in ["@example.com", "+91", "TR-45"]:
        assert leaked not in result["customer_message"]


def test_unknown_contact_is_refused_without_confirming_anything():
    result = verify_customer.invoke({"contact": "stranger@example.com"})
    assert result["verified"] is False
    assert "customer_id" not in result


def test_profile_masks_contact_details():
    profile = customer_profile("C-100")
    assert "•" in profile["email_masked"]
    assert profile["email_masked"].endswith("@example.com")
    assert profile["phone_masked"].startswith("•")
    assert "98765" not in profile["phone_masked"]


def test_profile_returns_only_that_customers_orders():
    profile = customer_profile("C-100")
    assert {o["order_id"] for o in profile["orders"]} == {"TR-4521", "TR-4524", "TR-4529"}
    # Newest first — a customer asking about "my order" usually means the last one.
    assert profile["orders"][0]["order_id"] == "TR-4529"


def test_profile_of_unknown_customer_is_none():
    assert customer_profile("C-999") is None


def test_order_tools_are_gated_in_the_graph():
    """The gate is a code path, not a prompt instruction — assert it exists."""
    from app.agent import ORDER_TOOLS

    assert ORDER_TOOLS == {
        "lookup_order", "check_return_eligibility",
        "initiate_return", "apply_delayed_credit",
    }


def test_only_verification_can_bind_a_session():
    """
    Regression guard: previously lookup_order and check_return_eligibility both
    bound the session implicitly. If either does again, the gate is bypassable.
    """
    import inspect

    from app import agent

    source = inspect.getsource(agent.tool_node)
    assignments = [
        line.strip() for line in source.splitlines()
        if "session_customer_id =" in line and "state.get" not in line
    ]
    assert len(assignments) == 1, f"more than one binding path: {assignments}"
    assert 'result["customer_id"]' in assignments[0]


# ---------------------------------------------------------- support hours

@pytest.mark.parametrize(
    "wall_utc,expect_open",
    [
        ("2026-08-06T12:00:00Z", True),    # 17:30 IST
        ("2026-08-06T15:00:00Z", True),    # 20:30 IST, closing soon
        ("2026-08-06T15:45:00Z", False),   # 21:15 IST, just closed
        ("2026-08-06T21:30:00Z", False),   # 03:00 IST
        ("2026-08-06T02:00:00Z", False),   # 07:30 IST, before opening
        ("2026-08-06T03:35:00Z", True),    # 09:05 IST, just opened
    ],
)
def test_support_hours_track_the_wall_clock(monkeypatch, wall_utc, expect_open):
    from app.tools import support_status

    monkeypatch.setenv("SUPPORT_NOW", wall_utc)
    assert support_status()["open_now"] is expect_open


def test_support_hours_ignore_the_frozen_dataset_clock(monkeypatch):
    """
    The two clocks must not be the same one. TRENDLY_NOW keeps the dataset's
    return windows stable; reading it for support hours made the agent promise
    "we're open until 9 PM IST" at three in the morning.
    """
    from app.tools import support_status

    monkeypatch.setenv("TRENDLY_NOW", "2026-08-05T12:00:00Z")   # 17:30 IST, open
    monkeypatch.setenv("SUPPORT_NOW", "2026-08-06T21:30:00Z")   # 03:00 IST, shut
    assert support_status()["open_now"] is False


def test_escalation_message_matches_the_hour(monkeypatch):
    from app.tools import escalate_to_human

    monkeypatch.setenv("SUPPORT_NOW", "2026-08-06T21:30:00Z")
    closed = escalate_to_human.invoke({"summary": "s", "reason": "lost_parcel_claim"})
    assert "offline" in closed["customer_message"]
    assert "9 AM IST" in closed["customer_message"]

    monkeypatch.setenv("SUPPORT_NOW", "2026-08-06T12:00:00Z")
    open_now = escalate_to_human.invoke({"summary": "s", "reason": "lost_parcel_claim"})
    assert "offline" not in open_now["customer_message"]
    assert "shortly" in open_now["customer_message"]
