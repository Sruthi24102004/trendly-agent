"""
Tool implementations.

Key design decision: every decision tool returns a single unambiguous
`outcome` plus a `customer_message` the model is instructed to reuse rather
than re-derive. The previous shape ({"eligible": true, "exchange_only": true})
let the model pick whichever field it liked — it read `final_sale` and told a
customer a final-sale item was "not eligible for return", which is wrong.
Moving the *explanation* out of the model, not just the decision, is what
fixes that class of failure.

Outcomes: eligible_refund | exchange_only | not_eligible | escalate
"""

import json
import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

from langchain_core.tools import tool

from app.policy_store import search_policy_section

DATA_PATH = Path(__file__).parent.parent / "data" / "orders.json"

RETURN_WINDOW_DAYS = 30          # policy 2.1
DAMAGE_REPORT_HOURS = 48         # policy 6.1
DELAY_THRESHOLD_DAYS = 3         # policy 1.5
FOOTWEAR_NO_BOX_DEDUCTION = 300  # policy 2.5
DELAY_CREDIT_AMOUNT = 250        # policy 1.5

# policy 2.3 — matched against the `category` field in orders.json
NON_RETURNABLE_CATEGORIES = {
    "innerwear", "socks", "jewellery", "beauty", "fragrance",
    "beauty and fragrance", "face masks", "gift cards",
}

ISSUE_TYPES_DAMAGE = {"damaged", "defective", "wrong_item"}

# In-memory idempotency guard so the ₹250 delay credit can't be issued twice
# in one process. A real deployment would persist this on the order record —
# noted as a limitation in SOLUTION.md.
_ISSUED_DELAY_CREDITS: set[str] = set()

# Same idea for returns/exchanges: without this, a customer who says "return
# it" twice in one session (or a model that re-calls the tool after a
# mis-parsed confirmation) raises two return_ids for the same item, and a
# human ends up reconciling two reverse pickups against one product. Keyed
# on (order_id, sku) rather than the generated return_id since the whole
# point is catching the call *before* a second ID is minted.
_OPEN_RETURNS: dict[tuple[str, str], str] = {}


def _now() -> datetime:
    """
    Current time, overridable via TRENDLY_NOW for deterministic tests and
    demos. Read per call (not cached) so tests can monkeypatch the env.
    """
    override = os.environ.get("TRENDLY_NOW")
    if override:
        return datetime.fromisoformat(override.replace("Z", "+00:00"))
    return datetime.now(timezone.utc)


def _load_data() -> dict:
    with open(DATA_PATH, encoding="utf-8") as f:
        return json.load(f)


def _find_order(order_id: str) -> dict | None:
    if not order_id:
        return None
    for order in _load_data()["orders"]:
        if order["order_id"].upper() == order_id.strip().upper():
            return order
    return None


def _stem(word: str) -> str:
    """Crude singular/plural fold so 'socks' matches 'Socks'."""
    return word[:-1] if len(word) > 3 and word.endswith("s") else word


_VOCAB_CACHE: set[str] | None = None


def _catalogue_vocabulary() -> set[str]:
    """Every word appearing in any product name across the catalogue."""
    global _VOCAB_CACHE
    if _VOCAB_CACHE is None:
        _VOCAB_CACHE = {
            _stem(w)
            for order in _load_data()["orders"]
            for item in order["items"]
            for w in re.findall(r"[a-z]+", item["name"].lower())
            if len(w) > 2
        }
    return _VOCAB_CACHE


def _find_item(order: dict, item_id: str) -> tuple[dict | None, str]:
    """
    Resolve an item reference to a real line item.

    Returns (item, match_kind). match_kind is "exact", "name", or "none".

    The old version had a third fallback that matched on ANY shared word of
    3+ characters, so "leather jacket" resolved to a "Woven Leather Belt" and
    "jeans belt" resolved to the jeans. That silently returned a verdict about
    the wrong product. It is deliberately gone: when we can't resolve the
    reference confidently we say so and let the agent ask.
    """
    needle = (item_id or "").strip().lower()

    if needle:
        for item in order["items"]:
            if item["sku"].lower() == needle:
                return item, "exact"

        # Whole-phrase containment in either direction, e.g. "socks" -> "Ankle
        # Socks 3-pack", or "Block-Print Kurta (large)" -> "Block-Print Kurta".
        for item in order["items"]:
            name = item["name"].lower()
            if needle in name or name in needle:
                return item, "name"

    # Plain-language reference ("the belt please", "my kurta"). Words are
    # scored against the catalogue vocabulary so that filler ("the", "please")
    # is ignored while a real product word that ISN'T on this order counts as
    # a contradiction. That is what stops "leather jacket" resolving to a
    # "Woven Leather Belt": `leather` matches, but `jacket` is a genuine
    # catalogue word that this item doesn't have, so the match is rejected.
    query = {_stem(w) for w in re.findall(r"[a-z]+", needle) if len(w) > 2}
    meaningful = query & _catalogue_vocabulary()

    if not meaningful:
        # The reference names no product at all — the model passed the order
        # ID, a positional label ("item_1"), or an empty string. If the order
        # has exactly one line item there is nothing to disambiguate, so
        # resolve to it rather than asking "which of these did you mean?" of
        # an order with one item. That question is nonsense to a customer and
        # it deadlocks the conversation.
        if len(order["items"]) == 1:
            return order["items"][0], "sole_item"
        return None, "none"

    matches = [
        item for item in order["items"]
        if meaningful <= {_stem(w) for w in re.findall(r"[a-z]+", item["name"].lower())}
    ]
    if len(matches) == 1:
        return matches[0], "name"

    # A real product word that doesn't match anything on this order ("a jacket"
    # on a kurta order) is a contradiction, not a missing reference — ask,
    # even if the order has only one item.
    return None, "none"


def _item_summary(order: dict) -> list[dict]:
    return [{"item_id": i["sku"], "name": i["name"]} for i in order["items"]]


def _days_since(iso_datetime: str | None) -> int | None:
    if not iso_datetime:
        return None
    then = datetime.fromisoformat(iso_datetime.replace("Z", "+00:00"))
    return (_now() - then).days


def _hours_since(iso_datetime: str | None) -> float | None:
    if not iso_datetime:
        return None
    then = datetime.fromisoformat(iso_datetime.replace("Z", "+00:00"))
    return (_now() - then).total_seconds() / 3600


def _days_past_expected(expected_delivery: str | None) -> int | None:
    if not expected_delivery:
        return None
    expected = datetime.fromisoformat(expected_delivery).replace(tzinfo=timezone.utc)
    return (_now() - expected).days


def _verdict(outcome: str, reason_code: str, customer_message: str, **extra) -> dict:
    result = {
        "outcome": outcome,
        "reason_code": reason_code,
        "customer_message": customer_message,
    }
    result.update(extra)
    return result


# ---------- TOOL 1: lookup_order ----------

@tool
def lookup_order(order_id: str) -> dict:
    """Look up an order by its order ID (format TR-XXXX). Returns status,
    items, dates, carrier and tracking. Use this whenever the customer asks
    about an order's status, contents or delivery. Requires a real order ID —
    orders cannot be looked up by customer name."""
    order = _find_order(order_id)
    if not order:
        return {
            "found": False,
            "customer_message": (
                f"I couldn't find an order with the ID {order_id}. Could you "
                "double-check it? It looks like TR-4521 and is in your "
                "confirmation email."
            ),
        }

    days_late = _days_past_expected(order.get("expected_delivery"))

    return {
        "found": True,
        "order_id": order["order_id"],
        "status": order["status"],
        "placed_at": order["placed_at"],
        "delivered_at": order["delivered_at"],
        "expected_delivery": order["expected_delivery"],
        "days_past_expected_delivery": days_late,
        "carrier": order["carrier"],
        "tracking_number": order["tracking_number"],
        "payment_method": order["payment_method"],
        "shipping_city": order["shipping_city"],
        # No customer name or contact details: the agent never needs them to
        # do its job, and their absence is what stops them being disclosed.
        "items": [
            {
                "item_id": i["sku"],
                "name": i["name"],
                "category": i["category"],
                "size": i.get("size"),
                "qty": i.get("qty"),
                "price": i.get("price"),
                "final_sale": i.get("final_sale", False),
                "shipped": i.get("shipped"),
                "backorder_eta": i.get("backorder_eta"),
            }
            for i in order["items"]
        ],
        "total": order["total"],
        "refund_status": order.get("refund_status"),
    }


# ---------- TOOL 2: search_policy ----------

@tool
def search_policy(topic: str) -> dict:
    """Search Trendly's shipping and returns policy for the section covering a
    topic (e.g. 'refund timeline for UPI', 'express shipping cost', 'damaged
    item'). Returns the policy text verbatim as grounding — reason over that
    text and never answer a policy question from your own knowledge. If it
    returns NOT FOUND, say you can't confirm it and offer a human agent."""
    return {"topic": topic, "policy_text": search_policy_section(topic)}


# ---------- TOOL 3: check_return_eligibility ----------

@tool
def check_return_eligibility(order_id: str, item_id: str, issue_type: str) -> dict:
    """Decide whether an item can be returned or exchanged, applying the full
    policy (return window, non-returnable categories, final sale, footwear,
    damaged-item rules, cancelled and lost orders) to the order data.

    issue_type must be one of: change_of_mind, wrong_size, damaged, defective,
    wrong_item. Pick the one matching what the customer actually said —
    damaged/defective/wrong_item follow a different policy path with a
    48-hour reporting window.

    Always call this before initiate_return, and always use its `outcome` and
    `customer_message` rather than judging eligibility yourself."""
    order = _find_order(order_id)
    if not order:
        return _verdict(
            "not_eligible", "order_not_found",
            f"I couldn't find an order with the ID {order_id}.",
        )

    item, match_kind = _find_item(order, item_id)
    if not item:
        return _verdict(
            "not_eligible", "item_not_found",
            "I want to make sure I'm looking at the right item before I check "
            "this — which of these did you mean?",
            items_on_order=_item_summary(order),
            agent_note=(
                "Do not retry with a guessed item_id. Ask the customer to "
                "choose from items_on_order, then call this tool again."
            ),
        )

    issue = (issue_type or "change_of_mind").strip().lower()
    category = item["category"].lower()

    # 1. Cancelled — policy 2.6
    if order["status"] == "cancelled":
        return _verdict(
            "not_eligible", "order_cancelled",
            "That order was cancelled before it shipped and the refund has "
            "already been processed, so there's nothing to return against it.",
        )

    # 2. Lost in transit — policy 1.6. Not a return; a human handles it.
    if order["status"] == "lost_in_transit":
        return _verdict(
            "escalate", "lost_parcel_claim",
            "The carrier has marked this parcel as lost. That's handled as a "
            "lost-parcel claim rather than a return — a colleague will pick "
            "this up and sort out a free replacement or a full refund, "
            "whichever you prefer, within 5 business days.",
        )

    # 3. Not delivered yet — nothing to return
    if not order["delivered_at"]:
        return _verdict(
            "not_eligible", "not_delivered",
            f"This order hasn't been delivered yet (it's currently "
            f"{order['status'].replace('_', ' ')}), so a return can't be "
            "raised on it yet. The return window opens once it arrives.",
        )

    # 4. Damaged / defective / wrong item — policy 6, which OVERRIDES the
    #    non-returnable categories in 2.3 and has its own 48-hour window.
    if issue in ISSUE_TYPES_DAMAGE:
        hours = _hours_since(order["delivered_at"])
        if hours is not None and hours <= DAMAGE_REPORT_HOURS:
            return _verdict(
                "escalate", "damaged_within_report_window",
                "I'm sorry that arrived in that condition. Because this was "
                "reported within 48 hours of delivery, you're covered for "
                "either a free replacement or a full refund including "
                "shipping — whichever you'd prefer. A colleague will pick "
                "this up to collect the photographs we need, since I can't "
                "take images here.",
                hours_since_delivery=round(hours, 1),
            )
        return _verdict(
            "not_eligible", "damage_report_window_expired",
            f"Damaged or incorrect items need to be reported within 48 hours "
            f"of delivery, and this one arrived about {int((hours or 0) // 24)} "
            "days ago, so I can't raise it under that policy. I can pass this "
            "to a colleague if you'd like them to take a look.",
            hours_since_delivery=round(hours, 1) if hours is not None else None,
            agent_note=(
                "Offer a human agent. Do not silently fall through to a "
                "change-of-mind return without the customer asking for one."
            ),
        )

    # 5. Non-returnable category — policy 2.3 (hygiene/safety, date-independent)
    if category in NON_RETURNABLE_CATEGORIES:
        return _verdict(
            "not_eligible", "non_returnable_category",
            f"{item['name']} falls under our {item['category']} category, "
            "which we can't accept back for hygiene and safety reasons — that "
            "applies regardless of how recently it arrived. If it turned up "
            "damaged or it's the wrong item, tell me and I'll handle it "
            "differently.",
        )

    # 6. Return window — policy 2.1
    days = _days_since(order["delivered_at"])
    if days is not None and days > RETURN_WINDOW_DAYS:
        return _verdict(
            "not_eligible", "outside_return_window",
            f"This was delivered {days} days ago, which is past our 30-day "
            "return window, so I'm not able to raise a return on it.",
            days_since_delivery=days,
        )

    # 7. Final sale — policy 2.4: size exchange only
    if item.get("final_sale"):
        return _verdict(
            "exchange_only", "final_sale",
            f"{item['name']} was a final sale item, so it's eligible for a "
            "size exchange only — I can't issue a refund or store credit on "
            "it. If you tell me the size you need, I can raise the exchange.",
            days_since_delivery=days,
        )

    # 8. Eligible. Footwear carries the shoe-box condition — policy 2.5
    extra_condition = None
    if category == "footwear":
        extra_condition = (
            f"Footwear needs to come back in its original shoe box — without "
            f"it there's a ₹{FOOTWEAR_NO_BOX_DEDUCTION} deduction from the refund."
        )

    message = (
        f"Good news — {item['name']} is eligible for return. It was delivered "
        f"{days} days ago, well inside the 30-day window."
    )
    if extra_condition:
        message += " " + extra_condition

    return _verdict(
        "eligible_refund", "within_policy", message,
        days_since_delivery=days,
        conditions=extra_condition,
    )


# ---------- TOOL 4: initiate_return ----------

@tool
def initiate_return(
    order_id: str,
    item_id: str,
    resolution: str,
    reason: str,
    new_size: str = "",
) -> dict:
    """Create the return or exchange. Only call this after
    check_return_eligibility returned outcome 'eligible_refund' or
    'exchange_only' for this exact order and item.

    resolution must be 'refund' or 'exchange'. Use 'exchange' when the
    outcome was exchange_only, or when the customer asked for a different
    size, and pass new_size. Never call this speculatively."""
    order = _find_order(order_id)
    if not order:
        return {"success": False, "error": "Order not found."}

    item, _ = _find_item(order, item_id)
    if not item:
        return {"success": False, "error": "Item not found on this order."}

    resolution = (resolution or "refund").strip().lower()

    open_key = (order["order_id"], item["sku"])
    existing_return_id = _OPEN_RETURNS.get(open_key)
    if existing_return_id:
        return {
            "success": False,
            "error": "already_initiated",
            "return_id": existing_return_id,
            "customer_message": (
                f"I've already raised this for your {item['name']} — the "
                f"reference is {existing_return_id}, so there's no need to "
                "start it again. Let me know if you'd like a status update "
                "on that instead."
            ),
        }

    if resolution == "exchange":
        if not new_size:
            return {
                "success": False,
                "error": "missing_size",
                "customer_message": "Which size would you like instead?",
            }
        exc_id = f"EXC-{order['order_id']}-{item['sku']}"
        _OPEN_RETURNS[open_key] = exc_id
        return {
            "success": True,
            "return_id": exc_id,
            "type": "size_exchange",
            "item": item["name"],
            "new_size": new_size,
            "customer_message": (
                f"Done — I've raised a size exchange for your {item['name']} "
                f"in size {new_size}. We'll arrange a free reverse pickup; "
                "you'll get a text to choose a slot. If that size turns out "
                "to be unavailable we'll convert it to a refund automatically."
            ),
        }

    payment_method = order["payment_method"]
    timelines = {
        "credit_card": "5–7 business days back to your card",
        "prepaid_card": "5–7 business days back to your card",
        "upi": "3–5 business days back to your UPI ID",
        "cash_on_delivery": "7–10 business days by bank transfer or store credit",
    }
    timeline = timelines.get(payment_method, "per the timelines in our refund policy")

    ret_id = f"RET-{order['order_id']}-{item['sku']}"
    _OPEN_RETURNS[open_key] = ret_id
    result = {
        "success": True,
        "return_id": ret_id,
        "type": "refund_return",
        "item": item["name"],
        "customer_message": (
            f"All set — I've raised the return for your {item['name']}. "
            "We'll arrange a free reverse pickup and you'll get a text to pick "
            "a slot. Once it reaches the warehouse and passes inspection "
            f"(2–3 business days), your refund goes out — {timeline}."
        ),
    }

    if payment_method == "cash_on_delivery":
        result["agent_note"] = (
            "COD refunds need bank details, which a human collects over a "
            "secure link (policy 3.3). Never ask for them in chat. Mention "
            "that a colleague will send the secure link."
        )
    return result


# ---------- TOOL 5: apply_delayed_credit ----------

@tool
def apply_delayed_credit(order_id: str) -> dict:
    """Check for and issue the ₹250 delayed-order store credit (policy 1.5)
    on an order more than 3 days past its expected delivery date. The
    customer does not need to cancel to receive it. Call this whenever a
    customer with a delayed order asks what can be done, or when lookup_order
    shows an order past its expected delivery date."""
    order = _find_order(order_id)
    if not order:
        return _verdict(
            "not_eligible", "order_not_found",
            f"I couldn't find an order with the ID {order_id}.",
        )

    # Lost parcels are a claim, not a delay (policy 1.6). Without this the
    # tool happily issued a ₹250 credit on a lost order and the model then
    # offered to cancel it for a refund — an action it has no tool for and no
    # authority to promise. The status check belongs here, not in the prompt.
    if order["status"] == "lost_in_transit":
        return _verdict(
            "escalate", "lost_parcel_claim",
            "The carrier has marked this parcel as lost, which we handle as a "
            "lost-parcel claim rather than a delay. A colleague will pick this "
            "up and sort out a free replacement or a full refund — whichever "
            "you'd prefer — within 5 business days.",
        )

    if order["status"] in ("delivered", "cancelled"):
        return _verdict(
            "not_eligible", "not_in_transit",
            "That order isn't currently delayed in transit, so the "
            "delayed-delivery credit doesn't apply to it.",
        )

    days_past = _days_past_expected(order.get("expected_delivery"))
    if days_past is None:
        return _verdict(
            "not_eligible", "no_expected_date",
            "There's no expected delivery date on record for this order, so I "
            "can't assess it for the delay credit — let me get a colleague to "
            "check.",
        )

    if days_past <= DELAY_THRESHOLD_DAYS:
        return _verdict(
            "not_eligible", "within_threshold",
            f"This order is {days_past} day(s) past its estimate, which is "
            "still inside the 3-day grace period before the delay credit "
            "kicks in. It should be with you shortly.",
            days_past_expected=days_past,
        )

    credit_id = f"CR-{order['order_id']}-DELAY"
    if credit_id in _ISSUED_DELAY_CREDITS:
        return _verdict(
            "already_issued", "duplicate_request",
            f"The ₹{DELAY_CREDIT_AMOUNT} delay credit has already been added "
            "to your account for this order — it should show at checkout.",
            credit_id=credit_id,
        )

    _ISSUED_DELAY_CREDITS.add(credit_id)
    return _verdict(
        "issued", "delayed_past_threshold",
        f"This is {days_past} days past the estimate, which I'm sorry about. "
        f"I've added a ₹{DELAY_CREDIT_AMOUNT} store credit to your account for "
        "the delay — you don't need to cancel anything to keep it, and the "
        "order is still on its way.",
        credit_amount=DELAY_CREDIT_AMOUNT,
        credit_id=credit_id,
        days_past_expected=days_past,
    )


# ---------- TOOL 6: escalate_to_human ----------

@tool
def escalate_to_human(summary: str, reason: str, order_id: str = "") -> dict:
    """Hand off to a human agent. Use when the request is outside what you can
    do (lost parcels, damaged-item photos, discounts, bank details, disputes),
    when a tool fails repeatedly, or when the policy doesn't cover the
    question. `summary` must be written for the agent picking this up: what
    the customer wants, what you already established, and what's left to do."""
    ticket_id = f"ESC-{order_id or 'GEN'}-{_now().strftime('%d%H%M%S')}"

    # Attach the order snapshot so the human doesn't repeat the lookup. This
    # is the difference between a ticket someone can action and a ticket
    # someone has to re-investigate.
    context = None
    order = _find_order(order_id) if order_id else None
    if order:
        context = {
            "order_id": order["order_id"],
            "status": order["status"],
            "placed_at": order["placed_at"],
            "expected_delivery": order["expected_delivery"],
            "delivered_at": order["delivered_at"],
            "carrier": order["carrier"],
            "tracking_number": order["tracking_number"],
            "payment_method": order["payment_method"],
            "order_total": order["total"],
            "items": _item_summary(order),
        }

    return {
        "escalated": True,
        "ticket_id": ticket_id,
        "reason": reason,
        "summary_for_agent": summary,
        "order_context": context,
        "customer_message": (
            f"I've passed this to a colleague who can help — your reference is "
            f"{ticket_id}. They'll be in touch during support hours, 9 AM to "
            "9 PM IST, seven days a week."
        ),
    }


ALL_TOOLS = [
    lookup_order,
    search_policy,
    check_return_eligibility,
    initiate_return,
    apply_delayed_credit,
    escalate_to_human,
]
