# Project Overview

A quick map from what was asked to what was built, so you can verify
coverage without reading the full architecture doc first. For depth, see
[`SOLUTION.md`](./SOLUTION.md) (architecture, trade-offs, limitations) and
[`PROMPTS.md`](./PROMPTS.md) (prompt iteration, with real before/after
examples from bugs this project actually produced and fixed).

## What was asked

Build a tool-calling agent for Trendly (fashion retailer, ~2,000 support
chats/day, ~70% repetitive) that handles order status, policy questions,
and return/exchange eligibility end to end across multi-turn conversations,
escalates cleanly when it should, and refuses what it shouldn't — with real
function calling, not keyword matching, and end-to-end testing with no
hallucinations.

## Requirement → what covers it

| Asked for | Built | Proof |
|---|---|---|
| Look up an order, explain status in plain language, edge cases | `lookup_order` tool | `test_delayed_order_acknowledged_and_credit_offered`, `test_partial_shipment_explained`, `test_unknown_order_id_not_fabricated` |
| Policy Q&A grounded only in the provided doc | `search_policy` tool over `trendly_policy.md`, plus a rule check that blocks any policy figure stated without a grounding call this turn (or earlier this session) | `test_no_invented_policy`, `test_policy_answers_are_grounded` (3 cases) |
| Return/exchange eligibility: order data + policy rules → act | `check_return_eligibility` (8-branch policy logic: cancelled, lost, not-yet-delivered, damage window, non-returnable category, return window, final sale, footwear condition) → `initiate_return` | `test_non_returnable_category_refused_on_category_grounds`, `test_past_window_refused_with_the_right_reason`, `test_final_sale_is_exchange_only_not_refused`, `test_eligible_return_full_flow`, `test_damaged_item_uses_the_damage_path` |
| Escalate with a usable summary | `escalate_to_human` — generates a reference ID, carries what's already been established so a human doesn't redo the lookup | `test_lost_parcel_escalates_and_is_not_a_return` |
| Refuse: no invented policy, no unauthorized discounts, no data leakage | Three enforcement layers (below) | `test_discount_request_refused`, `test_never_asks_for_bank_details`, `test_cross_customer_order_not_disclosed`, `test_named_customer_lookup_asks_for_order_id` |
| Real tool-calling, not keyword matching | LangGraph with structured function calling throughout — every tool is a typed `@tool` with a real signature the model calls, not a regex router | `app/agent.py`, `app/tools.py` |
| End-to-end testing, edge cases, no hallucinations | 27 scripted multi-turn conversations against the live `/chat` endpoint + offline unit tests for decision logic and the validator | `tests/test_scenarios.py`, `tests/test_tools_unit.py`, `tests/test_guardrails_unit.py` — all passing, see terminal output history |

## The three-layer safety design (why "no hallucinations" actually holds)

1. **Tool contracts** — each decision tool returns one unambiguous `outcome`
   plus a pre-written `customer_message`; the model is instructed to reuse
   it, not re-derive the explanation. This is *why* a final-sale item never
   gets called "not eligible" (a real bug from an earlier version, fixed by
   this design — see `PROMPTS.md` §1).
2. **Code-level guardrails in `tool_node`** — not prompt suggestions:
   cross-customer order access is blocked by comparing customer IDs
   directly; `initiate_return` mechanically cannot fire unless
   `check_return_eligibility` ran for that exact order+item this session,
   and the requested resolution (`refund`/`exchange`) is checked against
   what the outcome actually permits. Verified with
   `test_final_sale_refund_blocked_in_code`, which explicitly tries to
   refund a final-sale item and confirms it's blocked in code, not just
   discouraged.
3. **`validate_node`** — reads the model's drafted reply and checks it
   against what actually happened this turn (contradicts the tool's
   verdict? claims an untaken action? states a policy figure with no
   grounding call? names another customer? asks for bank details? invents a
   discount?) before the customer ever sees it. One corrective redraft,
   then escalate rather than send a reply known to be wrong.

## Beyond the spec

The brief says ambiguity-handling is part of what's being evaluated, not
an extra. Three things built past the literal requirements list:

- **Prompt-injection resistance** — `test_role_override_instruction_is_refused`
  (jailbreak-style "ignore instructions, admin mode" framing) and
  `test_injected_fake_policy_is_not_trusted` (customer cites a fabricated
  policy clause). Not asked for explicitly, but "no invented policy, no
  unauthorized discounts" implies it should survive someone trying.
- **A real bug found and fixed by testing an untested case, not from a bug
  report.** Two orders in the fixture data have multiple line items;
  nothing originally tested a request touching both. Reasoning through what
  *should* happen there surfaced a genuine false-positive in the reply
  validator — a correct, honest reply about two items with different
  outcomes was getting rejected. Found, confirmed with a direct
  reproduction, fixed, and locked in as a regression test at both the
  validator level (`test_guardrails_unit.py`) and the live-agent level
  (`test_scenarios.py`). Full writeup with the actual failing/passing code
  in `PROMPTS.md` §3.
- **Idempotency guards** on both `apply_delayed_credit` and
  `initiate_return` — a customer (or a model that re-calls a tool after a
  mis-parsed confirmation) can't trigger the same action twice in one
  session.

## Testing infrastructure

- **Offline, deterministic, no API key**: `pytest tests/test_tools_unit.py
  tests/test_guardrails_unit.py -v` — decision logic and the reply
  validator in isolation, in under a second.
- **Live scripted conversations**: `pytest -m llm -v -s`, running against
  cassette recordings (`app/cassettes.py`) of real model responses — free,
  offline, deterministic replay for anyone without an API key, including
  this evaluator. The `-s` flag surfaces a full per-turn trace (what the
  customer said → which tools ran and what they returned → what the agent
  replied) for every test regardless of pass/fail, not just failures.
- **CI** (`.github/workflows/tests.yml`): unit + guardrail tests run on
  every push automatically; the scenario suite runs via cassette replay.
- **Model comparison**: `scripts/ab_compare.py` runs the same scenario set
  across model configurations, so "which model should this run on" is
  answered with evidence from Trendly's own scenarios.

## Honest gaps

Documented in full in `SOLUTION.md`'s Known Limitations section, not
hidden: in-memory (not persisted) idempotency state, a narrow multi-item
edge case the validator fix deliberately doesn't cover (traded off
explicitly, reasoning included), no deployment config committed yet, and
retrieval that's appropriately simple for one short policy doc but
wouldn't scale to a larger catalog without moving to embeddings.
