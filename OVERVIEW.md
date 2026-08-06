# Features

Everything this project does, and why each piece exists. Written to be read
top to bottom — the live URLs are at the end.

---

## 1. The seven tools

The agent has no free-form knowledge of Trendly. Everything it knows comes
from a tool call.

| Tool | What it does |
|---|---|
| `verify_customer` | Matches an email or phone against the account. Nothing order-related runs until this succeeds |
| `lookup_order` | Status, items, dates, carrier, tracking. Deliberately returns **no** customer name or contact details |
| `search_policy` | Returns the relevant section of `trendly_policy.md` verbatim, as grounding |
| `check_return_eligibility` | Applies the whole return policy to one item and returns a single verdict |
| `initiate_return` | Creates the return or exchange. Only runs when eligibility permits it |
| `apply_delayed_credit` | Issues the ₹250 delay credit under policy 1.5, once per order |
| `escalate_to_human` | Hands off with a ticket, an order snapshot, and a routing reason |

**Every decision tool returns the same shape:** an `outcome` (the decision), a
`reason_code` (why), and a `customer_message` (the wording). The model reuses
that sentence rather than inventing its own explanation. This came from a real
failure — given `{"eligible": true, "exchange_only": true}`, the model told a
customer a final-sale item was "not eligible for return". Moving the
*explanation* out of the model, not just the decision, fixed a whole class of
wrong answers.

---

## 2. Policy coverage

Every branch of the policy document is implemented and tested against every
order in the dataset:

- **Return window** — 30 calendar days from delivery, not from order date
- **Non-returnable categories** — jewellery, innerwear, socks, beauty, refused on hygiene grounds regardless of date
- **Final sale** — size exchange only, never a refund
- **Damaged or wrong items** — a separate path with a 48-hour reporting window, which **overrides** the hygiene exclusion
- **Footwear** — returnable, with the ₹300 no-shoe-box deduction surfaced up front
- **Lost parcels** — a claim for a human, explicitly *not* a return and *not* a delay
- **Cancelled orders** — nothing to return against
- **Partial shipments** — explains what shipped, what's on backorder, and the ETA
- **Delayed orders** — the ₹250 credit, issued once, with no need to cancel
- **Refund timelines** — card, UPI, COD and store credit each get their own answer

---

## 3. Orchestration

A LangGraph state machine with four nodes:

- **agent** — the model, deciding what to call next
- **tools** — executes tool calls, and enforces the guardrails
- **validate** — checks the drafted reply before the customer sees it
- **force_escalate** — terminal path for repeated tool failures or a step limit

State persists per session in SQLite, so a conversation survives a page
refresh *and* a server restart. Conversation-scoped values (who's verified,
which orders have been seen, which eligibility decisions were made) carry
forward; per-turn counters reset.

---

## 4. Guardrails, enforced in code

The prompt describes these too, but the prompt isn't what holds the line.

**Verification gate.** `lookup_order`, `check_return_eligibility`,
`initiate_return` and `apply_delayed_credit` are blocked until
`verify_customer` succeeds. The agent won't even confirm an order *exists*
beforehand, because that leaks too.

**One binding path.** Exactly one line of code can attach a session to a
customer — a successful verification — and a test asserts it stays that way.

**Cross-customer refusal.** A verified customer asking about someone else's
order is refused in code. It doesn't confirm the order exists, doesn't say who
placed it, and doesn't ask them to confirm it's theirs — it can't verify that.
Asking about the *same* order twice escalates as a dispute; asking about a
different one doesn't, because that's a new question, not persistence.

**Eligibility before action.** `initiate_return` won't run without a matching
eligibility outcome, and the resolution must match it — a final-sale item
cannot be refunded even if the model tries.

**No invented items.** Item references resolve against the catalogue
vocabulary. "Leather jacket" on an order containing a Woven Leather Belt is
refused as a contradiction, not silently matched.

**Identity never reaches the model.** `lookup_order` doesn't return the
customer name. A field the model never receives can't be disclosed.

---

## 5. Reply validation

Guardrails constrain what the agent may **do**. This constrains what it may
**say** — because every wrong answer in this project came from the model
describing a *correct* verdict incorrectly.

Before a reply is sent, it's checked against the same turn's tool results for:

- contradicting the outcome it's reporting
- claiming an action no tool performed
- stating a policy figure with no grounding call anywhere in the conversation
- naming a customer other than the verified one
- soliciting bank or card details
- offering a discount

One corrective redraft; a second failure escalates rather than sending
something known to be wrong. It's deterministic — no second model call — so it
costs nothing and can't itself hallucinate.

---

## 6. Scope discipline

The agent declines poems, arithmetic, general knowledge and anything else
outside Trendly support, warmly and in one sentence. Account and data requests
(changing an email, closing an account) are explained first and *then*
escalated — never handed a bare ticket number.

The reasoning matters more than the rule: an agent that does arithmetic on
request is one a customer will keep testing, and every answer outside the
policy document is ungrounded by definition.

---

## 7. Reliability

**Model fallback.** Primary model, then a second with a separate quota, then
an honest handoff. A degraded answer beats no answer; an honest handoff beats
a stack trace.

**Provider-aware backoff.** Reads the provider's own suggested retry delay
rather than guessing. Short waits are honoured; a 60-second wait falls through
to the other model instead of stalling the conversation.

**Throttling.** Free tiers cap requests per *minute*, so scripted runs pace
themselves rather than collapsing into 429s.

**Idempotency.** A return or delay credit can't be raised twice for the same
item — a retry, a double-click or a model repeat is refused with the original
reference.

**Two clocks, deliberately separate.** `TRENDLY_NOW` freezes "today" for the
dataset, keeping return-window scenarios stable. Support hours use the real
wall clock, because whether a colleague is at their desk is a fact about now.
Conflating them made the agent promise "we're open until 9 PM" at 3 AM.

**Support-hours awareness.** Escalations say "shortly — we're open until 9 PM
IST" during hours, and "they'll pick this up when we open at 9 AM IST
tomorrow" outside them.

---

## 8. The customer interface

- **Verification-first.** The panel stays locked until the account is confirmed.
- **Account panel.** Name, verified badge, masked email and phone, and their
  orders with colour-coded status chips. Click an order to ask about it.
- **Quick actions** that change once verified — track an order, return an
  item, exchange a size, refund status, damaged item, policy, talk to a person.
- **Conversation persistence.** Refresh the page, come back later, it's still
  there — replayed from the server, not the browser.
- **End chat**, and a session that belongs to one customer for its lifetime.
- **No internal state on screen.** No session IDs, latencies, tool traces or
  validation counts — that's operator information.

---

## 9. Testing

**Two layers, deliberately separate.** Offline suites cover the rules; the
live suite covers whether the model uses them correctly. When something fails
you know immediately which half broke.

- `tests/test_tools_unit.py` — every policy branch against every order
- `tests/test_guardrails_unit.py` — reply validation and metrics
- `tests/test_verification.py` — contact matching, masking, the gate, support hours
- `tests/test_admin_routes.py` — operator routes stay closed
- `tests/test_scenarios.py` — live conversations end to end

All four offline suites run with **no API key, no network, in about a second**.

**Cassettes.** Model calls are recorded once and replayed after, so the live
suite runs offline and deterministically — which is what CI uses. This came
from a real problem: during a rate-limit outage, a test asserting "this should
escalate" passed because *everything* escalated. A false green.

**`scripts/e2e.py`** — three stages: offline suites, HTTP surface (including
verifying the guardrails exist as code by inspecting the graph), and the live
conversations, which are opt-in because they cost quota.

**`scripts/demo.py`** — twelve curated conversations printed as readable
transcripts with tool calls inline. Pass/fail says whether it's correct; this
says whether it sounds like support worth deploying.

**`scripts/ab_compare.py`** — the same scenarios across models, with pass
rate, escalation rate and latency side by side.

---

## 10. Observability

Every turn appends one JSONL line: tools called, outcomes, blocked calls,
validation retries, escalation reason, model used, whether it fell back, and
latency.

That feeds a **deflection rate** split into *policy-mandated* handoffs (the
system working as designed) and *agent limitations* (the actual backlog) —
the distinction an ops lead needs, and the one that makes the 70% figure in
the brief measurable rather than aspirational.

---

## 11. Three access tiers

| Who | Sees |
|---|---|
| **Customer** | Their own conversation and account, once verified |
| **Session holder** | A conversation replay, protected only by an unguessable session ID |
| **Operator** | Every conversation across all customers — token required |

Operator routes return **404** rather than 403 without a token, so an
unauthenticated caller doesn't learn the surface exists. The customer page
links to none of it, and a test asserts that.

---

## Live URLs

**Customer chat** — no token needed, this is the deliverable:

<https://trendly-agent-z7ll.onrender.com>

**Operator dashboard** — deflection rate, escalation split, guardrail
activity, latency percentiles:

<https://trendly-agent-z7ll.onrender.com/dashboard?token=IhaNx0IQEVmdWL9Sd_-iDXcjl9NxPdMS8f2v52qTSoM>

**Conversation history** — every session, grouped by customer, with full
transcripts and the tool calls behind each reply:

<https://trendly-agent-z7ll.onrender.com/history?token=IhaNx0IQEVmdWL9Sd_-iDXcjl9NxPdMS8f2v52qTSoM>

**Chat with the account switcher** — one-click sign-in as any seeded account,
for evaluation:

<https://trendly-agent-z7ll.onrender.com/?token=IhaNx0IQEVmdWL9Sd_-iDXcjl9NxPdMS8f2v52qTSoM>

The token is needed once per browser — it sets a cookie after that.

Two notes on the hosted instance. It's on a free tier, so it sleeps after
~15 minutes idle and takes 30–60 seconds to wake; that's the platform, not the
agent. And the disk is ephemeral, so a restart clears transcripts and the
dashboard reads zero. Running it locally with your own key avoids both — see
the README.
