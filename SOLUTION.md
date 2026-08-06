# Solution note

## Architecture

A LangGraph state machine with four nodes and seven tools. State persists per
session in SQLite, so a conversation survives a page refresh or a server
restart.

```
                    ┌──────────┐
   customer ───────▶│  agent   │──── tool calls ────▶┌───────┐
                    │  (model) │◀─── tool results ───│ tools │
                    └────┬─────┘                     └───┬───┘
                         │ no tool calls                 │ guardrails
                         ▼                               │ blocked → reason
                    ┌──────────┐                         │ fed back
                    │ validate │── clean ──▶ customer    │
                    │  (reply) │── rejected ──▶ redraft ─┘
                    └────┬─────┘── twice ──▶ escalate
                         │
                    ┌────▼──────────┐
                    │force_escalate │  tool failures, step limit
                    └───────────────┘
```

**Tools.** `verify_customer`, `lookup_order`, `search_policy`,
`check_return_eligibility`, `initiate_return`, `apply_delayed_credit`,
`escalate_to_human`.

**The organising idea: decisions in code, wording from the model.** Every
policy verdict is computed in `tools.py` and returned as an `outcome` plus a
`customer_message` the model reuses. Every precondition is enforced in the
graph. The model chooses which tool to call and adapts tone — nothing else.

The evidence for that split is in the failure history: every wrong answer this
project produced came from the model *describing* a correct verdict
incorrectly, and every fix that stuck moved a decision out of the prompt. The
one that makes the point best: on a lost parcel, the model reached for the
delay credit — the order genuinely is 30 days late, so the arithmetic invites
it — and the tool refused and routed to the claim path instead. Prompt said
one thing, model tried another, code decided.

---

## Key decisions and trade-offs

**Verdict + message, not a verdict alone.** Returning
`{"eligible": true, "exchange_only": true}` let the model pick a field and
guess; it told a customer a final-sale item was "not eligible for return".
Returning one `outcome` and the sentence to say fixed it. *Trade-off:* replies
are more uniform and less conversational than a free-running model would
produce. For refusals and money, that's the right side to err on.

**Preconditions satisfied, not requested.** When `check_return_eligibility`
was called before the order was established, the graph used to bounce it back
with "call lookup_order first". The model retried the same blocked call and
escalated. The graph now resolves the order itself. *Trade-off:* one fewer
place the model is forced to demonstrate it understood — recovered by moving
the safety property into `_find_item`, which returns real line items instead
of guessing.

**Verification as a hard gate.** Order tools are blocked until
`verify_customer` succeeds, and exactly one code path can bind a session — with
a test asserting that. It replaced a scheme that bound the session to whichever
order was mentioned first, which prevented wandering but never established
identity. *Trade-off:* every conversation costs an extra turn.

**Reply validation.** A deterministic check between draft and customer:
contradictions, unsupported action claims, ungrounded policy figures,
cross-customer names, sensitive-data requests, discounts. One redraft, then
escalate. *Trade-off:* false positives. It has had exactly one, fixed by
scoping grounding to the conversation while keeping action claims per-turn.

**Two test layers.** Offline suites cover policy logic, validation and
verification with no model; the live suite covers whether the model uses them
correctly. This came from a real problem — a test asserting `escalated is
True` passed during a total rate-limit outage, because everything escalated.
Now a failure tells you immediately which half broke.

**Record/replay cassettes.** Model calls are recorded once and replayed
thereafter, so the full suite runs offline, deterministically, in seconds, and
CI needs no API key. Free-tier limits (15 requests/minute) made a live suite
unusable as a gate.

**Provider as configuration.** Swapping Groq for Gemini touched the model
constructor, two type annotations, and one serialisation boundary. The graph,
tools, guardrails and tests were untouched. That's the payoff of keeping
decisions in code — and it surfaced three things worth knowing: Gemini rejects
`anyOf` in function declarations (so optional tool parameters need
flattening), it returns content as blocks rather than a string, and the two
SDKs cannot currently share a Python environment at all, because
`langchain-groq` is a major version behind on `langchain-core`. The
abstraction holds; the ecosystem doesn't. The Groq path is kept, with its
dependency isolated in `requirements-groq.txt`.

**Three access tiers.** The customer sees their own conversation and account.
`/session/{id}` is protected only by an unguessable id. Operator routes
(`/dashboard`, `/history`, `/metrics`, `/sessions`) require a token or a local
request, and 404 otherwise so their existence isn't advertised.

---

## What it does

Across twelve scripted conversations (37 turns) in the last full pass: **33
turns closed without a human, 4 escalated** — two policy-mandated (a lost
parcel and a genuine cross-customer dispute), two from free-tier rate limits
rather than agent limitations. Zero guardrail failures; zero validation
false positives.

Deflection is visible live at `/dashboard`, split into policy-mandated
handoffs (the system working as designed) and agent limitations (a backlog to
fix) — the distinction an ops lead actually needs.

Edge cases handled: partial shipments, delayed orders with the ₹250 credit
(once, idempotently), lost parcels as claims rather than returns, cancelled
orders, non-returnable categories, final-sale exchange-only, damaged items
overriding the hygiene exclusion inside 48 hours, ambiguous item references,
and orders belonging to someone else.

---

## Known limitations

**Verification is identification, not authentication.** Knowing an email
address opens an account. Real deployment needs an OTP to the contact on file,
or a session token from an already-signed-in app. `/session/{id}` has the same
shape of weakness: anyone holding the id can replay that conversation.

**Business days are approximated as calendar days.** Policy 1.2 and 1.5 both
say "business days"; the code counts calendar days. No holiday calendar, and
the dataset gives no way to infer one.

**Idempotency is in-memory.** Duplicate returns and delay credits are blocked
per process, not persisted. A restart or a second worker would forget.

**Footwear's shoe-box rule is implemented but untested.** The only footwear
order in the dataset never reaches the delivered state, so the ₹300 deduction
branch can't be exercised against the fixed data.

**The validator only sees one turn's tools.** A reply that refers to an action
from a previous turn can't be verified against it, only against the fact that
*some* grounding happened earlier.

**The deployed instance is a demonstration, not a deployment.** Render's free
tier sleeps after ~15 minutes and runs on an ephemeral disk, so conversation
state and the event log are wiped on every restart — the dashboard reads zero
on a cold instance. Everything is single-process: the SQLite checkpointer, the
in-memory idempotency guards and the throttle all assume one worker. Scaling
out needs a shared store for all three.

**Free-tier limits shape the experience.** 15 requests/minute on Flash-Lite,
20/day on 3.6 Flash, and a turn costs 2–6 calls. The agent paces itself,
honours the provider's retry hints and falls back to a second model, but under
sustained load it degrades to escalation. Latency is 2–8s per turn, most of it
model time.

**Cost per conversation is unmeasured.** At 2,000 chats a day the token cost
of a 6-call turn is a real budget line, and I haven't quantified it.

---

## Five questions for Trendly's ops team

**1. How is a customer authenticated before an order is discussed?** Is there
a session token from the app, or does support genuinely start from an email
address? This decides whether the verification step is a real control or
theatre — and it's the single biggest gap in what I've built.

**2. What does "business days" mean operationally?** Which holiday calendar,
and is it uniform across metros and partner-serviced pincodes? Every date
calculation in the return window, delivery estimate and delay threshold
depends on the answer.

**3. Is the ₹250 delay credit once per order, or once per delay event?** A
parcel delayed twice, or one order split across two shipments — does the
customer get ₹250 or ₹500? Section 1.5 doesn't say, and it's the only
money-issuing power the agent has.

**4. What's the actual intake for damaged-item photographs, and what's the SLA
once a ticket is raised?** The agent can't accept images, so every damaged
claim becomes a handoff. If photos arrive by email or WhatsApp, that's a
different integration and a different deflection ceiling.

**5. What's the per-conversation cost ceiling, and what's the current cost per
human-handled chat?** A 70% deflection target is only worth hitting if the
agent is cheaper than the alternative. That number also decides how much
model to buy — this workload runs on a small model precisely because the
decisions live in code.

---

## What I'd do next

1. **Real authentication** — OTP or session token. Everything else is downstream.
2. **Persist idempotency and the delay-credit ledger** on the order record.
3. **A retrieval eval set** for `search_policy` — labelled queries with
   recall@1, so "I improved retrieval" becomes a number.
4. **Cost per conversation** in the dashboard, alongside deflection.
5. **Policy hot-reload with a coverage report** — swap `trendly_policy.md`
   without redeploying, and flag clauses no tool implements. It would flag the
   footwear box rule today.
