# Solution Note

## Architecture

```
customer message
      |
      v
 +----------+     tool calls      +-----------+
 | agent_node| ------------------> | tool_node |
 | (LLM +    |                     | (guardrails|
 |  tools)   | <------------------ |  + tools) |
 +----------+     tool results     +-----------+
      |
      | draft reply
      v
 +--------------+   violations   +------------------+
 | validate_node| -------------> | correction retry |
 |  (rule-based |                | (1x) then        |
 |   checks)    | -- clean ----> | escalate_to_human|
 +--------------+                +------------------+
      |
      v
 reply to customer
```

Built on LangGraph. Session state (`session_customer_id`,
`looked_up_orders`, `eligibility_outcomes`, `session_grounded`) persists
across turns via the checkpointer; per-turn counters
(`iteration`, `validation_retries`, `consecutive_failures`) reset each turn.

**Six tools**, each returning a single unambiguous `outcome` plus a
pre-written `customer_message` (see `PROMPTS.md` §1 for why): `lookup_order`,
`search_policy`, `check_return_eligibility`, `initiate_return`,
`apply_delayed_credit`, `escalate_to_human`.

**Three enforcement layers**, deliberately not relying on any one of them
alone:
1. **Tool contracts** — the tool decides and explains; the model routes.
2. **`tool_node` guardrails** — code-level, not prompt-level: cross-customer
   lookups are blocked by comparing `session_customer_id` against the
   order's real owner; `initiate_return` is refused unless
   `check_return_eligibility` ran for that exact order+item this session and
   the requested resolution matches what the outcome actually permits
   (`ALLOWED_RESOLUTIONS = {"eligible_refund": {refund, exchange},
   "exchange_only": {exchange}}`). A final-sale refund attempt is refused in
   code, not just discouraged in the prompt — verified by
   `test_final_sale_refund_blocked_in_code`, which explicitly asks for one.
3. **`validate_node`** — catches what the first two can't: the model
   describing a correct decision incorrectly in English. Checks the drafted
   reply against the turn's actual tool results before the customer sees it
   (contradiction, unsupported action claim, ungrounded policy figure,
   cross-customer name leak, solicited bank details, invented discount). One
   corrective redraft, then escalate rather than send a reply known to be
   wrong.

**Testing**: `tests/test_tools_unit.py` and `test_guardrails_unit.py` cover
decision logic and the validator offline, deterministically, in under a
second — no model calls. `tests/test_scenarios.py` runs ~27 scripted
multi-turn conversations against the live `/chat` endpoint. Cassette
record/replay (`app/cassettes.py`) means these run free and offline for a
reviewer once recorded — real model responses fingerprinted on the model
name + full message history, so an edit anywhere in that history invalidates
the cassette rather than silently serving a stale one.
`scripts/ab_compare.py` runs the same scenario set across model
configurations to answer "which model should this run on?" with evidence
from Trendly's own data rather than a general leaderboard.

## Key trade-offs

**Policy retrieval is keyword + hand-tuned synonym matching, not
embeddings.** For a single, short, static policy document, embeddings are
disproportionate engineering for the actual retrieval problem, and keyword
matching is trivially debuggable — you can read exactly why a query matched
or didn't. The real risk is coverage: a customer phrasing that shares no
words with the doc and isn't in the synonym map. Tested this directly
against phrasings not literally in the map ("will I get my money back to my
wallet" → correctly finds Refunds) and confirmed the failure mode is safe —
an ambiguous case ("courier never shows up") returns `NOT FOUND` rather than
guessing a wrong section. This would not scale to a real multi-page policy
catalog; embeddings become the right call well before that point.

**The `tool_node` guardrails changed shape once, deliberately.** The
eligibility-before-action guardrail originally *bounced* a premature
`initiate_return` call back to the model with a text instruction to call
`check_return_eligibility` first. That relies on the model reading a
correction and retrying correctly — it didn't; it retried the same blocked
call and escalated instead. The fix was to make the precondition
mechanically satisfiable: the graph already has the order ID, so it resolves
the order itself rather than asking the model to. The safety property this
guardrail existed for (no invented item IDs) now lives in `_find_item`,
which returns the real line items instead of letting the model guess — this
is also what drives `test_ambiguous_item_reference_is_queried_not_guessed`.

**One redraft, then escalate — not a longer retry loop.** A second failed
validation attempt escalates rather than trying a third time. Longer retry
chains multiply latency and API cost for diminishing returns, and an honest
handoff to a human beats sending a reply that's already failed automated
review twice.

**The multi-item validator fix trades a rare miss for a common false
positive.** Documented in code and in `PROMPTS.md` §3: skipping the
contradiction check when a turn has more than one eligibility result means a
genuine contradiction across two items in one turn wouldn't be caught. That
was accepted deliberately — the alternative (attempting per-item phrase
attribution without the tool results carrying item identity) is real
additional plumbing, and this project's actual production bugs were false
rejections of correct replies, not undetected genuine ones.

## Known limitations

- **In-memory idempotency, not persisted.** `_ISSUED_DELAY_CREDITS` and
  `_OPEN_RETURNS` (both in `app/tools.py`) are process-local sets/dicts.
  They correctly stop a duplicate action within one running process, but a
  restart loses that state, and a real deployment needs this on the order
  record itself, not in application memory.
- **Multi-item validator gap**, as above — a genuine contradiction spanning
  two items in the same turn currently isn't caught by `validate_node`.
- **Retrieval doesn't scale past this document's size** — see trade-offs.
- **No conversation-length cap.** `MAX_ITERATIONS = 6` bounds tool-call
  loops within a single turn, but nothing bounds how long a session can run
  or how large its message history grows, which is a real cost and latency
  consideration for a long-running support chat.
- **Two providers, both free-tier.** The system supports both Groq and
  Gemini with a same-provider fallback model on failure
  (`agent_node`'s retry loop), but there's no cross-provider fallback if an
  entire provider is down — worth revisiting for production.
- **Cassette fingerprinting is exact-match.** A cassette matches only an
  identical conversation prefix (model name + full message history). This
  is deliberate — it stops a stale recording being served silently after a
  prompt or logic change — but it means any prompt edit requires
  re-recording the whole affected scenario, not just the changed part.

## Five discovery questions for Trendly's ops team

1. **What's the actual distribution of the 70% "repetitive" volume across
   order status, returns/exchanges, and policy questions** — and within
   returns, how often is it single-item versus multi-item? The fixture data
   here has 2 multi-item orders out of 10; if that ratio is much higher in
   real volume, the multi-item validator gap above moves from "known
   limitation" to "fix before launch."
2. **Who actually resolves an `escalate_to_human` ticket, and on what
   system?** The `escalate_to_human` tool generates a reference ID and a
   summary, but that's designed against a guess of what a human agent needs
   — I don't know if it's landing in a ticketing tool, a shared inbox, or
   something else, and the summary format should match that destination.
3. **Is `TRENDLY_NOW` (the frozen clock used for testing) something ops
   would ever want exposed as a real "as-of" query** — e.g., for auditing
   why a specific return was refused on a specific date — or is it purely a
   testing artifact that should never exist in production?
4. **What's the actual free-text volume this needs to handle** — do real
   customers ask compound questions in one message ("where's my order AND
   can I also ask about your return policy"), or is that rare enough not to
   prioritize? I tested multi-item returns but not mixed-intent-category
   messages, and don't know if that's a real pattern.
5. **What happens when a customer disputes an eligibility outcome?** Policy
   is applied deterministically here (window, category, final-sale), but
   real support conversations sometimes involve a customer with a
   legitimate exception the written policy doesn't anticipate. Is there a
   defined override path, or does every dispute simply become an escalation
   — and if the latter, is that an acceptable volume of escalations for the
   ops team to absorb?
