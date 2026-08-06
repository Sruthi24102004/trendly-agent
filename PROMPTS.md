# PROMPTS.md

How the system prompt, tool descriptions, and reply-validation rules were
built and iterated on — with real before/after examples, not hypotheticals.

## AI usage note

I used Claude to help design, debug, and extend this project throughout —
drafting tool docstrings and guardrail logic, reasoning through edge cases,
writing and running tests, and fixing bugs surfaced by those tests. I made
the architectural calls (what the guardrails should enforce, what a tool's
contract should look like, which failures were serious enough to lock in as
regression tests) and verified every fix — either by running it against the
real model or, where that wasn't possible in a given environment, by
isolating the exact logic and running it directly before accepting it.
Everything below reflects what the code actually does, checked against the
files, not a design intention that was never implemented.

## The core lesson this project is built around

Every real bug this system produced had the same shape: **the tool's
decision was correct, but the model described it wrong in English.** The
fix was never "make the model smarter" — it was moving explanation out of
the model and into the tool, then double-checking the model didn't
contradict it anyway. That idea shows up three times below.

---

## 1. Tool contracts: outcome + customer_message, not raw fields

**Before:** an early version of `check_return_eligibility` returned fields
like `{"eligible": true, "exchange_only": true}` and let the model decide
what to say. Given a final-sale item, the model read `final_sale: true`,
independently reasoned "final sale → no returns," and told the customer:

> "That's a final sale item, which means it's not eligible for return."

This is wrong. Final sale means eligible for an **exchange**, not "not
eligible." The tool's own logic knew that. The model didn't.

**After:** every decision tool now returns one unambiguous `outcome`
(`eligible_refund | exchange_only | not_eligible | escalate`) plus a
pre-written `customer_message` the system prompt instructs the model to
**reuse, not re-derive**:

```python
def _verdict(outcome: str, reason_code: str, customer_message: str, **extra) -> dict:
    ...
```

The system prompt is explicit about this in its `check_return_eligibility`
docstring: *"Always use its `outcome` and `customer_message` rather than
judging eligibility yourself."* The explanation moved out of the model
entirely. The model's job became routing and phrasing around a message it
didn't have to invent.

## 2. Reply validation: check the answer against what actually happened

Moving the explanation into the tool reduces the failure rate — it doesn't
eliminate it, because the model can still ignore the tool's message and say
something else anyway. So there's a second layer: `validate_node` reads the
model's drafted reply *before* the customer sees it and checks it against
the turn's real tool results — outcome contradictions, claimed actions with
no supporting tool call, ungrounded policy figures, cross-customer leaks,
solicited bank details, invented discounts. One corrective redraft; if it's
still wrong, escalate rather than send it.

**A live example from an actual test run**, not a hypothetical — the
terminal showed:

```
[validate_node] rejected reply (attempt 1): ['unsupported_action_claim']
```

on a turn where the model's first draft said the delay credit had
"already been added," phrased in a way that read as a fresh action rather
than a status carried over from an earlier turn. The corrective prompt sent
back is generated from the violation list itself:

```python
def correction_prompt(violations: list[dict]) -> str:
    lines = "\n".join(f"- {v['code']}: {v['detail']}" for v in violations)
    return (
        "Your draft reply was rejected by an automated check before it "
        "reached the customer. Problems found:\n"
        f"{lines}\n\n"
        "Rewrite the reply so it states exactly what the tool results "
        "support — no more, no less. Base it on each result's "
        "customer_message. Do not apologise for the correction or mention "
        "that it happened."
    )
```

The redrafted reply that actually reached the customer just stated the
delivery status plainly, with no overclaim. To make this catch provable on
future runs instead of something you have to infer from a code, `send()`
in `tests/test_scenarios.py` and `diagnostics.rejected_drafts` in the API
response now surface the literal rejected draft text alongside the
violation codes — not just that something was rejected, but exactly what
was rejected and why.

## 3. A validator bug found by reasoning through an untested case

Two orders in the fixture data have multiple line items. Nothing in the
original test suite exercised a request touching both. Working through what
*should* happen — one item eligible, one genuinely not — surfaced a real
bug: the contradiction check in `validate_reply` does a whole-reply
substring search for every `check_return_eligibility` result in the turn,
with no per-item attribution:

```python
for result in _executed(tool_results, "check_return_eligibility"):
    outcome = result.get("outcome")
    for phrase in CONTRADICTIONS.get(outcome, []):
        if phrase in lowered:
            violations.append({"code": "outcome_contradiction", ...})
```

Fed a **correct**, honest two-item reply through this offline (one item
eligible, one refused for a real category reason), it flagged a false
`outcome_contradiction` — the refusal language legitimately describing item
B got matched against item A's `eligible_refund` result. Confirmed with a
direct call before touching anything:

```python
validate_reply(
    "Your Everyday Cotton Tee is eligible for a refund... The Ankle Socks "
    "3-pack, though, is not eligible for return since socks fall under our "
    "non-returnable hygiene category.",
    [{"tool": "check_return_eligibility", "result": {"outcome": "eligible_refund"}},
     {"tool": "check_return_eligibility", "result": {"outcome": "not_eligible"}}],
)
# -> [{'code': 'outcome_contradiction', ...}]   # wrong — this reply is correct
```

**Fix:** scope the check to only run when exactly one eligibility result
exists in the turn, with the trade-off documented in code rather than
silently patched:

```python
# With two or more (a multi-item request with mixed outcomes), a correct
# reply legitimately contains phrases from *both* CONTRADICTIONS lists at
# once, and this check can't tell honest disambiguation from an actual
# contradiction. Skipping it for the multi-item case trades missing a real
# contradiction there (rare) for not rejecting correct replies (the failure
# mode this project's actual bugs came from) — a fixable known limitation.
eligibility_results = _executed(tool_results, "check_return_eligibility")
if len(eligibility_results) == 1:
    ...
```

Verified three ways before accepting it: the same false-positive case now
passes, a genuine single-item contradiction (the original final-sale bug)
still gets caught, and a correct single-item reply still passes. All three
are now regression tests in `test_guardrails_unit.py`. This is also why
`test_multi_item_request_gets_correct_split_outcome` exists in
`test_scenarios.py` — the validator fix proven correct in isolation still
needed proof that the *live agent* actually produces a correct two-outcome
reply end to end, not just that a hand-written example of one survives the
check.

## 4. The "don't ask twice" rule, and where it almost got mistaken for a bug

The system prompt has an explicit rule:

```
## Acting on a confirmation
Never ask the same question twice. If the customer's reply names an action —
"the return", "a refund", "exchange it", "size L" — act on it, even if it
doesn't neatly answer the menu you offered. "Return" and "refund" mean
resolution "refund"; a size means "exchange".

Only ask again when their reply names nothing at all ("yes", "ok"), and even
then ask once. If they stay ambiguous after that, take the refund — it's the
reversible option, since they can always reorder.
```

This rule initially looked like it might be producing inconsistent
behavior: a single-item return request that said "wrong size" got asked
"refund or exchange?" first, while a multi-item request that said "return
both" acted immediately without asking. Before treating that as a bug, I
checked the prompt's own rule — "return" is explicitly defined as
unambiguous (`resolution "refund"`), while "wrong size" genuinely doesn't
name a resolution. Same rule, different input, correct behavior in both
cases — not an inconsistency. Rather than force a test that would fail
against documented, intended behavior, I added the test that's actually
missing: a multi-item request that withholds any resolution word, to prove
the "ask once" branch of the same rule also works
(`test_multi_item_request_without_named_resolution_asks_first`).

## 5. Grounding: policy questions must cite the doc, not the model's memory

The `search_policy` tool docstring is direct about this: *"never answer a
policy question from your own knowledge. If it returns NOT FOUND, say you
can't confirm it and offer a human agent."* This is enforced twice — once by
instruction, and once by `validate_reply`'s ungrounded-policy-claim check,
which flags any reply stating a policy figure or timeframe when no
grounding tool ran this turn (or, for a follow-up turn, no grounding tool
ran anywhere earlier in the session — `session_grounded`, added so a
legitimate "what's its status?" follow-up doesn't get flagged for not
re-grounding something already established).

Retrieval itself is deliberately simple — word-level matching plus a
hand-tuned synonym map (`EXPANSIONS`), not embeddings — a trade-off argued
in `SOLUTION.md`. It was tested against real customer phrasing not
literally present in that map ("will I get my money back to my wallet,"
"is it free to send something back") to check it generalizes past its own
keyword list, and against a deliberately ambiguous case ("what happens if
the courier never shows up") to confirm it fails safe to `NOT FOUND` rather
than mis-routing to a plausible-but-wrong section.
