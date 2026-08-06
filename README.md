# Trendly Support Agent

A tool-calling support agent for Trendly, a direct-to-consumer fashion
retailer. Handles order status, policy questions, and return/exchange
eligibility end to end, and hands the rest to a human cleanly. Built on
LangGraph with real function calling (not keyword matching), a rule-based
reply validator that checks every draft against what actually happened
before it reaches the customer, and cassette-based offline testing so the
full scenario suite runs without an API key.

See [`SOLUTION.md`](./SOLUTION.md) for architecture and trade-offs, and
[`PROMPTS.md`](./PROMPTS.md) for how the prompts and guardrails were
iterated on, with real before/after examples from bugs this project
actually produced and fixed.

## Quick start

**Requirements:** Python 3.12+, and a free API key from either
[Groq](https://console.groq.com/keys) or
[Google AI Studio](https://aistudio.google.com/apikey) (Gemini).

```bash
git clone <this-repo-url>
cd trendly-agent
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env             # then fill in your API key
uvicorn app.main:app --reload
```

The chat UI is at `http://localhost:8000/` — a live view of the graph's
session state (looked-up orders, eligibility decisions, model used,
blocked guardrail calls) sits alongside the conversation, plus preset
scenario buttons for a quick walkthrough. `/dashboard` shows aggregated
metrics once some turns have run.

`start.sh` does the install + launch in one command if you'd rather not
set up the venv manually:
```bash
PORT=8000 ./start.sh
```

### `.env` reference

```dotenv
LLM_PROVIDER=gemini              # "gemini" or "groq" — both free-tier
GEMINI_API_KEY=...
GROQ_API_KEY=...
PRIMARY_MODEL=gemini-3.5-flash-lite
FALLBACK_MODEL=gemini-3.6-flash
TRENDLY_NOW=2026-08-05T12:00:00Z # freezes "today" so return-window outcomes
                                  # (30 days, 48-hour damage window, etc.)
                                  # stay stable instead of drifting daily
GEMINI_THINKING_LEVEL=medium
```

## Running the tests

Two layers, deliberately separated by speed and cost:

```bash
# Offline, deterministic, no API key — decision logic + the reply validator
pytest tests/test_tools_unit.py tests/test_guardrails_unit.py -v

# Live scripted conversations against the real agent graph
pytest -m llm -v -s
```

The second command needs `CASSETTE_MODE` set. `record` hits the real model
once and saves every response as a cassette under `tests/cassettes/`;
`replay` reruns the same suite offline, free, and deterministically from
those recordings — this is how a reviewer runs the full suite without an
API key, once cassettes are committed:

```bash
CASSETTE_MODE=record pytest -m llm -v      # first time, or after a prompt/logic change
CASSETTE_MODE=replay pytest -m llm -v -s   # every time after
```

`-s` matters for the second layer — without it, pytest only shows the
per-turn trace (customer message → tools that ran → agent's reply) for
tests that *fail*. With it, every test prints its trace regardless of
pass/fail, so you can read straight through and see exactly what the agent
did on every scripted conversation, not just the ones that broke.

For a plain-English read-through instead of pytest output:
```bash
python -m scripts.demo --cassettes auto
```

## Other useful commands

```bash
python -m scripts.ab_compare --cassettes replay    # compare model configs on the same scenarios
python -m scripts.demo --list                      # see available demo conversations
```

## Deploying / live endpoint

This runs anywhere that can run `uvicorn app.main:app`. No deployment
config (Dockerfile, Procfile, etc.) is committed yet — **this is a known
gap, not an oversight**: pick a target and add it before submitting if a
live URL is required, rather than the repo-runs-in-one-command fallback the
brief also accepts (`./start.sh` after `.env` is filled in).

## AI usage note

Built with Claude assisting throughout — drafting tool docstrings and
guardrail logic, reasoning through edge cases, writing and running tests,
and helping fix bugs the tests surfaced. Architectural decisions (what each
guardrail enforces, what a tool's contract looks like, which failures were
worth locking in as regression tests) were mine, and every fix was verified
before being accepted — either against the real model or by isolating the
exact logic and running it directly. `PROMPTS.md` documents specific
examples of this iteration, including a validator bug found by reasoning
through an untested case (multi-item requests) rather than from a bug
report.

## Repo layout

```
app/
  main.py          FastAPI surface: /chat, /health, /metrics, /dashboard, /
  agent.py          LangGraph: agent_node, tool_node (guardrails), validate_node
  tools.py          The 6 tools: lookup_order, search_policy,
                     check_return_eligibility, initiate_return,
                     apply_delayed_credit, escalate_to_human
  validation.py     Rule-based reply checks run before a draft reaches the customer
  policy_store.py   Keyword + synonym-map retrieval over trendly_policy.md
  cassettes.py      Record/replay for offline, deterministic, free testing
  observability.py  Turn-by-turn JSONL logging + /metrics aggregation
  prompts.py        The system prompt
data/
  orders.json           10 fixed orders (provided, loaded as-is)
  trendly_policy.md     Shipping & returns policy (provided, sole source of truth)
scripts/
  demo.py           Curated conversation transcripts, for reading not asserting
  ab_compare.py     Same scenarios across model configs, side by side
tests/
  test_tools_unit.py       Decision logic, offline, no model calls
  test_guardrails_unit.py  Reply validator, offline, no model calls
  test_scenarios.py        Scripted multi-turn conversations against /chat
  cassettes/               Recorded model responses for offline replay
.github/workflows/tests.yml  CI: unit+guardrails always, scenarios via cassette replay
```
