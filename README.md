# Trendly Support Agent

**Live:** <https://trendly-agent-z7ll.onrender.com>
**Operator views:** [dashboard](https://trendly-agent-z7ll.onrender.com/dashboard?token=IhaNx0IQEVmdWL9Sd_-iDXcjl9NxPdMS8f2v52qTSoM) · [conversation history](https://trendly-agent-z7ll.onrender.com/history?token=IhaNx0IQEVmdWL9Sd_-iDXcjl9NxPdMS8f2v52qTSoM)
**Account switcher** (evaluation convenience): [https://trendly-agent-z7ll.onrender.com/?token=…](https://trendly-agent-z7ll.onrender.com/?token=IhaNx0IQEVmdWL9Sd_-iDXcjl9NxPdMS8f2v52qTSoM)

The customer chat at `/` needs no token. The operator routes show every
conversation across all customers, so they're gated — 404 without a valid
token, deliberately, so the surface isn't discoverable. The token is needed
once per browser; it sets a cookie after that.

Free instance: it sleeps after ~15 minutes idle, so the first request after a
quiet spell takes 30–60 seconds to wake. That's the platform, not the agent.

A tool-calling support agent for Trendly, a direct-to-consumer fashion
retailer. It handles order status, returns, exchanges, refunds and policy
questions across multi-turn conversations, and hands off to a human when it
should — with a summary a person can actually act on.

Built on LangGraph with real function calling. The design principle
throughout: **decisions live in code, not in the prompt.** The model chooses
which tool to call and how to phrase the answer; every policy verdict, every
precondition and every refusal is enforced in the graph, where it can be
tested without an API key.

---

See **[FEATURES.md](FEATURES.md)** for a walkthrough of everything the agent
does, **[PROMPTS.md](PROMPTS.md)** for how the prompt got to its current shape,
and **[SOLUTION.md](SOLUTION.md)** for architecture, trade-offs and limitations.

## Quick start

```bash
git clone <your-repo-url>
cd trendly-agent

python -m venv .venv
source .venv/bin/activate          # Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

cp .env.example .env               # then add your API key
uvicorn app.main:app --reload
```

Open <http://127.0.0.1:8000>.

A free Gemini key from [Google AI Studio](https://aistudio.google.com/apikey)
is enough. Groq works too — set `LLM_PROVIDER=groq`.

**You can run the whole test suite without any API key.** See
[Testing](#testing).

---

## Configuration

Only `GEMINI_API_KEY` is required. Everything else has a working default.

| Variable | Default | What it does |
|---|---|---|
| `LLM_PROVIDER` | `gemini` | `gemini` or `groq` |
| `GEMINI_API_KEY` | — | Google AI Studio key (`GOOGLE_API_KEY` also accepted) |
| `GROQ_API_KEY` | — | Only needed when `LLM_PROVIDER=groq` |
| `PRIMARY_MODEL` | `gemini-3.5-flash-lite` | Main model |
| `FALLBACK_MODEL` | `gemini-3.6-flash` | Used when the primary is rate-limited |
| `GEMINI_THINKING_LEVEL` | `medium` | Flash-Lite's `minimal` default causes premature tool termination |
| `TRENDLY_NOW` | unset | Freezes "today" so return-window outcomes stay stable |
| `MIN_MODEL_INTERVAL_MS` | `0` | Minimum gap between model calls; scripted runs set 4500 |
| `MAX_RETRY_WAIT_S` | `12` | Cap on honouring a provider's suggested retry delay |
| `ADMIN_TOKEN` | unset | Guards the operator routes; unset means localhost-only |
| `CASSETTE_MODE` | `off` | `record` / `replay` / `auto` for offline conversation tests |
| `EVENT_LOG` | `logs/events.jsonl` | Where turn-level telemetry is written |
| `SESSIONS_DB` | `sessions.db` | LangGraph checkpoint store |

**A note on `TRENDLY_NOW`.** `orders.json` was authored with fixed distances
from "today" — TR-4523 is annotated *"well outside the 30-day window"*. Left
on real time, TR-4522 leaves the return window on 13 August and TR-4528 on 18
August, silently changing what the scenarios demonstrate. Setting
`TRENDLY_NOW=2026-08-05T12:00:00Z` preserves the cases the dataset was built
to exercise. Remove it for real behaviour.

---

## Endpoints

**Customer-facing**

| Route | Purpose |
|---|---|
| `GET /` | Chat interface |
| `POST /chat` | `{session_id, message}` → reply, tool calls, state |
| `GET /health` | Liveness. Returns `{"status":"ok"}` and nothing more to a customer |
| `GET /session/{id}` | Replays a conversation so the page survives a refresh |
| `GET /session/{id}/customer` | The signed-in customer's own profile and orders |

**Operator-facing** — every conversation across all customers, so these are
gated. With `ADMIN_TOKEN` unset they answer local requests only and 404
elsewhere; with it set, a matching token is required via `X-Admin-Token`,
`?token=…`, or the cookie the page sets on first use. 404 rather than 403, so
an unauthenticated caller doesn't learn the surface exists.

| Route | Purpose |
|---|---|
| `GET /dashboard` | Deflection rate, escalation split, guardrail activity, latency |
| `GET /history` | Every session, grouped by customer, with full transcripts |
| `GET /metrics` | The same numbers as JSON |
| `GET /sessions` | Session index |

The customer page links to none of them, and a test asserts it.

---

## Testing

Two layers, deliberately separated. When something fails you want to know
immediately whether the **rules** are wrong or the **model** is.

**Offline — no API key, no network, about a second**

```bash
pytest tests/test_tools_unit.py tests/test_guardrails_unit.py \
       tests/test_verification.py tests/test_admin_routes.py -v
```

Every policy branch against every order in the dataset, the reply validator,
contact matching and the verification gate, and the admin routes.

**End-to-end runner**

```bash
python -m scripts.e2e            # stages 1 and 2 — free
python -m scripts.e2e --live     # adds the conversations
```

Stage 1 runs the offline suites. Stage 2 checks the HTTP surface: route
gating, data exposure at the tool boundary, and that the verification gate and
single session-binding path exist *as code*, by inspecting the graph. Stage 3
runs twelve scripted conversations and is opt-in because it spends quota.

**Live scenarios**

```bash
pytest tests/test_scenarios.py -v
```

Paced at 4.5s between model calls, because free Gemini tiers cap requests per
*minute* (15 on Flash-Lite). Without pacing the run collapses into 429s and
every assertion fails against a "model unavailable" escalation — a false red
as misleading as a false green.

**Cassettes — run the live suite without a key**

```bash
CASSETTE_MODE=record pytest tests/test_scenarios.py   # once, with a key
CASSETTE_MODE=replay pytest tests/test_scenarios.py   # thereafter, free
```

Model calls are recorded to `tests/cassettes/` and replayed. This is what CI
runs.

**Reading conversations rather than assertions**

```bash
python -m scripts.demo --list
python -m scripts.demo --only happy_path final_sale
```

Prints transcripts with tool calls, guardrail blocks and validation results
inline. Pass/fail tells you whether the agent is correct; this tells you
whether it sounds like support worth deploying.

**Comparing models**

```bash
python -m scripts.ab_compare --cassettes replay
```

Same scenarios across model configurations, with pass rate, escalation rate
and latency side by side.

---

## Layout

```
app/
  agent.py          LangGraph graph, guardrails, retry and fallback
  tools.py          Seven tools; all policy logic lives here
  prompts.py        System prompt (see PROMPTS.md for how it got there)
  policy_store.py   Policy retrieval over trendly_policy.md
  validation.py     Reply validation — what the agent may *say*
  observability.py  Turn-level event log and metric aggregation
  cassettes.py      Record/replay for model calls
  main.py           FastAPI routes, customer UI, operator UI
scripts/
  demo.py           Conversation transcripts for reading
  e2e.py            Three-stage end-to-end runner
  ab_compare.py     Provider/model comparison
tests/              Four offline suites plus the live scenario suite
data/               orders.json and trendly_policy.md, unmodified
```

---

## Deployment

Deployed on Render's free tier. The repo also runs with one command, and for
evaluation that's the better path — no cold start, no shared rate limit, and
the operator views open without a token.

**To deploy your own:**

1. Push to GitHub, then Render → New → Web Service → connect the repo.
   - Build: `pip install -r requirements.txt`
   - Start: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
   - Health check: `/health`
2. Set `GEMINI_API_KEY` and `ADMIN_TOKEN` in Render's Environment tab, plus
   `PYTHON_VERSION=3.12.7`. `render.yaml` documents the rest.

**Dependency notes, learned the hard way.** Four clean-build failures, each a
real constraint that a working local environment had been hiding:

- Python 3.14 has no wheels yet for several packages here — pinned to 3.12.
- `httpx` had to move from `==0.27.2` to `>=0.28.1`; `google-genai` requires it.
- `langgraph 0.2.x` pins `langchain-core<0.4`, which `langchain-google-genai 4.x`
  can't satisfy. Upgraded langgraph rather than downgrade the Google SDK.
- `langchain-groq 0.2.0` pins `langchain-core<0.4` too, and has no release that
  works with 1.x. It is **not** in `requirements.txt` — see
  `requirements-groq.txt`. The Groq code path still exists and its import is
  local to the function, so nothing breaks; it just can't share an environment
  with the Gemini SDK today.

The last one is worth stating plainly: the provider abstraction is sound —
swapping vendors touched a constructor, two annotations and one serialisation
boundary — but the two SDKs currently can't co-exist, because `langchain-groq`
is a major version behind on `langchain-core`.

**What to expect from a free instance.** It sleeps after ~15 minutes idle and
takes 30–60s to wake, so the first request after a quiet spell is slow — that's
the platform, not the agent. Conversation state lives in SQLite on an ephemeral
disk, so a restart clears transcripts and metrics; the code doesn't care, but
the dashboard will read zero afterwards. And the Gemini free tier allows 15
requests a minute *in total*, so concurrent visitors will see the agent fall
back to its second model and, under real load, escalate. Running it locally
with your own key avoids all three.

## Known constraints

- **Free-tier rate limits.** Flash-Lite allows 15 requests/minute, 3.6 Flash
  20/day. A turn costs 2–6 model calls. Scripted runs set
  `MIN_MODEL_INTERVAL_MS`; the agent honours the provider's own suggested
  retry delay and falls back to the other model, which has a separate quota.
- **Verification is identification, not authentication.** Knowing an email is
  enough to open an account. Real deployment needs an OTP or a session token
  from an already-signed-in app. Discussed in SOLUTION.md.
- **In-memory idempotency.** Duplicate returns and delay credits are blocked
  per process, not persisted.

---

## AI usage

Claude was used heavily and directly, mostly as a reviewer and pair on this
codebase rather than a code generator working from a blank page.

**Written by me:** the original architecture and tool design, the LangGraph
graph structure, the guardrail concept, the first scenario suite, the
idempotency guard on `initiate_return`, the CI workflow, the prompt-injection
and retrieval-robustness tests, multi-item scenarios, verbose test tracing,
and the rejected-draft visibility in diagnostics. I also found the multi-item
mixed-outcome bug in the reply validator.

**Generated by Claude, reviewed and integrated by me:** the eligibility
verdict shape (`outcome` + `customer_message`), the policy retrieval rewrite,
item-reference matching, most of the current system prompt, the Gemini
migration, reply validation, cassettes, observability, the dashboard and
history browser, the verification gate, admin gating, and the e2e runner.

Every generated change was reviewed, run against the test suites, and in
several cases corrected — a few are documented in PROMPTS.md, including two
where a "fix" silently no-oped and only a failing test caught it. I can
explain and modify any part of this.
