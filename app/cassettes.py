"""
Record/replay for model calls.

The scenario suite hits a live model, which made it slow (10-23 minutes),
non-deterministic, and — on free tiers with a 20-request daily cap — usually
impossible to finish. Worse, a rate-limited run produced *false greens*: tests
asserting `escalated is True` passed because everything escalated.

So model calls are recorded once to tests/cassettes/*.json and replayed
afterwards. The suite then runs offline, in seconds, deterministically, and a
reviewer can run it without an API key at all.

Modes (CASSETTE_MODE):
    off     — always call the real model (default)
    record  — call the real model and save every response
    replay  — never call the model; a cassette miss is an error
    auto    — replay when a cassette exists, record when it doesn't

Keys hash the model name plus the full message history, so a cassette is only
reused for an identical conversation prefix. Change the prompt and the keys
change, which is correct: a stale recording would silently test the old
behaviour.
"""

import hashlib
import json
import os
from pathlib import Path

from langchain_core.messages import AIMessage

CASSETTE_DIR = Path(os.environ.get("CASSETTE_DIR", "tests/cassettes"))


def mode() -> str:
    return os.environ.get("CASSETTE_MODE", "off").strip().lower()


def _text_of(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for b in content:
            if isinstance(b, str):
                parts.append(b)
            elif isinstance(b, dict) and b.get("type") == "text":
                parts.append(b.get("text", ""))
        return "\n".join(parts)
    return str(content or "")


def _fingerprint(model_name: str, messages) -> str:
    """
    Canonical representation of a request. Deliberately ignores provider
    metadata (thought signatures, token counts, request IDs) so a cassette
    recorded on one provider can be matched on another with the same
    conversation — which is what makes the A/B harness cheap to re-run.
    """
    parts = [f"model={model_name}"]
    for m in messages:
        kind = getattr(m, "type", m.__class__.__name__)
        entry = {"type": kind, "content": _text_of(getattr(m, "content", ""))}
        tool_calls = getattr(m, "tool_calls", None)
        if tool_calls:
            entry["tool_calls"] = [
                {"name": tc.get("name"), "args": tc.get("args")} for tc in tool_calls
            ]
        if getattr(m, "name", None):
            entry["name"] = m.name
        parts.append(json.dumps(entry, sort_keys=True, ensure_ascii=False))
    blob = "\n".join(parts)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:24]


def _path_for(key: str) -> Path:
    return CASSETTE_DIR / f"{key}.json"


def _serialize(response: AIMessage) -> dict:
    return {
        "content": _text_of(getattr(response, "content", "")),
        "tool_calls": [
            {"name": tc.get("name"), "args": tc.get("args"), "id": tc.get("id")}
            for tc in (getattr(response, "tool_calls", None) or [])
        ],
    }


def _deserialize(data: dict) -> AIMessage:
    tool_calls = [
        {
            "name": tc["name"],
            "args": tc.get("args") or {},
            "id": tc.get("id") or f"replay_{i}",
            "type": "tool_call",
        }
        for i, tc in enumerate(data.get("tool_calls") or [])
    ]
    return AIMessage(content=data.get("content") or "", tool_calls=tool_calls)


class CassetteLLM:
    """
    Wraps a tool-bound chat model, preserving the .invoke() contract the graph
    depends on. Transparent when CASSETTE_MODE is off.
    """

    def __init__(self, inner, model_name: str):
        self._inner = inner
        self.model_name = model_name
        self.hits = 0
        self.misses = 0

    def invoke(self, messages):
        current = mode()
        if current == "off":
            return self._inner.invoke(messages)

        key = _fingerprint(self.model_name, messages)
        path = _path_for(key)

        if current in ("replay", "auto") and path.exists():
            self.hits += 1
            with open(path, encoding="utf-8") as f:
                return _deserialize(json.load(f))

        if current == "replay":
            raise RuntimeError(
                f"Cassette miss for {self.model_name} (key {key}). Re-record "
                "with CASSETTE_MODE=record, or check whether the system prompt "
                "changed since the cassettes were made."
            )

        # record / auto-miss: call the real model and save the result
        self.misses += 1
        response = self._inner.invoke(messages)
        try:
            CASSETTE_DIR.mkdir(parents=True, exist_ok=True)
            payload = _serialize(response)
            payload["_meta"] = {"model": self.model_name, "key": key}
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
        except Exception as e:  # pragma: no cover
            print(f"[cassettes] failed to save {key}: {e!r}")
        return response

    def __getattr__(self, item):
        # Anything else (bind_tools, with_config, ...) passes through.
        return getattr(self._inner, item)
