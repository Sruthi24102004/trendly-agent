"""
Loads trendly_policy.md and splits it into labeled sections so search_policy
can return just the relevant section(s) instead of the whole document.

Retrieval is word-level with a small synonym map. The previous version scored
by whole-string substring count, so any multi-word topic ("refund timeline",
"cash on delivery refund") scored zero and returned NOT FOUND — which the
system prompt turns into an escalation. That made the agent escalate on
ordinary, answerable policy questions.
"""

import re
from pathlib import Path

POLICY_PATH = Path(__file__).parent.parent / "data" / "trendly_policy.md"

# Terms that carry no retrieval signal.
STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "do", "does", "did", "can",
    "could", "will", "would", "should", "my", "me", "i", "you", "your", "it",
    "for", "of", "on", "in", "to", "and", "or", "if", "how", "what", "when",
    "long", "much", "many", "get", "got", "have", "has", "about", "with",
    "from", "this", "that", "there", "policy", "trendly",
}

# Maps a query word to additional terms that appear in the policy document.
# Deliberately small and hand-tuned — the doc is one page, so this is cheaper
# and more predictable than embeddings, and it is auditable at review time.
EXPANSIONS = {
    "refund": ["refund", "refunds", "refunded"],
    "refunds": ["refund", "refunds"],
    "money": ["refund", "refunds"],
    "timeline": ["business days", "time after inspection"],
    "timelines": ["business days"],
    "long": ["business days"],
    "cod": ["cash on delivery"],
    "cash": ["cash on delivery"],
    "card": ["credit", "debit"],
    "upi": ["upi"],
    "return": ["return", "returns", "returned", "returnable"],
    "returns": ["return", "returns"],
    "window": ["30 calendar days", "return window"],
    "exchange": ["exchange", "exchanges", "size exchange"],
    "exchanges": ["exchange", "exchanges"],
    "size": ["size exchange"],
    "colour": ["colour"],
    "color": ["colour"],
    "shipping": ["shipping", "dispatch", "delivery"],
    "delivery": ["delivery", "shipping", "dispatch"],
    "deliver": ["delivery", "shipping"],
    "ship": ["shipping", "dispatch"],
    "express": ["express shipping"],
    "free": ["free standard shipping"],
    "fee": ["shipping fee", "flat"],
    "fees": ["shipping fee", "flat"],
    "charge": ["shipping charges", "fee"],
    "cost": ["fee", "charges"],
    "delay": ["delayed", "delayed orders"],
    "delayed": ["delayed", "store credit"],
    "late": ["delayed"],
    "credit": ["store credit"],
    "lost": ["lost", "lost-parcel", "lost parcel"],
    "missing": ["lost", "no tracking movement"],
    "damaged": ["damaged", "defective", "incorrect"],
    "damage": ["damaged", "defective"],
    "broken": ["damaged", "defective"],
    "faulty": ["defective", "damaged"],
    "defective": ["defective", "damaged"],
    "wrong": ["incorrect", "wrong item"],
    "pickup": ["pickup", "reverse pickup"],
    "collect": ["pickup"],
    "courier": ["carrier", "courier"],
    "address": ["address", "delivery addresses"],
    "cancel": ["cancelled", "cancellation"],
    "cancelled": ["cancelled", "cancellation"],
    "jewellery": ["jewellery"],
    "jewelry": ["jewellery"],
    "innerwear": ["innerwear"],
    "socks": ["socks"],
    "shoes": ["footwear", "shoe box"],
    "footwear": ["footwear", "shoe box"],
    "sneakers": ["footwear"],
    "final": ["final sale"],
    "sale": ["final sale"],
    "hours": ["support hours"],
    "contact": ["human support agent", "support hours"],
    "human": ["human support agent"],
    "agent": ["human support agent"],
    "partial": ["partial shipments", "partial refunds"],
    "backorder": ["backordered"],
    "backordered": ["backordered"],
    "tags": ["original tags"],
    "condition": ["unworn", "unwashed", "original tags"],
    "photos": ["photographs"],
    "photographs": ["photographs"],
}

_SECTIONS_CACHE: dict[str, str] | None = None

# Below this, we would be guessing. The prompt turns NOT FOUND into
# "say you're not certain and offer a human" — that is the desired behaviour
# for genuinely uncovered topics, and only for those.
MIN_SCORE = 3


def load_policy_sections() -> dict[str, str]:
    """Parse the markdown into {section title: full section text}."""
    text = POLICY_PATH.read_text(encoding="utf-8")
    parts = re.split(r"\n(?=## )", text)

    sections: dict[str, str] = {}
    for part in parts:
        part = part.strip()
        if not part.startswith("## "):
            continue  # intro block before the first heading
        title = part.splitlines()[0].replace("## ", "").strip()
        sections[title] = part
    return sections


def _terms(topic: str) -> list[str]:
    """Split a topic into scoring terms, expanded with policy vocabulary."""
    words = [w for w in re.findall(r"[a-z]+", topic.lower()) if w not in STOPWORDS]
    terms: list[str] = []
    for w in words:
        if len(w) > 2:
            terms.append(w)
        terms.extend(EXPANSIONS.get(w, []))
    # Preserve order, drop duplicates.
    seen, out = set(), []
    for t in terms:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def _score(term: str, title: str, body: str) -> int:
    s = 0
    if term in title.lower():
        s += 5
    hits = body.lower().count(term)
    if hits:
        s += min(hits, 3)
    return s


def search_policy_section(topic: str, max_sections: int = 2) -> str:
    """
    Return the best-matching policy section(s) verbatim, or a NOT FOUND
    marker the agent is instructed to treat as "say you don't know and
    offer a human agent".
    """
    global _SECTIONS_CACHE
    if _SECTIONS_CACHE is None:
        _SECTIONS_CACHE = load_policy_sections()

    terms = _terms(topic)
    if not terms:
        terms = [topic.lower().strip()]

    scored = []
    for title, body in _SECTIONS_CACHE.items():
        total = sum(_score(t, title, body) for t in terms)
        if total > 0:
            scored.append((total, title, body))

    scored.sort(key=lambda x: x[0], reverse=True)

    if not scored or scored[0][0] < MIN_SCORE:
        return (
            "NOT FOUND: No policy section covers this topic. Do not invent an "
            "answer — tell the customer this isn't something you can confirm "
            "and offer to hand them to a human agent."
        )

    best_score = scored[0][0]
    chosen = [b for s, _, b in scored[:max_sections] if s >= best_score * 0.4]
    return "\n\n".join(chosen)
