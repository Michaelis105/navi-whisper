"""Deterministic post-generation guardrails for concise 1-3 bullet summaries."""

from __future__ import annotations

import re

_LEADING_GLYPH_RE = re.compile(r"^\s*(?:[-*•]+|\d+[.)])\s*")
_PREAMBLE_RE = re.compile(
    r"^(?:here (?:is|are)|summary|sure|certainly|below is)\b.*?:\s*$",
    re.IGNORECASE,
)


def _truncate_words(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    trimmed = " ".join(words[:max_words]).rstrip(",;:-")
    if not trimmed.endswith((".", "!", "?")):
        trimmed += "..."
    return trimmed


def enforce_summary_guardrails(
    raw_bullets: list[str],
    max_bullets: int = 3,
    max_words: int = 18,
) -> list[str]:
    """Normalize, deduplicate, cap, and word-limit summary bullets."""
    cleaned: list[str] = []
    for item in raw_bullets:
        line = _LEADING_GLYPH_RE.sub("", item.strip())
        line = re.sub(r"\s+", " ", line).strip()
        if not line or _PREAMBLE_RE.match(line):
            continue
        cleaned.append(_truncate_words(line, max_words=max_words))
        if len(cleaned) >= max_bullets:
            break

    if not cleaned:
        return ["No concise summary available."]
    return cleaned
