"""Deterministic <1ms pre-filter to drop filler, backchannels, and call logistics."""

from __future__ import annotations

import re

from app.schemas.events import TranscriptSegment

_FILLER_PHRASES = (
    "you know",
    "i mean",
    "sort of",
    "kind of",
)

_FILLER_WORDS = frozenset(
    {
        "um",
        "uh",
        "uhh",
        "umm",
        "ah",
        "er",
        "erm",
        "hmm",
        "hm",
        "like",
        "basically",
        "literally",
        "actually",
        "right",
        "okay",
        "ok",
        "yeah",
        "yep",
        "yup",
        "nope",
        "nah",
        "cool",
        "sure",
        "alright",
        "well",
        "so",
    }
)

_LOGISTICS_PATTERNS = (
    re.compile(r"\bcan you (see|hear) (my screen|me)\b", re.IGNORECASE),
    re.compile(r"\byou(?:'re| are) on mute\b", re.IGNORECASE),
    re.compile(r"\bhold on\b|\bone sec(?:ond)?\b|\bgive me a (?:sec|second|minute)\b", re.IGNORECASE),
    re.compile(r"\bsounds good\b|\bmakes sense\b|\bgot it\b|\bthank you\b|\bthanks\b", re.IGNORECASE),
    re.compile(r"\bhello\b|\bhi everyone\b|\bgood morning\b|\bgood afternoon\b|\bbye\b", re.IGNORECASE),
)

_WORD_RE = re.compile(r"[A-Za-z0-9_'./+-]+")
_SPECIALIZED_TOKEN_RE = re.compile(
    r"^(?:"
    r"[A-Z]{2,}[0-9]*[A-Za-z]*"  # Acronyms: LTV, API, EBITDA, K8s, MRNA
    r"|[a-z]+[A-Z][A-Za-z0-9]*"  # camelCase: fastWhisper, useState
    r"|[A-Za-z0-9]+_[A-Za-z0-9_]+"  # snake_case: borrow_checker
    r"|[A-Za-z0-9]+(?:/[A-Za-z0-9]+)+"  # slash terms: ci/cd, tcp/ip
    r"|[A-Za-z0-9]+-[A-Za-z0-9-]+"  # hyphenated: zero-trust, p99-latency
    r")$"
)


def _strip_fillers(text: str) -> str:
    cleaned = text.strip()
    for phrase in _FILLER_PHRASES:
        cleaned = re.sub(rf"\b{re.escape(phrase)}\b", " ", cleaned, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", cleaned).strip()


def has_specialized_token(tokens: list[str]) -> bool:
    """Return True if any token looks like an acronym or technical identifier."""
    return any(_SPECIALIZED_TOKEN_RE.match(tok) for tok in tokens)


def is_segment_worth_inspecting(segment: TranscriptSegment) -> bool:
    """Return True if the utterance may contain a specialized topic worth inspecting."""
    raw = segment.text.strip()
    if not raw:
        return False

    for pattern in _LOGISTICS_PATTERNS:
        if pattern.search(raw) and len(_WORD_RE.findall(raw)) <= 7:
            return False

    stripped = _strip_fillers(raw)
    if not stripped:
        return False

    raw_tokens = _WORD_RE.findall(stripped)
    meaningful = [t for t in raw_tokens if t.lower() not in _FILLER_WORDS]
    if not meaningful:
        return False

    if has_specialized_token(meaningful):
        return True

    return len(meaningful) >= 4


def filter_segments(segments: list[TranscriptSegment]) -> list[TranscriptSegment]:
    """Filter a batch of segments down to those worth passing to the LLM gatekeeper."""
    return [seg for seg in segments if is_segment_worth_inspecting(seg)]
