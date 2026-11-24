"""In-memory session state: sliding transcript buffer and topic deduplication registry."""

from __future__ import annotations

import re
from collections import deque

from app.schemas.events import TopicCandidate, TopicCard, TranscriptSegment

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def normalize_topic_id(raw: str) -> str:
    """Normalize a topic name or alias into a canonical lowercase slug."""
    slug = _SLUG_RE.sub("-", raw.strip().lower()).strip("-")
    return slug or "topic"


class TranscriptRingBuffer:
    """Bounded sliding time-window of finalized transcript segments."""

    def __init__(self, window_seconds: float = 30.0, max_items: int = 24) -> None:
        self._window_seconds = window_seconds
        self._max_items = max_items
        self._items: deque[TranscriptSegment] = deque()

    def append(self, segment: TranscriptSegment) -> None:
        self._items.append(segment)
        self._prune(reference_ts=segment.timestamp)

    def _prune(self, reference_ts: float) -> None:
        cutoff = reference_ts - self._window_seconds
        while self._items and (
            len(self._items) > self._max_items or self._items[0].timestamp < cutoff
        ):
            self._items.popleft()

    def snapshot(self) -> list[TranscriptSegment]:
        return list(self._items)

    def clear(self) -> None:
        self._items.clear()


class TopicRegistry:
    """Tracks detected topics, aliases, and dismissed suppressions for a session."""

    def __init__(self) -> None:
        self._cards: dict[str, TopicCard] = {}
        self._alias_to_id: dict[str, str] = {}
        self._suppressed_ids: set[str] = set()

    def is_seen(self, canonical_id: str, aliases: list[str] | None = None) -> bool:
        keys = [normalize_topic_id(canonical_id)]
        if aliases:
            keys.extend(normalize_topic_id(a) for a in aliases if a.strip())
        return any(
            k in self._cards or k in self._alias_to_id or k in self._suppressed_ids
            for k in keys
        )

    def register_detected(self, candidate: TopicCandidate) -> TopicCard:
        topic_id = normalize_topic_id(candidate.canonical_id)
        card = TopicCard(
            topic_id=topic_id,
            display_name=candidate.display_name,
            category=candidate.category,
            status="researching",
            source_segment_id=candidate.source_segment_id,
        )
        self._cards[topic_id] = card
        self._alias_to_id[topic_id] = topic_id
        self._alias_to_id[normalize_topic_id(candidate.display_name)] = topic_id
        for alias in candidate.aliases:
            norm = normalize_topic_id(alias)
            if norm:
                self._alias_to_id[norm] = topic_id
        self._suppressed_ids.discard(topic_id)
        return card

    def set_summary(
        self, topic_id: str, bullets: list[str], latency_ms: int
    ) -> TopicCard | None:
        card = self._cards.get(topic_id)
        if card is None:
            return None
        card.status = "complete"
        card.bullets = bullets
        card.latency_ms = latency_ms
        card.error_message = None
        return card

    def set_error(self, topic_id: str, message: str) -> TopicCard | None:
        card = self._cards.get(topic_id)
        if card is None:
            return None
        card.status = "error"
        card.error_message = message
        return card

    def dismiss(self, topic_id: str, allow_redetect: bool = False) -> bool:
        norm = normalize_topic_id(topic_id)
        resolved = self._alias_to_id.get(norm, norm)
        existed = resolved in self._cards
        self._cards.pop(resolved, None)

        if allow_redetect:
            self._suppressed_ids.discard(resolved)
            self._alias_to_id = {
                k: v for k, v in self._alias_to_id.items() if v != resolved
            }
        else:
            self._suppressed_ids.add(resolved)
        return existed

    def list_seen_ids(self) -> list[str]:
        seen = set(self._cards.keys()) | self._suppressed_ids
        return sorted(seen)

    def list_cards(self) -> list[TopicCard]:
        return list(self._cards.values())

    def clear(self) -> None:
        self._cards.clear()
        self._alias_to_id.clear()
        self._suppressed_ids.clear()


class SessionStore:
    """Ephemeral in-memory container for the active call session."""

    def __init__(self, context_window_seconds: float = 30.0) -> None:
        self.transcripts: list[TranscriptSegment] = []
        self.ring_buffer = TranscriptRingBuffer(window_seconds=context_window_seconds)
        self.topics = TopicRegistry()

    def add_final_segment(self, segment: TranscriptSegment) -> None:
        self.transcripts.append(segment)
        self.ring_buffer.append(segment)

    def clear(self) -> None:
        self.transcripts.clear()
        self.ring_buffer.clear()
        self.topics.clear()
