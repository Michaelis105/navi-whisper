"""Base protocol for LLM summary providers."""

from __future__ import annotations

from typing import Protocol

from app.schemas.events import TopicCandidate, TranscriptSegment


class LLMProvider(Protocol):
    """Generates concise 1-3 bullet summaries for a detected topic."""

    async def summarize(
        self,
        candidate: TopicCandidate,
        context_window: list[TranscriptSegment],
    ) -> list[str]: ...
