"""Base protocol for topic detectors."""

from __future__ import annotations

from typing import Protocol

from app.schemas.events import TopicCandidate, TranscriptSegment


class TopicDetector(Protocol):
    """Extracts specialized topics from new transcript segments using sliding context."""

    async def detect(
        self,
        new_segments: list[TranscriptSegment],
        context_window: list[TranscriptSegment],
        excluded_ids: list[str],
    ) -> list[TopicCandidate]: ...
