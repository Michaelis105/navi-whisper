"""ContextRetriever protocol and V1 NoOpRetriever stub."""

from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel

from app.schemas.events import TranscriptSegment


class RetrievedChunk(BaseModel):
    """A single document chunk returned by a RAG retriever."""

    source: str
    title: str
    content: str
    score: float = 1.0


class ContextRetriever(Protocol):
    """Interface for retrieving external knowledge chunks for a topic."""

    async def retrieve(
        self, query: str, context_window: list[TranscriptSegment]
    ) -> list[RetrievedChunk]: ...


class NoOpRetriever:
    """V1 stub returning no external chunks."""

    async def retrieve(
        self, query: str, context_window: list[TranscriptSegment]
    ) -> list[RetrievedChunk]:
        return []
