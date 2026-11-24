"""Configurable session persistence sinks (ephemeral memory vs. disk export)."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

from app.schemas.events import TopicCard, TranscriptSegment


class SessionSink(Protocol):
    """Interface for recording finalized transcripts and topic cards."""

    async def on_transcript(self, segment: TranscriptSegment) -> None: ...

    async def on_topic_card(self, card: TopicCard) -> None: ...

    async def close(self) -> None: ...


class MemoryOnlySink:
    """Default ephemeral sink: stores nothing on disk."""

    async def on_transcript(self, segment: TranscriptSegment) -> None:
        return None

    async def on_topic_card(self, card: TopicCard) -> None:
        return None

    async def close(self) -> None:
        return None


class JsonlMarkdownSink:
    """Appends finalized transcripts (JSONL) and topic summaries (Markdown) to disk."""

    def __init__(self, base_dir: Path, session_id: str | None = None) -> None:
        stamp = session_id or datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        self._session_path = base_dir / stamp
        self._lock = asyncio.Lock()
        self._initialized = False

    @property
    def session_path(self) -> Path:
        return self._session_path

    def _ensure_dir(self) -> None:
        if not self._initialized:
            self._session_path.mkdir(parents=True, exist_ok=True)
            self._initialized = True

    async def on_transcript(self, segment: TranscriptSegment) -> None:
        line = json.dumps(segment.model_dump(), ensure_ascii=False) + "\n"
        async with self._lock:
            self._ensure_dir()
            await asyncio.to_thread(self._append_text, self._session_path / "transcript.jsonl", line)

    async def on_topic_card(self, card: TopicCard) -> None:
        if card.status != "complete" or not card.bullets:
            return
        bullets_md = "\n".join(f"- {b}" for b in card.bullets)
        block = f"## {card.display_name} ({card.category})\n{bullets_md}\n\n"
        async with self._lock:
            self._ensure_dir()
            await asyncio.to_thread(self._append_text, self._session_path / "summary.md", block)

    @staticmethod
    def _append_text(path: Path, text: str) -> None:
        with path.open("a", encoding="utf-8") as f:
            f.write(text)

    async def close(self) -> None:
        return None
