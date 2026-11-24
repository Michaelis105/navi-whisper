"""Central async event bus orchestrating audio, topic detection, summaries, and UI."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from pathlib import Path
from typing import Protocol

from fastapi import WebSocket
from pydantic import TypeAdapter, ValidationError

from app.config import Settings
from app.controller.sinks import JsonlMarkdownSink, MemoryOnlySink, SessionSink
from app.controller.state import SessionStore, normalize_topic_id
from app.schemas.events import (
    ClientCommand,
    DevicesSelectCommand,
    ServerEvent,
    SessionClearCommand,
    SessionClearedEvent,
    SessionToggleCommand,
    SettingsUpdateCommand,
    SystemStatusEvent,
    TopicCandidate,
    TopicDetectedEvent,
    TopicDismissCommand,
    TopicErrorEvent,
    TopicManualLookupCommand,
    TopicSummaryEvent,
    TranscriptFinalEvent,
    TranscriptPartialEvent,
    TranscriptSegment,
)

logger = logging.getLogger(__name__)
_COMMAND_ADAPTER: TypeAdapter[ClientCommand] = TypeAdapter(ClientCommand)


class TopicDetectorPort(Protocol):
    """Port for extracting specialized topics from batched transcript segments."""

    async def detect(
        self,
        new_segments: list[TranscriptSegment],
        context_window: list[TranscriptSegment],
        excluded_ids: list[str],
    ) -> list[TopicCandidate]: ...


class SummaryServicePort(Protocol):
    """Port for generating 1-3 bullet summaries for a detected topic."""

    async def summarize(
        self,
        candidate: TopicCandidate,
        context_window: list[TranscriptSegment],
    ) -> list[str]: ...


class AudioControlPort(Protocol):
    """Optional port for controlling live audio capture and device selection."""

    async def set_listening(self, listening: bool) -> None: ...

    async def select_devices(
        self, mic_id: str | None = None, loopback_id: str | None = None
    ) -> None: ...


class BackendController:
    """Single mediator relaying events between pipeline stages and WebSocket clients."""

    def __init__(
        self,
        settings: Settings,
        topic_detector: TopicDetectorPort | None = None,
        summary_service: SummaryServicePort | None = None,
        audio_control: AudioControlPort | None = None,
    ) -> None:
        self.settings = settings
        self.topic_detector = topic_detector
        self.summary_service = summary_service
        self.audio_control = audio_control

        self.store = SessionStore(context_window_seconds=settings.context_window_seconds)
        self.persist_session = settings.persist_session
        self._sink: SessionSink = self._build_sink(self.persist_session)

        self.listening = False
        self.gpu_active = False
        self.mic_device: str | None = None
        self.loopback_device: str | None = None

        self._clients: set[WebSocket] = set()
        self._transcript_queue: asyncio.Queue[TranscriptSegment] = asyncio.Queue(maxsize=256)
        self._pending_detection: list[TranscriptSegment] = []
        self._debounce_task: asyncio.Task[None] | None = None
        self._worker_task: asyncio.Task[None] | None = None
        self._summary_tasks: set[asyncio.Task[None]] = set()
        self._summary_semaphore = asyncio.Semaphore(settings.max_concurrent_summaries)
        self._running = False

    def _build_sink(self, persist: bool) -> SessionSink:
        if persist:
            return JsonlMarkdownSink(Path(self.settings.session_dir))
        return MemoryOnlySink()

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._worker_task = asyncio.create_task(self._transcript_worker(), name="transcript-worker")

    async def stop(self) -> None:
        self._running = False
        if self._debounce_task and not self._debounce_task.done():
            self._debounce_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._debounce_task
        if self._worker_task and not self._worker_task.done():
            self._worker_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._worker_task
        for task in list(self._summary_tasks):
            task.cancel()
        if self._summary_tasks:
            await asyncio.gather(*self._summary_tasks, return_exceptions=True)
        await self._sink.close()

    # --- WebSocket Connection & Broadcast ---

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self._clients.add(websocket)
        await self._send_to(websocket, self.status_event())

    def disconnect(self, websocket: WebSocket) -> None:
        self._clients.discard(websocket)

    def status_event(self) -> SystemStatusEvent:
        return SystemStatusEvent(
            listening=self.listening,
            mic_device=self.mic_device,
            loopback_device=self.loopback_device,
            gpu_active=self.gpu_active,
            persist_session=self.persist_session,
        )

    async def _send_to(self, websocket: WebSocket, event: ServerEvent) -> None:
        await websocket.send_text(event.model_dump_json())

    async def broadcast(self, event: ServerEvent) -> None:
        if not self._clients:
            return
        payload = event.model_dump_json()
        stale: list[WebSocket] = []
        for ws in list(self._clients):
            try:
                await ws.send_text(payload)
            except Exception:
                stale.append(ws)
        for ws in stale:
            self._clients.discard(ws)

    # --- Transcript Ingestion ---

    async def ingest_transcript(self, segment: TranscriptSegment) -> None:
        if not segment.text.strip():
            return
        if not segment.is_final:
            await self.broadcast(
                TranscriptPartialEvent(
                    segment_id=segment.segment_id,
                    speaker=segment.speaker,
                    text=segment.text,
                    timestamp=segment.timestamp,
                )
            )
            return

        if self._transcript_queue.full():
            with contextlib.suppress(asyncio.QueueEmpty):
                self._transcript_queue.get_nowait()
                self._transcript_queue.task_done()
        await self._transcript_queue.put(segment)

    async def _transcript_worker(self) -> None:
        while self._running:
            segment = await self._transcript_queue.get()
            try:
                self.store.add_final_segment(segment)
                await self._sink.on_transcript(segment)
                await self.broadcast(
                    TranscriptFinalEvent(
                        segment_id=segment.segment_id,
                        speaker=segment.speaker,
                        text=segment.text,
                        timestamp=segment.timestamp,
                    )
                )
                if self.topic_detector is not None:
                    self._schedule_topic_detection(segment)
            except Exception:
                logger.exception("Error processing transcript segment %s", segment.segment_id)
            finally:
                self._transcript_queue.task_done()

    def _schedule_topic_detection(self, segment: TranscriptSegment) -> None:
        self._pending_detection.append(segment)
        if self._debounce_task and not self._debounce_task.done():
            self._debounce_task.cancel()
        self._debounce_task = asyncio.create_task(
            self._debounced_detect(), name="topic-debounce"
        )

    async def _debounced_detect(self) -> None:
        try:
            await asyncio.sleep(self.settings.topic_debounce_seconds)
        except asyncio.CancelledError:
            return

        batch = self._pending_detection[:]
        self._pending_detection.clear()
        if not batch or self.topic_detector is None:
            return

        context = self.store.ring_buffer.snapshot()
        excluded = self.store.topics.list_seen_ids()
        try:
            candidates = await self.topic_detector.detect(batch, context, excluded)
        except Exception:
            logger.exception("Topic detection failed")
            return

        for candidate in candidates:
            await self._handle_candidate(candidate)

    async def _handle_candidate(self, candidate: TopicCandidate, force: bool = False) -> None:
        if not force and self.store.topics.is_seen(candidate.canonical_id, candidate.aliases):
            return

        card = self.store.topics.register_detected(candidate)
        await self.broadcast(
            TopicDetectedEvent(
                topic_id=card.topic_id,
                display_name=card.display_name,
                category=card.category,
                source_segment_id=card.source_segment_id,
            )
        )

        if self.summary_service is None:
            return

        task = asyncio.create_task(
            self._run_summary(candidate, card.topic_id),
            name=f"summary-{card.topic_id}",
        )
        self._summary_tasks.add(task)
        task.add_done_callback(self._summary_tasks.discard)

    async def _run_summary(self, candidate: TopicCandidate, topic_id: str) -> None:
        if self.summary_service is None:
            return
        context = self.store.ring_buffer.snapshot()
        started = time.perf_counter()
        try:
            async with self._summary_semaphore:
                bullets = await self.summary_service.summarize(candidate, context)
            latency_ms = int((time.perf_counter() - started) * 1000)
            card = self.store.topics.set_summary(topic_id, bullets, latency_ms)
            if card is not None:
                await self._sink.on_topic_card(card)
                await self.broadcast(
                    TopicSummaryEvent(
                        topic_id=topic_id,
                        bullets=bullets,
                        latency_ms=latency_ms,
                    )
                )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("Summary generation failed for %s", topic_id)
            self.store.topics.set_error(topic_id, str(exc))
            await self.broadcast(TopicErrorEvent(topic_id=topic_id, message=str(exc)))

    # --- Client Commands ---

    async def handle_raw_command(self, raw_text: str) -> None:
        try:
            cmd = _COMMAND_ADAPTER.validate_json(raw_text)
        except ValidationError as exc:
            logger.warning("Invalid WebSocket command: %s", exc)
            return

        if isinstance(cmd, SessionToggleCommand):
            await self.set_listening(cmd.listening)
        elif isinstance(cmd, SessionClearCommand):
            await self.clear_session()
        elif isinstance(cmd, SettingsUpdateCommand):
            if cmd.persist_session is not None:
                await self.set_persist_session(cmd.persist_session)
        elif isinstance(cmd, TopicManualLookupCommand):
            await self.manual_lookup(cmd.text, cmd.context_segment_id)
        elif isinstance(cmd, TopicDismissCommand):
            self.store.topics.dismiss(cmd.topic_id, allow_redetect=cmd.allow_redetect)
        elif isinstance(cmd, DevicesSelectCommand):
            if self.audio_control is not None:
                await self.audio_control.select_devices(cmd.mic_id, cmd.loopback_id)

    async def set_listening(self, listening: bool) -> None:
        self.listening = listening
        if self.audio_control is not None:
            await self.audio_control.set_listening(listening)
        await self.broadcast(self.status_event())

    async def clear_session(self) -> None:
        self._pending_detection.clear()
        if self._debounce_task and not self._debounce_task.done():
            self._debounce_task.cancel()
        self.store.clear()
        await self.broadcast(SessionClearedEvent())

    async def set_persist_session(self, enabled: bool) -> None:
        if enabled == self.persist_session:
            return
        await self._sink.close()
        self.persist_session = enabled
        self._sink = self._build_sink(enabled)
        await self.broadcast(self.status_event())

    async def manual_lookup(
        self, text: str, context_segment_id: str | None = None
    ) -> None:
        cleaned = text.strip()
        if not cleaned:
            return
        candidate = TopicCandidate(
            canonical_id=normalize_topic_id(cleaned),
            display_name=cleaned,
            category="Manual Lookup",
            search_query=cleaned,
            source_segment_id=context_segment_id,
        )
        await self._handle_candidate(candidate, force=True)
