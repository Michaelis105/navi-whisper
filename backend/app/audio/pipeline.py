"""Async coordinator bridging WASAPI dual capture, adaptive VAD, Whisper, and Controller."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections import deque
from dataclasses import asdict
from dataclasses import dataclass
from typing import Protocol

from app.audio.adaptive_vad import AdaptiveChannelVAD, VADChunkEvent
from app.audio.wasapi_capture import (
    RawAudioChunk,
    WASAPIDualCapture,
    list_wasapi_devices,
)
from app.config import Settings
from app.controller.orchestrator import BackendController
from app.schemas.events import Speaker, TranscriptSegment
from app.transcription.base import Transcriber
from app.transcription.whisper_local import (
    FasterWhisperTranscriber,
    SentenceBoundarySlicer,
    resolve_whisper_device_and_compute,
)

logger = logging.getLogger(__name__)


class CaptureSource(Protocol):
    """Protocol for audio capture backends (real WASAPI or test fakes)."""

    @property
    def is_running(self) -> bool: ...

    mic_device_name: str | None
    loopback_device_name: str | None

    def drain_chunks(self, max_items: int = 64) -> list[RawAudioChunk]: ...

    def set_preferred_devices(
        self, mic_id: str | None = None, loopback_id: str | None = None
    ) -> None: ...

    def start(self) -> None: ...

    def stop(self) -> None: ...


@dataclass(frozen=True)
class _QueuedVADJob:
    event: VADChunkEvent
    seq: int


class AudioPipelineManager:
    """Implements `AudioControlPort` and manages real-time VAD + GPU transcription."""

    def __init__(
        self,
        settings: Settings,
        controller: BackendController,
        capture: CaptureSource | None = None,
        transcriber: Transcriber | None = None,
        vad_you: AdaptiveChannelVAD | None = None,
        vad_others: AdaptiveChannelVAD | None = None,
        poll_interval_s: float = 0.03,
    ) -> None:
        self._settings = settings
        self._controller = controller
        self._capture: CaptureSource = capture or WASAPIDualCapture()
        self._transcriber: Transcriber = transcriber or FasterWhisperTranscriber(settings)
        self._vads: dict[Speaker, AdaptiveChannelVAD] = {
            "you": vad_you or AdaptiveChannelVAD(speaker="you"),
            "others": vad_others or AdaptiveChannelVAD(speaker="others"),
        }
        self._slicer = SentenceBoundarySlicer()
        self._poll_interval_s = poll_interval_s

        self._vad_queue: asyncio.Queue[_QueuedVADJob] = asyncio.Queue(maxsize=128)
        self._seq_counter = 0
        self._latest_interim_seq: dict[str, int] = {}
        self._final_queued_ids: set[str] = set()
        self._completed_final_ids: deque[str] = deque(maxlen=128)

        self._ingest_task: asyncio.Task[None] | None = None
        self._stt_task: asyncio.Task[None] | None = None
        self._running = False

    async def start(self) -> None:
        """Start background audio ingestion and sequential GPU transcription loops."""
        if self._running:
            return
        self._running = True

        # Detect GPU & default device labels upfront without opening hardware streams
        _, _, gpu_ready = resolve_whisper_device_and_compute(
            self._settings.whisper_device, self._settings.whisper_compute_type
        )
        self._controller.gpu_active = gpu_ready or self._transcriber.gpu_active
        self._populate_default_device_names()

        self._ingest_task = asyncio.create_task(
            self._ingest_loop(), name="audio-vad-ingest"
        )
        self._stt_task = asyncio.create_task(
            self._transcription_worker(), name="whisper-stt-worker"
        )

    async def stop(self) -> None:
        """Stop hardware capture and background workers cleanly."""
        self._running = False
        with contextlib.suppress(Exception):
            self._capture.stop()

        for task in (self._ingest_task, self._stt_task):
            if task is not None and not task.done():
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
        self._ingest_task = None
        self._stt_task = None

    def _populate_default_device_names(self) -> None:
        if self._controller.mic_device or self._controller.loopback_device:
            return
        try:
            devs = list_wasapi_devices()
            default_mic = next((d for d in devs["mics"] if d.is_default), None)
            default_lb = next((d for d in devs["loopbacks"] if d.is_default), None)
            if default_mic:
                self._controller.mic_device = default_mic.name
            if default_lb:
                self._controller.loopback_device = default_lb.name
        except Exception:
            logger.debug("Could not enumerate default WASAPI devices on startup", exc_info=True)

    # --- AudioControlPort Implementation ---

    async def set_listening(self, listening: bool) -> None:
        if listening:
            if isinstance(self._transcriber, FasterWhisperTranscriber):
                await asyncio.to_thread(self._transcriber.ensure_loaded)
            self._controller.gpu_active = self._transcriber.gpu_active
            for vad in self._vads.values():
                vad.reset()
            self._slicer.clear_all()
            await asyncio.to_thread(self._capture.start)
            self._controller.mic_device = self._capture.mic_device_name
            self._controller.loopback_device = self._capture.loopback_device_name
        else:
            await asyncio.to_thread(self._capture.stop)
            # Drain any last captured chunks and flush open speech so trailing words aren't lost
            self._drain_capture_into_vad()
            for vad in self._vads.values():
                if vad.in_speech:
                    flushed = vad._flush_final()
                    if flushed is not None:
                        self._enqueue_vad_event(flushed)

    async def select_devices(
        self, mic_id: str | None = None, loopback_id: str | None = None
    ) -> None:
        self._capture.set_preferred_devices(mic_id=mic_id, loopback_id=loopback_id)
        if self._capture.is_running:
            await asyncio.to_thread(self._capture.stop)
            await asyncio.to_thread(self._capture.start)
            self._controller.mic_device = self._capture.mic_device_name
            self._controller.loopback_device = self._capture.loopback_device_name
            await self._controller.broadcast(self._controller.status_event())

    def get_available_devices(self) -> dict[str, list[dict[str, object]]]:
        devs = list_wasapi_devices()
        return {
            "mics": [asdict(d) for d in devs["mics"]],
            "loopbacks": [asdict(d) for d in devs["loopbacks"]],
        }

    # --- Internal Loops ---

    def _drain_capture_into_vad(self) -> int:
        chunks = self._capture.drain_chunks(max_items=96)
        for chunk in chunks:
            vad = self._vads[chunk.speaker]
            for evt in vad.push_audio(chunk.samples_16k, timestamp=chunk.timestamp):
                self._enqueue_vad_event(evt)
        return len(chunks)

    def _enqueue_vad_event(self, evt: VADChunkEvent) -> None:
        self._seq_counter += 1
        seq = self._seq_counter
        if evt.kind == "interim":
            self._latest_interim_seq[evt.segment_id] = seq
        else:
            self._final_queued_ids.add(evt.segment_id)
            self._latest_interim_seq.pop(evt.segment_id, None)

        job = _QueuedVADJob(event=evt, seq=seq)
        if self._vad_queue.full():
            with contextlib.suppress(asyncio.QueueEmpty):
                dropped = self._vad_queue.get_nowait()
                self._vad_queue.task_done()
                logger.warning("VAD queue full; dropped %s segment %s", dropped.event.kind, dropped.event.segment_id)
        self._vad_queue.put_nowait(job)

    async def _ingest_loop(self) -> None:
        while self._running:
            try:
                if self._capture.is_running:
                    self._drain_capture_into_vad()
                    # Check idle wall-clock timeout for both channels (handles WASAPI loopback silence stalls)
                    for vad in self._vads.values():
                        idle_evt = vad.flush_if_idle()
                        if idle_evt is not None:
                            self._enqueue_vad_event(idle_evt)
                await asyncio.sleep(self._poll_interval_s)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Unexpected error in audio ingest loop")
                await asyncio.sleep(0.1)

    async def _transcription_worker(self) -> None:
        while self._running:
            job = await self._vad_queue.get()
            evt = job.event
            try:
                if evt.kind == "interim":
                    await self._handle_interim_job(evt, job.seq)
                else:
                    await self._handle_final_job(evt)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Error transcribing %s segment %s", evt.kind, evt.segment_id)
            finally:
                self._vad_queue.task_done()

    async def _handle_interim_job(self, evt: VADChunkEvent, seq: int) -> None:
        # Skip stale interim pass if a final chunk or newer interim chunk is already queued
        if (
            evt.segment_id in self._final_queued_ids
            or evt.segment_id in self._completed_final_ids
            or self._latest_interim_seq.get(evt.segment_id, -1) > seq
        ):
            return

        result = await asyncio.to_thread(
            self._transcriber.transcribe_pcm, evt.pcm_16k, True
        )
        self._controller.gpu_active = self._transcriber.gpu_active
        if not result.text:
            return

        # Re-check staleness after GPU inference completes
        if (
            evt.segment_id in self._final_queued_ids
            or evt.segment_id in self._completed_final_ids
        ):
            return

        vad = self._vads[evt.speaker]
        decision = self._slicer.evaluate(
            segment_id=evt.segment_id,
            words=result.words,
            total_samples=int(evt.pcm_16k.size),
        )
        if decision is not None:
            new_seg_id = vad.commit_prefix(evt.segment_id, decision.cut_samples)
            if new_seg_id is not None:
                self._completed_final_ids.append(evt.segment_id)
                self._latest_interim_seq.pop(evt.segment_id, None)
                await self._controller.ingest_transcript(
                    TranscriptSegment(
                        segment_id=evt.segment_id,
                        speaker=evt.speaker,
                        text=decision.committed_text,
                        timestamp=evt.timestamp,
                        is_final=True,
                    )
                )
                if decision.remaining_preview_text:
                    tail_ts = evt.timestamp + (decision.cut_samples / 16000.0)
                    await self._controller.ingest_transcript(
                        TranscriptSegment(
                            segment_id=new_seg_id,
                            speaker=evt.speaker,
                            text=decision.remaining_preview_text,
                            timestamp=tail_ts,
                            is_final=False,
                        )
                    )
                return

        if vad.current_segment_id == evt.segment_id:
            await self._controller.ingest_transcript(
                TranscriptSegment(
                    segment_id=evt.segment_id,
                    speaker=evt.speaker,
                    text=result.text,
                    timestamp=evt.timestamp,
                    is_final=False,
                )
            )

    async def _handle_final_job(self, evt: VADChunkEvent) -> None:
        self._final_queued_ids.discard(evt.segment_id)
        self._latest_interim_seq.pop(evt.segment_id, None)
        self._slicer.clear_segment(evt.segment_id)

        if evt.segment_id in self._completed_final_ids:
            return

        result = await asyncio.to_thread(
            self._transcriber.transcribe_pcm, evt.pcm_16k, False
        )
        self._controller.gpu_active = self._transcriber.gpu_active
        self._completed_final_ids.append(evt.segment_id)
        if not result.text:
            return

        await self._controller.ingest_transcript(
            TranscriptSegment(
                segment_id=evt.segment_id,
                speaker=evt.speaker,
                text=result.text,
                timestamp=evt.timestamp,
                is_final=True,
            )
        )
