"""Phase 3 tests: resampling, adaptive VAD, hallucination filter, sentence slicing, and WASAPI/GPU smoke."""

from __future__ import annotations

import asyncio
import math
import queue

import numpy as np

from app.audio.adaptive_vad import (
    FRAME_SAMPLES,
    SAMPLE_RATE,
    AdaptiveChannelVAD,
    build_silero_prob_fn,
)
from app.audio.pipeline import AudioPipelineManager
from app.audio.wasapi_capture import (
    RawAudioChunk,
    StatefulResampler,
    list_wasapi_devices,
)
from app.config import Settings
from app.controller.orchestrator import BackendController
from app.schemas.events import TranscriptSegment
from app.transcription.base import TranscriptionResult, WordTiming
from app.transcription.whisper_local import (
    SentenceBoundarySlicer,
    is_whisper_hallucination,
    resolve_whisper_device_and_compute,
)


def test_resampler_48k_and_44k1_continuity_and_smart_single_mic() -> None:
    # 1) 44.1 kHz stereo split across irregular chunk sizes -> smooth 16 kHz sine wave
    sr_in = 44100
    duration_s = 1.0
    t = np.arange(int(sr_in * duration_s), dtype=np.float32) / sr_in
    sine = 0.6 * np.sin(2.0 * math.pi * 440.0 * t).astype(np.float32)
    stereo = np.stack([sine, sine], axis=1).reshape(-1)

    resampler = StatefulResampler(source_rate=sr_in, channels=2, smart_single_mic=False)
    outputs: list[np.ndarray] = []
    idx = 0
    chunk_frames = [317, 1024, 509, 1411, 777]
    step_i = 0
    while idx < sine.size:
        n = min(chunk_frames[step_i % len(chunk_frames)], sine.size - idx)
        outputs.append(resampler.process(stereo[idx * 2 : (idx + n) * 2]))
        idx += n
        step_i += 1

    merged = np.concatenate(outputs)
    assert abs(merged.size - 16000) <= 2
    # Max sample-to-sample jump for 440 Hz @ 16 kHz is ~2*pi*440/16000 * 0.6 ≈ 0.104.
    # Boundary clicks would spike well above 0.25.
    max_jump = float(np.max(np.abs(np.diff(merged[16:]))))
    assert max_jump < 0.14

    # 2) Smart single-channel XLR mic (Left=signal, Right=0.0) preserves full amplitude
    xlr_stereo = np.stack([sine[:4800], np.zeros(4800, dtype=np.float32)], axis=1).reshape(-1)
    mic_resampler = StatefulResampler(source_rate=48000, channels=2, smart_single_mic=True)
    lb_resampler = StatefulResampler(source_rate=48000, channels=2, smart_single_mic=False)

    mic_out = mic_resampler.process(xlr_stereo)
    lb_out = lb_resampler.process(xlr_stereo)
    assert float(np.max(np.abs(mic_out))) > 0.50
    assert float(np.max(np.abs(lb_out))) < 0.32


def test_adaptive_vad_cadence_and_idle_loopback_flush() -> None:
    def amp_prob(frame: np.ndarray) -> float:
        return 0.9 if float(np.mean(np.abs(frame))) > 0.1 else 0.05

    vad = AdaptiveChannelVAD(
        speaker="others",
        prob_fn=amp_prob,
        initial_hangover_s=0.55,
        min_hangover_s=0.32,
        max_hangover_s=0.75,
    )

    voiced_frame = np.full(FRAME_SAMPLES, 0.4, dtype=np.float32)
    silent_frame = np.zeros(FRAME_SAMPLES, dtype=np.float32)

    # Simulate rapid speaker taking multiple 160ms (5-frame) micro-pauses inside speech
    for _ in range(4):
        vad.push_audio(np.tile(voiced_frame, 12), timestamp=10.0)
        vad.push_audio(np.tile(silent_frame, 5), timestamp=10.5)
    vad.push_audio(np.tile(voiced_frame, 10), timestamp=11.0)

    assert vad.current_hangover_s < 0.45

    # Now simulate WASAPI loopback stalling completely (no push_audio calls during digital silence)
    vad.push_audio(np.tile(voiced_frame, 10), timestamp=12.0, now_monotonic=100.0)
    assert vad.in_speech is True
    assert vad.flush_if_idle(now_monotonic=100.10) is None
    flushed = vad.flush_if_idle(now_monotonic=100.60)
    assert flushed is not None
    assert flushed.kind == "final"
    assert vad.in_speech is False


def test_whisper_hallucination_guard() -> None:
    quiet_pcm = np.full(16000, 0.0005, dtype=np.float32)
    normal_pcm = np.full(16000, 0.08, dtype=np.float32)

    assert is_whisper_hallucination(
        TranscriptionResult(text="Thank you for watching.", no_speech_prob=0.7), quiet_pcm
    )
    assert is_whisper_hallucination(
        TranscriptionResult(text="Subtitles by the Amara.org community"), normal_pcm
    )
    assert is_whisper_hallucination(
        TranscriptionResult(text="yeah yeah yeah yeah yeah yeah yeah yeah"), normal_pcm
    )
    assert not is_whisper_hallucination(
        TranscriptionResult(text="We migrated session state to etcd.", no_speech_prob=0.05),
        normal_pcm,
    )


def test_sentence_boundary_slicer_local_agreement_and_abbreviations() -> None:
    slicer = SentenceBoundarySlicer(sample_rate=16000)

    # Abbreviations like "Dr." must not slice
    abbrev_words = [
        WordTiming("Talk", 0.0, 0.3),
        WordTiming(" to", 0.3, 0.5),
        WordTiming(" Dr.", 0.5, 0.8),
        WordTiming(" Smith", 0.9, 1.2),
        WordTiming(" tomorrow", 1.2, 1.6),
    ]
    assert slicer.evaluate("seg_1", abbrev_words, total_samples=32000) is None

    # Pass 1: sentence boundary with 2 trailing words seeds LocalAgreement prefix
    pass1 = [
        WordTiming("We", 0.1, 0.3),
        WordTiming(" chose", 0.3, 0.7),
        WordTiming(" etcd.", 0.7, 1.2),
        WordTiming(" Latency", 1.36, 1.8),
        WordTiming(" dropped", 1.8, 2.2),
    ]
    assert slicer.evaluate("seg_fast", pass1, total_samples=int(2.3 * 16000)) is None

    # Pass 2: prefix "we chose etcd" stabilizes with trailing speech -> slices at gap midpoint (1.28s)
    pass2 = pass1 + [WordTiming(" sharply", 2.2, 2.7)]
    decision = slicer.evaluate("seg_fast", pass2, total_samples=int(2.9 * 16000))
    assert decision is not None
    assert decision.committed_text == "We chose etcd."
    assert decision.remaining_preview_text == "Latency dropped sharply"
    assert abs(decision.cut_samples - int(round(1.28 * 16000))) <= 2


class _FakeCapture:
    def __init__(self) -> None:
        self._running = False
        self.mic_device_name = "Fake Mic"
        self.loopback_device_name = "Fake Loopback"
        self._q: queue.Queue[RawAudioChunk] = queue.Queue()

    @property
    def is_running(self) -> bool:
        return self._running

    def start(self) -> None:
        self._running = True

    def stop(self) -> None:
        self._running = False

    def set_preferred_devices(
        self, mic_id: str | None = None, loopback_id: str | None = None
    ) -> None:
        pass

    def feed(self, speaker: str, pcm_16k: np.ndarray, ts: float) -> None:
        self._q.put(RawAudioChunk(speaker=speaker, samples_16k=pcm_16k, timestamp=ts))  # type: ignore[arg-type]

    def drain_chunks(self, max_items: int = 64) -> list[RawAudioChunk]:
        out: list[RawAudioChunk] = []
        while len(out) < max_items and not self._q.empty():
            out.append(self._q.get_nowait())
        return out


class _ScriptedRapidTranscriber:
    gpu_active = True

    def __init__(self) -> None:
        self.interim_calls = 0

    def transcribe_pcm(
        self, pcm_16k: np.ndarray, word_timestamps: bool = False
    ) -> TranscriptionResult:
        if word_timestamps:
            self.interim_calls += 1
            base = [
                WordTiming("We", 0.1, 0.3),
                WordTiming(" deployed", 0.3, 0.8),
                WordTiming(" etcd.", 0.8, 1.3),
                WordTiming(" Failover", 1.5, 1.9),
                WordTiming(" works", 1.9, 2.2),
            ]
            if self.interim_calls >= 2:
                base.append(WordTiming(" cleanly", 2.2, 2.7))
            return TranscriptionResult(
                text="".join(w.word for w in base).strip(),
                words=base,
            )
        return TranscriptionResult(text="Failover works cleanly now.")


async def test_audio_pipeline_rapid_speaker_midstream_commit() -> None:
    settings = Settings(topic_debounce_seconds=5.0)
    controller = BackendController(settings)
    received: list[TranscriptSegment] = []

    orig_ingest = controller.ingest_transcript

    async def capture_ingest(seg: TranscriptSegment) -> None:
        received.append(seg)
        await orig_ingest(seg)

    controller.ingest_transcript = capture_ingest  # type: ignore[method-assign]

    def amp_prob(frame: np.ndarray) -> float:
        return 0.9 if float(np.mean(np.abs(frame))) > 0.1 else 0.02

    vad_others = AdaptiveChannelVAD(
        speaker="others",
        prob_fn=amp_prob,
        continuous_interim_start_s=1.8,
        interim_interval_s=0.8,
        initial_hangover_s=0.40,
    )
    fake_cap = _FakeCapture()
    fake_stt = _ScriptedRapidTranscriber()
    pipeline = AudioPipelineManager(
        settings=settings,
        controller=controller,
        capture=fake_cap,
        transcriber=fake_stt,
        vad_others=vad_others,
        poll_interval_s=0.01,
    )
    controller.audio_control = pipeline

    await controller.start()
    await pipeline.start()
    try:
        await controller.set_listening(True)
        voiced_1s = np.full(SAMPLE_RATE, 0.35, dtype=np.float32)

        # Feed 2.1s of continuous rapid speech -> Interim #1
        fake_cap.feed("others", np.tile(voiced_1s, 2), ts=1000.0)
        fake_cap.feed("others", np.full(3200, 0.35, dtype=np.float32), ts=1002.0)
        await asyncio.sleep(0.08)

        # Feed another 1.0s of continuous speech -> Interim #2 triggers LocalAgreement slice!
        fake_cap.feed("others", voiced_1s, ts=1003.0)
        await asyncio.sleep(0.08)

        # Feed silence hangover -> Finalizes the remaining tail sentence!
        fake_cap.feed("others", np.zeros(int(0.6 * SAMPLE_RATE), dtype=np.float32), ts=1004.0)
        await asyncio.sleep(0.10)

        finals = [s.text for s in received if s.is_final]
        partials = [s.text for s in received if not s.is_final]
        assert finals == ["We deployed etcd.", "Failover works cleanly now."]
        assert len(partials) >= 1
    finally:
        await pipeline.stop()
        await controller.stop()


def test_hardware_wasapi_silero_and_cuda_smoke() -> None:
    # 1) CUDA resolution on RTX 5070 Ti
    device, compute_type, gpu_active = resolve_whisper_device_and_compute()
    assert device == "cuda"
    assert compute_type == "float16"
    assert gpu_active is True

    # 2) Silero ONNX VAD single-frame execution
    prob_fn, reset_fn = build_silero_prob_fn()
    p = prob_fn(np.zeros(FRAME_SAMPLES, dtype=np.float32))
    assert 0.0 <= p <= 1.0
    reset_fn()

    # 3) WASAPI device enumeration
    devices = list_wasapi_devices()
    assert "mics" in devices and "loopbacks" in devices
    assert len(devices["loopbacks"]) >= 1
