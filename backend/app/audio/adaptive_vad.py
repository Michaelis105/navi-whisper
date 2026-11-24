"""Per-channel Silero VAD state machine with EWMA cadence adaptation and buffer slicing."""

from __future__ import annotations

import time
import uuid
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np

from app.schemas.events import Speaker

SAMPLE_RATE = 16000
FRAME_SAMPLES = 512  # 32 ms at 16 kHz (exact native window for Silero VAD)
FRAME_DURATION_S = FRAME_SAMPLES / float(SAMPLE_RATE)  # 0.032 s

ChunkKind = Literal["interim", "final"]
SpeechProbFn = Callable[[np.ndarray], float]


@dataclass(frozen=True)
class VADChunkEvent:
    """Speech audio snapshot emitted by AdaptiveChannelVAD for Whisper transcription."""

    kind: ChunkKind
    segment_id: str
    speaker: Speaker
    pcm_16k: np.ndarray
    timestamp: float


def build_silero_prob_fn() -> tuple[SpeechProbFn, Callable[[], None]]:
    """Create an isolated stateful Silero ONNX VAD callable and reset hook for one channel."""
    import silero_vad
    import torch

    model = silero_vad.load_silero_vad(onnx=True)

    def _prob(frame_512: np.ndarray) -> float:
        tensor = torch.from_numpy(np.ascontiguousarray(frame_512, dtype=np.float32))
        return float(model(tensor, SAMPLE_RATE))

    def _reset() -> None:
        if hasattr(model, "reset_states"):
            model.reset_states()

    return _prob, _reset


def _new_segment_id() -> str:
    return f"seg_{uuid.uuid4().hex[:10]}"


class AdaptiveChannelVAD:
    """Per-speaker VAD adapting silence hangover between 320ms and 750ms based on speech cadence.

    Features:
    - 256 ms pre-roll ring buffer so leading consonants are never clipped.
    - EWMA pause-duration & speech-density tracking for rapid vs. deliberate speakers.
    - Rolling `interim` snapshots every ~1.0s once continuous speech exceeds ~2.0s.
    - Race-safe `commit_prefix(segment_id, cut_samples)` allowing word-timestamp sentence
      slicing (`LocalAgreement`) to trim finalized sentences without losing trailing audio.
    - Idle wall-clock flush handling Windows WASAPI Loopback callback stalls during silence.
    """

    def __init__(
        self,
        speaker: Speaker,
        prob_fn: SpeechProbFn | None = None,
        reset_fn: Callable[[], None] | None = None,
        speech_threshold: float = 0.5,
        silence_threshold: float = 0.35,
        min_hangover_s: float = 0.32,
        max_hangover_s: float = 0.75,
        initial_hangover_s: float = 0.52,
        min_speech_s: float = 0.224,
        continuous_interim_start_s: float = 2.0,
        interim_interval_s: float = 1.0,
        max_utterance_s: float = 18.0,
        preroll_frames: int = 8,
    ) -> None:
        self.speaker = speaker
        if prob_fn is None:
            prob_fn, reset_fn = build_silero_prob_fn()
        self._prob_fn = prob_fn
        self._reset_fn = reset_fn or (lambda: None)

        self.speech_threshold = speech_threshold
        self.silence_threshold = silence_threshold
        self.min_hangover_s = min_hangover_s
        self.max_hangover_s = max_hangover_s
        self._ewma_hangover_s = initial_hangover_s

        self.min_speech_frames = max(1, int(round(min_speech_s / FRAME_DURATION_S)))
        self.continuous_interim_start_samples = int(round(continuous_interim_start_s * SAMPLE_RATE))
        self.interim_interval_samples = max(
            FRAME_SAMPLES, int(round(interim_interval_s * SAMPLE_RATE))
        )
        self.max_utterance_samples = int(round(max_utterance_s * SAMPLE_RATE))

        # Pending leftover sub-512 samples
        self._frame_remainder = np.empty(0, dtype=np.float32)
        self._preroll: deque[np.ndarray] = deque(maxlen=preroll_frames)

        # Active utterance state
        self._in_speech = False
        self._segment_id = _new_segment_id()
        self._utterance_start_ts: float = 0.0
        self._speech_chunks: list[np.ndarray] = []
        self._speech_samples_count = 0
        self._voiced_frames = 0
        self._total_frames_in_utterance = 0
        self._trailing_silence_frames = 0
        self._samples_since_last_interim = 0

        # Inter-phrase / inter-utterance pause tracking
        self._inter_utterance_silence_frames = 0
        self._has_prior_utterance = False
        self._last_audio_monotonic = time.monotonic()

    @property
    def current_hangover_s(self) -> float:
        return float(np.clip(self._ewma_hangover_s, self.min_hangover_s, self.max_hangover_s))

    @property
    def in_speech(self) -> bool:
        return self._in_speech

    @property
    def current_segment_id(self) -> str:
        return self._segment_id

    def reset(self) -> None:
        self._frame_remainder = np.empty(0, dtype=np.float32)
        self._preroll.clear()
        self._clear_utterance_state()
        self._inter_utterance_silence_frames = 0
        self._has_prior_utterance = False
        self._reset_fn()

    def _clear_utterance_state(self) -> None:
        self._in_speech = False
        self._segment_id = _new_segment_id()
        self._speech_chunks.clear()
        self._speech_samples_count = 0
        self._voiced_frames = 0
        self._total_frames_in_utterance = 0
        self._trailing_silence_frames = 0
        self._samples_since_last_interim = 0

    def push_audio(
        self,
        samples_16k: np.ndarray,
        timestamp: float | None = None,
        now_monotonic: float | None = None,
    ) -> list[VADChunkEvent]:
        """Feed arbitrary-length 16kHz mono float32 samples and return any triggered VAD events."""
        if samples_16k.size == 0:
            return []

        wall_ts = timestamp if timestamp is not None else time.time()
        self._last_audio_monotonic = (
            now_monotonic if now_monotonic is not None else time.monotonic()
        )

        if self._frame_remainder.size > 0:
            buf = np.concatenate((self._frame_remainder, samples_16k))
        else:
            buf = np.asarray(samples_16k, dtype=np.float32)

        n_frames = len(buf) // FRAME_SAMPLES
        remainder_start = n_frames * FRAME_SAMPLES
        self._frame_remainder = buf[remainder_start:].copy()

        events: list[VADChunkEvent] = []
        for i in range(n_frames):
            frame = buf[i * FRAME_SAMPLES : (i + 1) * FRAME_SAMPLES]
            frame_ts = wall_ts - (n_frames - 1 - i) * FRAME_DURATION_S
            evt = self._process_frame(frame, frame_ts)
            if evt is not None:
                events.append(evt)
        return events

    def _process_frame(self, frame: np.ndarray, frame_ts: float) -> VADChunkEvent | None:
        prob = self._prob_fn(frame)

        if not self._in_speech:
            if prob >= self.speech_threshold:
                # Adapt cadence based on the silence gap since the previous utterance
                if self._has_prior_utterance and self._inter_utterance_silence_frames > 0:
                    gap_s = self._inter_utterance_silence_frames * FRAME_DURATION_S
                    if gap_s <= 1.5:
                        self._update_hangover_from_pause(gap_s)

                self._in_speech = True
                self._segment_id = _new_segment_id()
                self._utterance_start_ts = frame_ts - len(self._preroll) * FRAME_DURATION_S
                self._speech_chunks = [f.copy() for f in self._preroll]
                self._speech_chunks.append(frame.copy())
                self._speech_samples_count = len(self._speech_chunks) * FRAME_SAMPLES
                self._voiced_frames = 1
                self._total_frames_in_utterance = 1
                self._trailing_silence_frames = 0
                self._samples_since_last_interim = self._speech_samples_count
                self._preroll.clear()
                self._inter_utterance_silence_frames = 0
            else:
                self._preroll.append(frame.copy())
                if self._has_prior_utterance:
                    self._inter_utterance_silence_frames += 1
            return None

        # Currently inside active speech
        self._speech_chunks.append(frame.copy())
        self._speech_samples_count += FRAME_SAMPLES
        self._samples_since_last_interim += FRAME_SAMPLES
        self._total_frames_in_utterance += 1

        if prob >= self.silence_threshold:
            # Brief intra-utterance micro-pause just ended -> adapt cadence!
            if 2 <= self._trailing_silence_frames:
                micro_pause_s = self._trailing_silence_frames * FRAME_DURATION_S
                self._update_hangover_from_pause(micro_pause_s)
            self._voiced_frames += 1
            self._trailing_silence_frames = 0
        else:
            self._trailing_silence_frames += 1

        # 1) Check silence hangover completion
        hangover_frames = max(1, int(round(self.current_hangover_s / FRAME_DURATION_S)))
        if self._trailing_silence_frames >= hangover_frames:
            return self._flush_final()

        # 2) Check hard maximum utterance length safety cap
        if self._speech_samples_count >= self.max_utterance_samples:
            return self._flush_final()

        # 3) Check rolling interim snapshot trigger during continuous rapid speech
        if (
            self._speech_samples_count >= self.continuous_interim_start_samples
            and self._samples_since_last_interim >= self.interim_interval_samples
            and self._voiced_frames >= self.min_speech_frames
        ):
            self._samples_since_last_interim = 0
            return VADChunkEvent(
                kind="interim",
                segment_id=self._segment_id,
                speaker=self.speaker,
                pcm_16k=self._snapshot_pcm(),
                timestamp=self._utterance_start_ts,
            )

        return None

    def _update_hangover_from_pause(self, observed_pause_s: float) -> None:
        """Adapt hangover target smoothly toward the speaker's natural breath gap."""
        # Target hangover sits slightly above typical intra-thought breath gaps,
        # clamped strictly within [min_hangover_s, max_hangover_s].
        target = float(
            np.clip(observed_pause_s * 1.35, self.min_hangover_s, self.max_hangover_s)
        )
        alpha = 0.30
        self._ewma_hangover_s = (1.0 - alpha) * self._ewma_hangover_s + alpha * target

    def _snapshot_pcm(self) -> np.ndarray:
        if not self._speech_chunks:
            return np.empty(0, dtype=np.float32)
        if len(self._speech_chunks) == 1:
            return self._speech_chunks[0].copy()
        joined = np.concatenate(self._speech_chunks)
        # Consolidate chunk list so repeated interim snapshots stay O(1)
        self._speech_chunks = [joined]
        return joined.copy()

    def _flush_final(self) -> VADChunkEvent | None:
        voiced_enough = self._voiced_frames >= self.min_speech_frames
        seg_id = self._segment_id
        start_ts = self._utterance_start_ts
        pcm = self._snapshot_pcm() if voiced_enough else np.empty(0, dtype=np.float32)

        # Tighten hangover slightly if speech density was extremely rapid (>88% voiced)
        if voiced_enough and self._total_frames_in_utterance >= 25:
            density = self._voiced_frames / float(self._total_frames_in_utterance)
            if density > 0.88:
                self._ewma_hangover_s = max(
                    self.min_hangover_s, self._ewma_hangover_s * 0.92
                )

        self._clear_utterance_state()
        self._reset_fn()
        if voiced_enough and pcm.size > 0:
            self._has_prior_utterance = True
            self._inter_utterance_silence_frames = 0
            return VADChunkEvent(
                kind="final",
                segment_id=seg_id,
                speaker=self.speaker,
                pcm_16k=pcm,
                timestamp=start_ts,
            )
        return None

    def commit_prefix(self, segment_id: str, cut_samples: int) -> str | None:
        """Trim `cut_samples` of a finalized sentence from the front of the active buffer.

        Returns the newly assigned `segment_id` for the remaining live tail, or `None`
        if `segment_id` is stale (e.g., silence hangover already finalized that utterance).
        """
        if not self._in_speech or self._segment_id != segment_id:
            return None
        if cut_samples <= 0 or cut_samples >= self._speech_samples_count:
            return None

        full = self._snapshot_pcm()
        remaining = full[cut_samples:].copy()
        cut_duration_s = cut_samples / float(SAMPLE_RATE)

        self._speech_chunks = [remaining] if remaining.size > 0 else []
        self._speech_samples_count = int(remaining.size)
        self._utterance_start_ts += cut_duration_s
        self._segment_id = _new_segment_id()
        # Ensure remaining voiced count stays valid so trailing speech can finalize normally
        remaining_frames = max(1, remaining.size // FRAME_SAMPLES)
        self._voiced_frames = min(self._voiced_frames, remaining_frames)
        self._total_frames_in_utterance = remaining_frames
        self._samples_since_last_interim = 0
        return self._segment_id

    def flush_if_idle(self, now_monotonic: float | None = None) -> VADChunkEvent | None:
        """Flush buffered speech if stream callbacks stalled (e.g. WASAPI loopback silence)."""
        if not self._in_speech:
            return None
        now = now_monotonic if now_monotonic is not None else time.monotonic()
        stalled_s = now - self._last_audio_monotonic
        already_silent_s = self._trailing_silence_frames * FRAME_DURATION_S
        if (stalled_s + already_silent_s) >= self.current_hangover_s:
            return self._flush_final()
        return None
