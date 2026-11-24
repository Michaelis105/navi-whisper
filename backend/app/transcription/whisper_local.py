"""Local CUDA/CPU faster-whisper transcriber, hallucination guard, and LocalAgreement slicer."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

import numpy as np

from app.config import Settings, register_cuda_dll_directories
from app.transcription.base import TranscriptionResult, WordTiming

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000

_HALLUCINATION_EXACT = frozenset(
    {
        "thank you",
        "thank you for watching",
        "thanks for watching",
        "thank you very much",
        "subtitles by the amara org community",
        "subtitles by amara org",
        "amara org",
        "please subscribe",
        "like and subscribe",
        "see you next time",
        "bye",
        "you",
        "the",
        "blank audio",
    }
)

_HALLUCINATION_SUBSTRINGS = (
    "subtitles by",
    "captions by",
    "transcribed by",
    "amara.org",
    "[blank_audio]",
    "(blank audio)",
)

_NON_ALNUM_RE = re.compile(r"[^a-z0-9\s]+")
_SENTENCE_END_RE = re.compile(r"[.?!][\"')\]]*$")
_ABBREV_RE = re.compile(
    r"^(?:mr|mrs|ms|dr|prof|sr|jr|vs|etc|fig|inc|ltd|co|st|ave|jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec|e\.g|i\.e|v\d+)\.$",
    re.IGNORECASE,
)


def is_whisper_hallucination(
    result: TranscriptionResult, pcm_16k: np.ndarray | None = None
) -> bool:
    """Return True if a Whisper transcription matches known silence hallucinations."""
    raw = result.text.strip()
    if not raw:
        return True

    lower = raw.lower()
    if any(sub in lower for sub in _HALLUCINATION_SUBSTRINGS):
        return True

    norm = re.sub(r"\s+", " ", _NON_ALNUM_RE.sub(" ", lower)).strip()
    if not norm:
        return True

    # Very low acoustic energy paired with phantom filler phrases
    rms = float(np.sqrt(np.mean(np.square(pcm_16k)))) if (pcm_16k is not None and pcm_16k.size > 0) else 0.05
    if rms < 0.0018 and (result.no_speech_prob > 0.45 or norm in _HALLUCINATION_EXACT):
        return True

    if result.no_speech_prob > 0.80 and result.avg_logprob < -0.75:
        return True

    if norm in _HALLUCINATION_EXACT and (result.no_speech_prob > 0.55 or rms < 0.004):
        return True

    # Detect degenerate repetition loops (e.g., "yeah yeah yeah yeah yeah yeah")
    tokens = norm.split()
    if len(tokens) >= 8 and len(set(tokens)) <= 2:
        return True

    return False


@dataclass(frozen=True)
class SentenceSliceDecision:
    """Result of evaluating an interim word-timestamp transcript for sentence slicing."""

    committed_text: str
    remaining_preview_text: str
    cut_samples: int


def _is_sentence_boundary_word(word_text: str) -> bool:
    stripped = word_text.strip()
    if not stripped or not _SENTENCE_END_RE.search(stripped):
        return False
    if _ABBREV_RE.match(stripped):
        return False
    # Ignore single initials like "J." or numeric decimals
    core = stripped.rstrip(".?!\"')]")
    if len(core) == 1 and core.isalpha() and core.isupper():
        return False
    return True


def _normalize_prefix(words: list[WordTiming]) -> str:
    joined = " ".join(w.word.strip() for w in words).lower()
    return re.sub(r"\s+", " ", _NON_ALNUM_RE.sub(" ", joined)).strip()


class SentenceBoundarySlicer:
    """Commits completed sentences mid-stream for rapid speakers using LocalAgreement.

    Tracks consecutive interim passes per `segment_id`. When a sentence-ending word
    (`.`, `?`, `!`) either:
      1. Stabilizes across two consecutive interim passes (`LocalAgreement`) with `>= 2`
         trailing words after the boundary, OR
      2. Has `>= 4` words before AND `>= 4` words after the boundary with a clear
         inter-word gap (`>= 100 ms`),
    it computes the exact midpoint sample index inside the inter-word gap so the active
    audio buffer can be sliced without clipping phonemes on either side.
    """

    def __init__(self, sample_rate: int = SAMPLE_RATE) -> None:
        self.sample_rate = sample_rate
        # Maps active segment_id -> normalized candidate prefix from previous interim pass
        self._prev_prefixes: dict[str, set[str]] = {}

    def clear_segment(self, segment_id: str) -> None:
        self._prev_prefixes.pop(segment_id, None)

    def clear_all(self) -> None:
        self._prev_prefixes.clear()

    def evaluate(
        self,
        segment_id: str,
        words: list[WordTiming],
        total_samples: int,
    ) -> SentenceSliceDecision | None:
        if len(words) < 5 or total_samples <= 0:
            return None

        prev_seen = self._prev_prefixes.get(segment_id, set())
        current_prefixes: set[str] = set()
        chosen_idx: int | None = None

        # Scan candidate sentence boundaries that still leave >= 2 trailing words
        for idx in range(2, len(words) - 2):
            w = words[idx]
            if not _is_sentence_boundary_word(w.word):
                continue

            prefix_words = words[: idx + 1]
            trailing_words = words[idx + 1 :]
            norm_prefix = _normalize_prefix(prefix_words)
            if not norm_prefix:
                continue
            current_prefixes.add(norm_prefix)

            agreed_with_prev = norm_prefix in prev_seen and len(trailing_words) >= 2
            gap_s = max(0.0, trailing_words[0].start - w.end)
            strong_single_pass = (
                len(prefix_words) >= 4
                and len(trailing_words) >= 4
                and (gap_s >= 0.10 or total_samples >= 6 * self.sample_rate)
            )

            if agreed_with_prev or strong_single_pass:
                chosen_idx = idx
                break

        self._prev_prefixes[segment_id] = current_prefixes
        if chosen_idx is None:
            return None

        end_word = words[chosen_idx]
        next_word = words[chosen_idx + 1]
        # Place sample cut at the midpoint of the inter-sentence breath gap
        if next_word.start >= end_word.end:
            cut_sec = 0.5 * (end_word.end + next_word.start)
        else:
            cut_sec = end_word.end

        cut_samples = int(round(cut_sec * self.sample_rate))
        # Ensure sensible bounds (at least 0.4s committed and at least 0.2s kept)
        min_commit = int(0.4 * self.sample_rate)
        max_commit = total_samples - int(0.2 * self.sample_rate)
        if cut_samples < min_commit or cut_samples >= max_commit:
            return None

        committed_text = "".join(w.word for w in words[: chosen_idx + 1]).strip()
        remaining_text = "".join(w.word for w in words[chosen_idx + 1 :]).strip()
        if not committed_text:
            return None

        self._prev_prefixes.pop(segment_id, None)
        return SentenceSliceDecision(
            committed_text=committed_text,
            remaining_preview_text=remaining_text,
            cut_samples=cut_samples,
        )


def resolve_whisper_device_and_compute(
    requested_device: str = "auto", requested_compute: str = "auto"
) -> tuple[str, str, bool]:
    """Determine optimal faster-whisper `(device, compute_type, gpu_active)`."""
    register_cuda_dll_directories()
    cuda_available = False
    try:
        import ctranslate2

        cuda_available = ctranslate2.get_cuda_device_count() > 0
    except Exception:
        cuda_available = False

    if requested_device == "auto":
        device = "cuda" if cuda_available else "cpu"
    else:
        device = requested_device.lower()

    gpu_active = device == "cuda"
    if requested_compute == "auto":
        compute_type = "float16" if gpu_active else "int8"
    else:
        compute_type = requested_compute

    return device, compute_type, gpu_active


class FasterWhisperTranscriber:
    """Local GPU/CPU speech-to-text engine backed by `faster-whisper`."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._model = None
        self._device = "cpu"
        self._compute_type = "int8"
        self._gpu_active = False

    @property
    def gpu_active(self) -> bool:
        return self._gpu_active

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def ensure_loaded(self) -> None:
        """Load the Whisper model lazily, falling back cleanly to CPU int8 on CUDA errors."""
        if self._model is not None:
            return

        from faster_whisper import WhisperModel

        device, compute_type, gpu_active = resolve_whisper_device_and_compute(
            self._settings.whisper_device, self._settings.whisper_compute_type
        )
        try:
            logger.info(
                "Loading faster-whisper model=%s device=%s compute_type=%s",
                self._settings.whisper_model,
                device,
                compute_type,
            )
            self._model = WhisperModel(
                self._settings.whisper_model,
                device=device,
                compute_type=compute_type,
            )
            self._device = device
            self._compute_type = compute_type
            self._gpu_active = gpu_active
        except Exception as exc:
            if device == "cuda":
                logger.warning(
                    "CUDA Whisper load failed (%s); falling back to CPU int8.", exc
                )
                self._model = WhisperModel(
                    self._settings.whisper_model,
                    device="cpu",
                    compute_type="int8",
                )
                self._device = "cpu"
                self._compute_type = "int8"
                self._gpu_active = False
            else:
                raise

    def transcribe_pcm(
        self, pcm_16k: np.ndarray, word_timestamps: bool = False
    ) -> TranscriptionResult:
        """Transcribe a 16 kHz mono float32 PCM buffer."""
        if pcm_16k.size == 0:
            return TranscriptionResult(text="")

        self.ensure_loaded()
        assert self._model is not None

        audio = np.ascontiguousarray(pcm_16k, dtype=np.float32)
        try:
            segments_iter, info = self._model.transcribe(
                audio,
                language="en" if self._settings.whisper_model.endswith(".en") else None,
                beam_size=1 if word_timestamps else 2,
                best_of=1,
                temperature=0.0,
                condition_on_previous_text=False,
                vad_filter=False,
                without_timestamps=not word_timestamps,
                word_timestamps=word_timestamps,
            )
            segments = list(segments_iter)
        except Exception as exc:
            if self._device == "cuda":
                logger.warning(
                    "CUDA Whisper inference failed (%s); reloading on CPU int8.", exc
                )
                from faster_whisper import WhisperModel

                self._model = WhisperModel(
                    self._settings.whisper_model, device="cpu", compute_type="int8"
                )
                self._device = "cpu"
                self._compute_type = "int8"
                self._gpu_active = False
                return self.transcribe_pcm(audio, word_timestamps=word_timestamps)
            raise

        if not segments:
            return TranscriptionResult(text="")

        texts: list[str] = []
        words: list[WordTiming] = []
        no_speech_probs: list[float] = []
        logprobs: list[float] = []

        for seg in segments:
            if seg.text:
                texts.append(seg.text)
            no_speech_probs.append(float(getattr(seg, "no_speech_prob", 0.0)))
            logprobs.append(float(getattr(seg, "avg_logprob", 0.0)))
            if word_timestamps and getattr(seg, "words", None):
                for w in seg.words:
                    words.append(
                        WordTiming(
                            word=str(w.word),
                            start=float(w.start),
                            end=float(w.end),
                            probability=float(getattr(w, "probability", 1.0)),
                        )
                    )

        full_text = "".join(texts).strip()
        result = TranscriptionResult(
            text=full_text,
            words=words,
            no_speech_prob=float(np.mean(no_speech_probs)) if no_speech_probs else 0.0,
            avg_logprob=float(np.mean(logprobs)) if logprobs else 0.0,
            language=str(getattr(info, "language", "en") or "en"),
        )
        if is_whisper_hallucination(result, pcm_16k=audio):
            return TranscriptionResult(
                text="",
                words=[],
                no_speech_prob=result.no_speech_prob,
                avg_logprob=result.avg_logprob,
                language=result.language,
            )
        return result
