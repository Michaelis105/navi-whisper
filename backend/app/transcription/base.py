"""Transcriber protocol and structured word-timestamp models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class WordTiming:
    """Word-level timestamp and confidence from Whisper."""

    word: str
    start: float
    end: float
    probability: float = 1.0


@dataclass(frozen=True)
class TranscriptionResult:
    """Normalized transcription output for a 16 kHz mono PCM utterance."""

    text: str
    words: list[WordTiming] = field(default_factory=list)
    no_speech_prob: float = 0.0
    avg_logprob: float = 0.0
    language: str = "en"


class Transcriber(Protocol):
    """Interface for speech-to-text engines."""

    @property
    def gpu_active(self) -> bool: ...

    def transcribe_pcm(
        self, pcm_16k: np.ndarray, word_timestamps: bool = False
    ) -> TranscriptionResult: ...
