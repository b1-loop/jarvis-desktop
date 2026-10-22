"""``STT`` interface and the transcript it produces.

Speech-to-text sits between the audio engine and the agent: it receives the
float32 16 kHz mono utterance the engine captured and returns text plus the
timing and confidence the dashboard and benchmark need. Providers differ
only in where the model runs.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray

from jarvis.utils.errors import JarvisError

Samples = NDArray[np.float32]
"""Mono float32 PCM in [-1, 1], the audio engine's native format."""

ENGINE_SAMPLE_RATE: Final = 16_000
"""Rate the audio engine captures at; every STT model in use expects it too."""


class TranscriptionError(JarvisError):
    """The audio could not be transcribed (bad input or a model failure)."""


@dataclass(frozen=True, slots=True)
class TranscriptSegment:
    """One timed piece of the transcript.

    Attributes:
        start: Seconds from the beginning of the utterance.
        end: Seconds from the beginning of the utterance.
        text: Words spoken in this span.
        confidence: Model confidence in ``[0, 1]`` if the provider has one.
    """

    start: float
    end: float
    text: str
    confidence: float | None = None


@dataclass(frozen=True, slots=True)
class Transcript:
    """The result of transcribing one utterance.

    Attributes:
        text: The full transcript, whitespace-normalised.
        language: ISO 639-1 code the model detected, if it reports one.
        segments: Timed pieces, in order; empty for silence.
        duration: Length of the audio in seconds.
        confidence: Duration-weighted mean of the segment confidences, or
            ``None`` when the provider gives no confidence.
    """

    text: str
    language: str | None
    segments: tuple[TranscriptSegment, ...]
    duration: float
    confidence: float | None

    @property
    def is_empty(self) -> bool:
        """True when nothing intelligible was heard; the agent skips the turn."""
        return not self.text


class STT(ABC):
    """Turns a captured utterance into text."""

    @abstractmethod
    async def transcribe(
        self, samples: Samples, sample_rate: int = ENGINE_SAMPLE_RATE
    ) -> Transcript:
        """Transcribe ``samples``.

        Args:
            samples: Mono float32 PCM in ``[-1, 1]``.
            sample_rate: Rate of ``samples``; providers resample if their
                model needs something else.

        Raises:
            TranscriptionError: If the audio is malformed or the model fails.
        """


# Design notes:
# - Segments carry timings even though the agent only needs text: the
#   latency benchmark (step 14) and the dashboard transcript view use them,
#   and they are free to keep.
# - An empty transcript is a value, not an error. Silence after a wake word
#   is normal and the orchestrator simply returns to IDLE.
