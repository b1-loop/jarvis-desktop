"""``TTS`` interface with streaming synthesis.

A provider turns one sentence into audio and yields it in chunks as they
become available, so the ``Speaker`` can start playing before the sentence
is fully synthesised. The orchestrator feeds providers sentence by sentence
(see ``jarvis.llm.streaming.SentenceSplitter``), which is what keeps the
time from the first LLM token to the first spoken word short.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from jarvis.utils.errors import JarvisError

Samples = NDArray[np.float32]
"""Mono float32 PCM in [-1, 1], what ``jarvis.audio.engine.Speaker`` plays."""


class SynthesisError(JarvisError):
    """Speech could not be produced (network, service or decoding failure).

    The orchestrator degrades to printing the answer when it sees this.
    """


@dataclass(frozen=True, slots=True)
class AudioChunk:
    """A piece of synthesised speech.

    Attributes:
        samples: Mono float32 PCM.
        sample_rate: Rate of ``samples``; constant for a given provider.
    """

    samples: Samples
    sample_rate: int

    @property
    def duration_seconds(self) -> float:
        """Play time of this chunk."""
        return float(self.samples.size) / self.sample_rate


class TTS(ABC):
    """Speaks text, one sentence at a time, as a stream of PCM chunks."""

    @property
    @abstractmethod
    def voice(self) -> str:
        """Provider-specific voice identifier, for logs and the settings UI."""

    @property
    @abstractmethod
    def sample_rate(self) -> int:
        """Rate of every chunk ``synthesize`` yields; the playback device is opened at it."""

    @abstractmethod
    def synthesize(self, text: str) -> AsyncIterator[AudioChunk]:
        """Synthesise ``text`` and yield audio as it is produced.

        Args:
            text: One sentence or short utterance. Empty text yields nothing.

        Yields:
            Chunks in playback order, all at :attr:`sample_rate`.

        Raises:
            SynthesisError: If the provider or the decoder fails.
        """


# Design notes:
# - sample_rate is a property on the provider, not only on each chunk, so the
#   playback device can be opened once at the right rate before any speech
#   exists; chunks repeat it so a mismatch is detectable.
# - The interface takes a sentence, not a whole answer: sentence-level calls
#   are what let playback overlap with generation, and they keep barge-in
#   cheap because at most one sentence of audio is ever in flight.
