"""Fakes for the voice pipeline: native engine with barge-in, playback, STT, TTS."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field

import numpy as np

from jarvis.audio.engine import Samples
from jarvis.stt.base import ENGINE_SAMPLE_RATE, STT, Transcript, TranscriptionError
from jarvis.tts.base import TTS, AudioChunk, SynthesisError
from tests.audio.fakes import FakeNativeEngine, FakeNativeEvent, FakeNativePlayback, native_event

TTS_RATE = 24_000


@dataclass
class BargeInEngine(FakeNativeEngine):
    """Behaves like the native engine on ``SPEECH_STARTED``: stops the barge-in target."""

    def poll_events(self) -> list[FakeNativeEvent]:
        events = super().poll_events()
        target = self.barge_in_targets[-1] if self.barge_in_targets else None
        if target is not None and any(e.type.name == "SPEECH_STARTED" for e in events):
            target.stop()
        return events

    def say_utterance(self, *, samples: int = 1600) -> None:
        """Script one utterance: speech starts, then ends."""
        self.captured = np.full(samples, 0.1, dtype=np.float32)
        self.script(native_event("SPEECH_STARTED"))
        self.script(native_event("SPEECH_ENDED"))


@dataclass
class DevicePlayback(FakeNativePlayback):
    """Playback that renders instantly unless ``hold`` keeps audio queued.

    ``played`` keeps every chunk ever enqueued (``chunks`` is cleared by stop).
    """

    hold: bool = False
    played: list[Samples] = field(default_factory=list)

    def enqueue(self, samples: Samples) -> None:
        self.played.append(samples)
        super().enqueue(samples)
        if not self.hold:
            self.queued = 0

    def release(self) -> None:
        """Let held audio finish."""
        self.hold = False
        self.queued = 0


@dataclass
class ScriptedSTT(STT):
    """Returns the scripted transcripts in order; an exception entry is raised."""

    texts: list[str | Exception] = field(default_factory=list)
    calls: list[int] = field(default_factory=list)

    async def transcribe(
        self, samples: Samples, sample_rate: int = ENGINE_SAMPLE_RATE
    ) -> Transcript:
        self.calls.append(int(samples.size))
        item = self.texts.pop(0) if self.texts else ""
        if isinstance(item, Exception):
            raise item
        return Transcript(text=item, language="en", segments=(), duration=1.0, confidence=0.9)


def stt_error() -> TranscriptionError:
    return TranscriptionError("model failed")


class ScriptedTTS(TTS):
    """Yields ``chunks`` chunks per sentence with ``delay`` between them; logs what it did.

    Sentences listed in ``failing`` raise ``SynthesisError`` before any audio.
    """

    def __init__(
        self,
        *,
        chunks: int = 2,
        delay: float = 0.0,
        failing: frozenset[str] = frozenset(),
        log: list[str] | None = None,
    ) -> None:
        self._chunks = chunks
        self._delay = delay
        self._failing = failing
        self.log: list[str] = log if log is not None else []
        self.cancelled: list[str] = []

    @property
    def voice(self) -> str:
        return "fake-voice"

    @property
    def sample_rate(self) -> int:
        return TTS_RATE

    async def synthesize(self, text: str) -> AsyncIterator[AudioChunk]:
        self.log.append(f"synth:{text}")
        if text in self._failing:
            raise SynthesisError("service unavailable")
        try:
            for index in range(self._chunks):
                await asyncio.sleep(self._delay)
                yield AudioChunk(np.full(240, 0.01 * (index + 1), dtype=np.float32), TTS_RATE)
        except asyncio.CancelledError:
            self.cancelled.append(text)
            raise
        self.log.append(f"synth-done:{text}")


async def eventually(condition: Callable[[], object], *, limit_s: float = 2.0) -> None:
    """Wait until ``condition()`` is truthy; the fakes expose no events to await."""
    async with asyncio.timeout(limit_s):
        while True:
            if condition():
                return
            await asyncio.sleep(0.002)
