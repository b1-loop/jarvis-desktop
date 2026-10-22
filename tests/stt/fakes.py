"""Scripted stand-ins for the faster-whisper model, matching the Protocols in jarvis.stt."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field

from jarvis.stt.base import Samples


@dataclass(frozen=True)
class FakeSegment:
    start: float
    end: float
    text: str
    avg_logprob: float = -0.1


@dataclass(frozen=True)
class FakeInfo:
    language: str = "en"
    duration: float = 1.0


def lazy(segments: list[FakeSegment], log: list[str]) -> Iterator[FakeSegment]:
    """A generator that records when it is actually consumed, like faster-whisper's."""
    for segment in segments:
        log.append("decoded")
        yield segment


@dataclass
class FakeWhisperModel:
    """Returns scripted segments and records how it was called."""

    segments: list[FakeSegment] = field(default_factory=list)
    info: FakeInfo = field(default_factory=FakeInfo)
    calls: list[dict[str, object]] = field(default_factory=list)
    log: list[str] = field(default_factory=list)
    failure: Exception | None = None

    def transcribe(
        self,
        audio: Samples,
        *,
        language: str | None,
        beam_size: int,
        vad_filter: bool,
        hotwords: str | None = None,
    ) -> tuple[Iterable[FakeSegment], FakeInfo]:
        if self.failure is not None:
            raise self.failure
        self.calls.append(
            {
                "samples": int(audio.size),
                "language": language,
                "beam_size": beam_size,
                "vad_filter": vad_filter,
                "hotwords": hotwords,
            }
        )
        return lazy(self.segments, self.log), self.info
