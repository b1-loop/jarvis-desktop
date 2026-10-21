"""Fakes for the native audio objects, matching the Protocols in jarvis.audio.engine."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from jarvis.audio.engine import NativePlayback, Samples


@dataclass(frozen=True)
class FakeEventType:
    name: str


@dataclass(frozen=True)
class FakeNativeEvent:
    type: FakeEventType
    value: float
    sample_position: int


def native_event(name: str, value: float = 0.0, sample_position: int = 0) -> FakeNativeEvent:
    return FakeNativeEvent(FakeEventType(name), value, sample_position)


@dataclass
class FakeNativeEngine:
    """Scripted stand-in for ``jarvis_audio.AudioEngine``.

    Each ``poll_events`` call returns the next scripted batch, so a test can
    control exactly when an event becomes visible to the facade.
    """

    batches: deque[list[FakeNativeEvent]] = field(default_factory=deque)
    captured: Samples = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    listening_calls: list[bool] = field(default_factory=list)
    barge_in_targets: list[NativePlayback | None] = field(default_factory=list)
    started: int = 0
    stopped: int = 0
    dropped: int = 0
    level: float = 0.0
    _listening: bool = False

    def script(self, *events: FakeNativeEvent) -> None:
        self.batches.append(list(events))

    def start(self) -> None:
        self.started += 1

    def stop(self) -> None:
        self.stopped += 1

    def running(self) -> bool:
        return self.started > self.stopped

    def set_listening(self, listening: bool) -> None:
        self.listening_calls.append(listening)
        self._listening = listening

    def listening(self) -> bool:
        return self._listening

    def take_captured(self) -> Samples:
        captured, self.captured = self.captured, np.zeros(0, dtype=np.float32)
        return captured

    def poll_events(self) -> list[FakeNativeEvent]:
        return self.batches.popleft() if self.batches else []

    def set_barge_in_target(self, playback: NativePlayback | None) -> None:
        self.barge_in_targets.append(playback)

    def input_level(self) -> float:
        return self.level

    def dropped_samples(self) -> int:
        return self.dropped


@dataclass
class FakeNativePlayback:
    """Stand-in for ``jarvis_audio.Playback``; drains on demand."""

    rate: int = 24_000
    chunks: list[Samples] = field(default_factory=list)
    queued: int = 0
    end_marked: bool = False
    stops: int = 0

    def enqueue(self, samples: Samples) -> None:
        self.chunks.append(samples)
        self.queued += int(samples.size)
        self.end_marked = False

    def stop(self) -> None:
        self.chunks.clear()
        self.queued = 0
        self.end_marked = True
        self.stops += 1

    def mark_end_of_stream(self) -> None:
        self.end_marked = True

    def is_playing(self) -> bool:
        return self.queued > 0 or not self.end_marked

    def queued_samples(self) -> int:
        return self.queued

    def stop_count(self) -> int:
        return self.stops

    def sample_rate(self) -> int:
        return self.rate

    def drain(self) -> None:
        """Pretend the device consumed everything."""
        self.queued = 0
