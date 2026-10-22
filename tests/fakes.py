"""Fakes shared across the agent, tool and utility tests."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta, timezone


@dataclass
class FakeClock:
    """Deterministic ``jarvis.utils.timing.Clock``.

    ``advance`` moves both wall-clock and monotonic time so a test can assert
    exact durations and timestamps.
    """

    wall: datetime = field(
        default_factory=lambda: datetime(
            2026, 10, 21, 15, 42, 7, tzinfo=timezone(timedelta(hours=3))
        )
    )
    ticks: float = 100.0

    def now(self) -> datetime:
        return self.wall

    def monotonic(self) -> float:
        return self.ticks

    def advance(self, seconds: float) -> None:
        self.ticks += seconds
        self.wall += timedelta(seconds=seconds)

    def utc(self) -> datetime:
        return self.wall.astimezone(UTC)
