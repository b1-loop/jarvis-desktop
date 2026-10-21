"""Test doubles for the memory package."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from jarvis.memory.models import Message

EPOCH = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)


@dataclass
class FakeClock:
    """Deterministic ``now`` that only moves when a test says so."""

    now: datetime = EPOCH

    def __call__(self) -> datetime:
        return self.now

    def advance(self, **delta: float) -> datetime:
        self.now += timedelta(**delta)
        return self.now


@dataclass
class RecordingSummariser:
    """Summariser hook that remembers what it was asked to summarise."""

    summary: str = "we talked"
    calls: list[tuple[Message, ...]] = field(default_factory=list)

    async def __call__(self, messages: Sequence[Message]) -> str:
        self.calls.append(tuple(messages))
        return self.summary
