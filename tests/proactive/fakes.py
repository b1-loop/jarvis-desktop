"""Fakes and stubs for the Protocols in ``jarvis.proactive``.

``FakeClock`` is the heart of the suite: scheduler sleeps become futures
that ``advance`` resolves in time order, so a week of daily jobs runs in
milliseconds and no test ever waits for real.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

SETTLE_ROUNDS = 25


class FakeClock:
    """Manual clock: ``sleep`` parks the caller until ``advance`` reaches its wake time.

    Time is kept in UTC internally and reported in ``start``'s zone, so an
    ``advance`` across a DST change is physically correct (25 wall-clock
    hours really are 25 hours).
    """

    def __init__(self, start: datetime) -> None:
        if start.tzinfo is None:
            raise ValueError("FakeClock needs an aware start time")
        self._tz = start.tzinfo
        self._now = start.astimezone(UTC)
        self._sleepers: list[tuple[datetime, asyncio.Future[None]]] = []

    def now(self) -> datetime:
        return self._now.astimezone(self._tz)

    @property
    def sleeping(self) -> int:
        """How many sleepers are parked right now."""
        return len(self._sleepers)

    async def sleep(self, seconds: float) -> None:
        wake_at = self._now + timedelta(seconds=seconds)
        future: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        entry = (wake_at, future)
        self._sleepers.append(entry)
        try:
            await future
        finally:
            if entry in self._sleepers:
                self._sleepers.remove(entry)

    async def settle(self) -> None:
        """Yield to the loop enough times for woken tasks to re-arm."""
        for _ in range(SETTLE_ROUNDS):
            await asyncio.sleep(0)

    def jump(self, moment: datetime) -> None:
        """Move the clock without waking anyone (simulates a suspended laptop)."""
        self._now = moment.astimezone(UTC)

    async def advance(self, delta: timedelta) -> None:
        """Move forward by ``delta``, waking sleepers in order and letting them run."""
        target = self._now + delta
        # Let tasks woken by other means (a new job, a cancel) re-arm first.
        await self.settle()
        while True:
            due = [entry for entry in self._sleepers if entry[0] <= target and not entry[1].done()]
            if not due:
                break
            wake_at, future = min(due, key=lambda entry: entry[0])
            self._now = max(self._now, wake_at)
            future.set_result(None)
            await self.settle()
        self._now = target
        await self.settle()
