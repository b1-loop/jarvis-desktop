"""Latency measurement helpers for the pipeline, dashboard and benchmark.

Every stage of a turn (STT, LLM first token, tool call, TTS first chunk) is
wrapped in :func:`timed`, which publishes a ``LatencySample`` on the event
bus and logs it. :class:`LatencyTracker` folds those samples into rolling
per-stage statistics that the dashboard shows live and the benchmark script
prints as a table. Time sources are injected through the :class:`Clock`
Protocol so tests are deterministic.
"""

from __future__ import annotations

import math
import time
from collections import deque
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from statistics import fmean
from typing import Final, Protocol

from jarvis.core.events import EventBus, LatencySample
from jarvis.utils.logging import get_logger

log = get_logger(__name__)

DEFAULT_WINDOW: Final = 200
"""Samples per stage kept for percentiles; enough for a session, cheap to sort."""

_MS_PER_SECOND: Final = 1000.0


class Clock(Protocol):
    """Time source injected into anything that measures or reports time."""

    def now(self) -> datetime:
        """Current wall-clock time, timezone-aware, in the local zone."""

    def monotonic(self) -> float:
        """Seconds from an arbitrary origin; only differences are meaningful."""


class SystemClock:
    """The real clock: ``datetime.now`` and ``time.perf_counter``."""

    def now(self) -> datetime:
        """Local, timezone-aware current time."""
        return datetime.now(UTC).astimezone()

    def monotonic(self) -> float:
        """High-resolution monotonic seconds."""
        return time.perf_counter()


class Stopwatch:
    """Measures elapsed milliseconds from construction (or ``restart``)."""

    def __init__(self, clock: Clock | None = None) -> None:
        self._clock = clock or SystemClock()
        self._start = self._clock.monotonic()
        self._stopped_at: float | None = None

    @property
    def elapsed_ms(self) -> float:
        """Milliseconds since start, frozen once ``stop`` has been called."""
        end = self._stopped_at if self._stopped_at is not None else self._clock.monotonic()
        return (end - self._start) * _MS_PER_SECOND

    def stop(self) -> float:
        """Freeze the watch and return the elapsed milliseconds."""
        if self._stopped_at is None:
            self._stopped_at = self._clock.monotonic()
        return self.elapsed_ms

    def restart(self) -> None:
        """Start measuring again from now."""
        self._start = self._clock.monotonic()
        self._stopped_at = None


@dataclass(frozen=True, slots=True)
class StageStats:
    """Rolling statistics for one pipeline stage.

    Attributes:
        stage: Stage name as passed to ``timed``.
        count: Samples recorded since construction (not limited by the window).
        mean_ms: Mean over the retained window.
        p50_ms: Median over the retained window.
        p95_ms: 95th percentile over the retained window (nearest rank).
        last_ms: The most recent sample.
    """

    stage: str
    count: int
    mean_ms: float
    p50_ms: float
    p95_ms: float
    last_ms: float


def percentile(sorted_values: list[float], fraction: float) -> float:
    """Nearest-rank percentile of an ascending list.

    Args:
        sorted_values: Non-empty, ascending.
        fraction: 0.5 for the median, 0.95 for p95.
    """
    rank = math.ceil(fraction * len(sorted_values))
    index = min(max(rank - 1, 0), len(sorted_values) - 1)
    return sorted_values[index]


class LatencyTracker:
    """Keeps a rolling window of samples per stage and summarises them."""

    def __init__(self, *, window: int = DEFAULT_WINDOW) -> None:
        self._window = window
        self._samples: dict[str, deque[float]] = {}
        self._counts: dict[str, int] = {}

    def record(self, stage: str, ms: float) -> None:
        """Add one sample for ``stage``."""
        self._samples.setdefault(stage, deque(maxlen=self._window)).append(ms)
        self._counts[stage] = self._counts.get(stage, 0) + 1

    def stats(self, stage: str) -> StageStats | None:
        """Statistics for ``stage``, or ``None`` if nothing was recorded."""
        samples = self._samples.get(stage)
        if not samples:
            return None
        ordered = sorted(samples)
        return StageStats(
            stage=stage,
            count=self._counts[stage],
            mean_ms=fmean(ordered),
            p50_ms=percentile(ordered, 0.5),
            p95_ms=percentile(ordered, 0.95),
            last_ms=samples[-1],
        )

    def snapshot(self) -> dict[str, StageStats]:
        """Statistics for every stage, keyed by name, in first-seen order."""
        return {stage: stats for stage in self._samples if (stats := self.stats(stage)) is not None}

    def attach(self, bus: EventBus) -> Callable[[], None]:
        """Record every ``LatencySample`` published on ``bus``.

        Returns:
            A function that detaches the tracker again.
        """
        return bus.subscribe_callback(LatencySample, self._on_sample)

    def _on_sample(self, event: LatencySample) -> None:
        self.record(event.stage, event.ms)


@asynccontextmanager
async def timed(
    stage: str,
    *,
    bus: EventBus | None = None,
    tracker: LatencyTracker | None = None,
    clock: Clock | None = None,
) -> AsyncIterator[Stopwatch]:
    """Time the enclosed block and report it as ``stage``.

    The sample is logged, recorded on ``tracker`` and published as a
    ``LatencySample`` on ``bus`` when given. It is reported even if the block
    raises, because a failed stage still cost the user that time.

    Yields:
        The running stopwatch, so the block can read intermediate timings.
    """
    watch = Stopwatch(clock)
    try:
        yield watch
    finally:
        ms = watch.stop()
        log.debug("latency.sample", stage=stage, ms=round(ms, 1))
        if tracker is not None:
            tracker.record(stage, ms)
        if bus is not None:
            bus.publish(LatencySample(stage=stage, ms=ms))


# Design notes:
# - Clock is a Protocol with both wall-clock and monotonic time so a single
#   fake drives timestamps in tool output and durations in tests.
# - Nearest-rank percentiles over a bounded deque: no numpy dependency in a
#   module every stage imports, and 200 floats sort in microseconds.
# - ``count`` is the lifetime total while mean/percentiles use the window:
#   the dashboard wants "how many turns so far" and "how fast lately".
