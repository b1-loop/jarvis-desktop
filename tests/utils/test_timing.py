import asyncio

import pytest

from jarvis.core.events import EventBus, LatencySample
from jarvis.utils.timing import (
    LatencyTracker,
    Stopwatch,
    SystemClock,
    percentile,
    timed,
)
from tests.fakes import FakeClock


def test_stopwatch_measures_with_the_injected_clock() -> None:
    clock = FakeClock()
    watch = Stopwatch(clock)

    clock.advance(0.25)
    assert watch.elapsed_ms == pytest.approx(250.0)

    assert watch.stop() == pytest.approx(250.0)
    clock.advance(1.0)
    assert watch.elapsed_ms == pytest.approx(250.0)
    assert watch.stop() == pytest.approx(250.0)


def test_stopwatch_restart_resets_the_origin() -> None:
    clock = FakeClock()
    watch = Stopwatch(clock)
    clock.advance(1.0)
    watch.stop()

    watch.restart()
    clock.advance(0.1)

    assert watch.elapsed_ms == pytest.approx(100.0)


def test_system_clock_is_timezone_aware_and_monotonic() -> None:
    clock = SystemClock()

    assert clock.now().tzinfo is not None
    assert clock.monotonic() <= clock.monotonic()


@pytest.mark.parametrize(
    ("fraction", "expected"),
    [(0.5, 3.0), (0.95, 5.0), (0.0, 1.0), (1.0, 5.0)],
)
def test_percentile_uses_nearest_rank(fraction: float, expected: float) -> None:
    assert percentile([1.0, 2.0, 3.0, 4.0, 5.0], fraction) == expected


def test_tracker_reports_count_mean_and_percentiles() -> None:
    tracker = LatencyTracker()
    for ms in (10.0, 20.0, 30.0, 40.0, 100.0):
        tracker.record("stt", ms)

    stats = tracker.stats("stt")

    assert stats is not None
    assert stats.count == 5
    assert stats.mean_ms == pytest.approx(40.0)
    assert stats.p50_ms == 30.0
    assert stats.p95_ms == 100.0
    assert stats.last_ms == 100.0


def test_tracker_window_bounds_percentiles_but_not_count() -> None:
    tracker = LatencyTracker(window=2)
    for ms in (1000.0, 1.0, 2.0):
        tracker.record("llm", ms)

    stats = tracker.stats("llm")

    assert stats is not None
    assert stats.count == 3
    assert stats.mean_ms == pytest.approx(1.5)


def test_tracker_snapshot_and_unknown_stage() -> None:
    tracker = LatencyTracker()
    tracker.record("tts", 5.0)
    tracker.record("stt", 7.0)

    assert tracker.stats("missing") is None
    assert list(tracker.snapshot()) == ["tts", "stt"]


async def test_timed_publishes_records_and_yields_the_watch() -> None:
    clock = FakeClock()
    bus = EventBus()
    tracker = LatencyTracker()
    subscription = bus.subscribe(LatencySample)

    async with timed("stt", bus=bus, tracker=tracker, clock=clock) as watch:
        clock.advance(0.05)
        assert watch.elapsed_ms == pytest.approx(50.0)
        clock.advance(0.05)

    sample = await asyncio.wait_for(anext(subscription), 1.0)
    assert (sample.stage, sample.ms) == ("stt", pytest.approx(100.0))
    stats = tracker.stats("stt")
    assert stats is not None and stats.count == 1


async def test_timed_reports_even_when_the_block_raises() -> None:
    tracker = LatencyTracker()

    with pytest.raises(RuntimeError):
        async with timed("llm", tracker=tracker, clock=FakeClock()):
            raise RuntimeError("provider down")

    assert tracker.stats("llm") is not None


async def test_timed_works_without_bus_or_tracker() -> None:
    async with timed("tts") as watch:
        pass

    assert watch.elapsed_ms >= 0


async def test_tracker_attached_to_a_bus_records_published_samples() -> None:
    bus = EventBus()
    tracker = LatencyTracker()
    detach = tracker.attach(bus)

    bus.publish(LatencySample(stage="tts", ms=12.0))
    detach()
    bus.publish(LatencySample(stage="tts", ms=99.0))

    stats = tracker.stats("tts")
    assert stats is not None
    assert stats.count == 1
