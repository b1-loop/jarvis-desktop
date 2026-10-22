"""CaptureWatchdog: stall detection, restart with backoff, one notification per outage.

Everything runs on a fake clock and a fake capture, so no test sleeps and
no device is opened.
"""

import asyncio
from collections import deque
from dataclasses import dataclass, field
from itertools import pairwise

import pytest
from structlog.testing import capture_logs

from jarvis.audio.watchdog import (
    MICROPHONE_BACK,
    MICROPHONE_LOST,
    NO_SAMPLES,
    CaptureWatchdog,
    WatchdogConfig,
)

CONFIG = WatchdogConfig(
    stall_seconds=2.5,
    lost_grace_seconds=0.5,
    check_interval_seconds=0.25,
    initial_backoff_seconds=1.0,
    max_backoff_seconds=8.0,
    notify_after_attempts=3,
)
PERIOD = 160  # samples per 10 ms device period at 16 kHz


class DeviceGoneError(RuntimeError):
    """Stands in for the native AudioDeviceError."""


@dataclass
class FakeClock:
    now: float = 100.0

    def __call__(self) -> float:
        return self.now


@dataclass
class FakeCapture:
    """A capture whose sample counter and restarts the test controls."""

    received: int = 0
    restarts: int = 0
    errors: deque[Exception] = field(default_factory=deque)
    heal_on_restart: bool = False
    alive: bool = True

    def samples_received(self) -> int:
        if self.alive:
            self.received += PERIOD
        return self.received

    async def restart_capture(self) -> bool:
        self.restarts += 1
        if self.errors:
            raise self.errors.popleft()
        if self.heal_on_restart:
            self.alive = True
        return True


@dataclass
class Notices:
    texts: list[str] = field(default_factory=list)

    async def __call__(self, text: str) -> None:
        self.texts.append(text)


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def capture() -> FakeCapture:
    return FakeCapture()


@pytest.fixture
def notices() -> Notices:
    return Notices()


@pytest.fixture
def watchdog(capture: FakeCapture, clock: FakeClock, notices: Notices) -> CaptureWatchdog:
    return CaptureWatchdog(
        capture,
        config=CONFIG,
        notify=notices,
        clock=clock,
        device_name=lambda: "Microphone Array (AMD Audio Device)",
        restart_errors=(DeviceGoneError,),
    )


async def advance(watchdog: CaptureWatchdog, clock: FakeClock, seconds: float) -> None:
    """Run checks at the configured interval for ``seconds`` of fake time."""
    steps = round(seconds / CONFIG.check_interval_seconds)
    for _ in range(steps):
        clock.now += CONFIG.check_interval_seconds
        await watchdog.check()


async def stall(watchdog: CaptureWatchdog, capture: FakeCapture, clock: FakeClock) -> None:
    """A healthy baseline, then a capture that stops delivering for the stall window."""
    await watchdog.check()
    capture.alive = False
    await advance(watchdog, clock, CONFIG.stall_seconds)


async def test_silence_that_still_delivers_samples_is_never_a_stall(
    watchdog: CaptureWatchdog, capture: FakeCapture, clock: FakeClock
) -> None:
    await advance(watchdog, clock, 600.0)

    assert capture.restarts == 0
    assert not watchdog.stalled


async def test_a_flat_counter_shorter_than_the_window_is_tolerated(
    watchdog: CaptureWatchdog, capture: FakeCapture, clock: FakeClock
) -> None:
    await watchdog.check()
    capture.alive = False
    await advance(watchdog, clock, CONFIG.stall_seconds - 0.5)

    assert capture.restarts == 0


async def test_a_stall_is_logged_and_restarts_the_capture(
    watchdog: CaptureWatchdog, capture: FakeCapture, clock: FakeClock
) -> None:
    with capture_logs() as logs:
        await stall(watchdog, capture, clock)

    assert capture.restarts == 1
    assert watchdog.stalled
    stalled = next(e for e in logs if e["event"] == "audio.capture.stalled")
    assert stalled["reason"] == NO_SAMPLES
    assert stalled["silent_s"] == pytest.approx(CONFIG.stall_seconds)
    assert stalled["device"] == "Microphone Array (AMD Audio Device)"


async def test_frames_after_a_restart_end_the_outage(
    watchdog: CaptureWatchdog, capture: FakeCapture, clock: FakeClock
) -> None:
    capture.heal_on_restart = True
    with capture_logs() as logs:
        await stall(watchdog, capture, clock)
        await advance(watchdog, clock, CONFIG.check_interval_seconds)

    assert not watchdog.stalled
    recovered = next(e for e in logs if e["event"] == "audio.capture.recovered")
    assert recovered["attempts"] == 1
    assert recovered["reason"] == NO_SAMPLES
    assert recovered["outage_s"] == pytest.approx(CONFIG.stall_seconds + 0.25, abs=0.1)


async def test_failed_restarts_back_off_exponentially_up_to_the_cap(
    watchdog: CaptureWatchdog, capture: FakeCapture, clock: FakeClock
) -> None:
    capture.errors.extend(DeviceGoneError("no device") for _ in range(10))
    await stall(watchdog, capture, clock)
    attempts_at: list[float] = [clock.now]

    for _ in range(120):
        before = capture.restarts
        await advance(watchdog, clock, CONFIG.check_interval_seconds)
        if capture.restarts != before:
            attempts_at.append(clock.now)

    gaps = [round(b - a, 2) for a, b in pairwise(attempts_at)]
    assert gaps[:5] == [1.0, 2.0, 4.0, 8.0, 8.0]


async def test_a_restart_that_delivers_nothing_counts_as_failed(
    watchdog: CaptureWatchdog, capture: FakeCapture, clock: FakeClock
) -> None:
    await stall(watchdog, capture, clock)
    await advance(watchdog, clock, CONFIG.backoff(1))

    assert capture.restarts == 2
    assert watchdog.stalled


async def test_a_lasting_outage_is_told_once_and_its_end_too(
    watchdog: CaptureWatchdog, capture: FakeCapture, clock: FakeClock, notices: Notices
) -> None:
    capture.errors.extend(DeviceGoneError("no device") for _ in range(6))
    await stall(watchdog, capture, clock)
    await advance(watchdog, clock, 60.0)

    assert capture.restarts > CONFIG.notify_after_attempts + 2
    assert notices.texts == [MICROPHONE_LOST]

    capture.heal_on_restart = True
    await advance(watchdog, clock, 10.0)

    assert not watchdog.stalled
    assert notices.texts == [MICROPHONE_LOST, MICROPHONE_BACK]


async def test_a_short_outage_does_not_bother_the_user(
    watchdog: CaptureWatchdog, capture: FakeCapture, clock: FakeClock, notices: Notices
) -> None:
    capture.errors.append(DeviceGoneError("endpoint re-enumerating"))
    capture.heal_on_restart = True
    await stall(watchdog, capture, clock)
    await advance(watchdog, clock, 5.0)

    assert not watchdog.stalled
    assert notices.texts == []


async def test_each_new_outage_may_notify_again(
    watchdog: CaptureWatchdog, capture: FakeCapture, clock: FakeClock, notices: Notices
) -> None:
    for _ in range(2):
        capture.errors.extend(DeviceGoneError("no device") for _ in range(4))
        capture.heal_on_restart = True
        await stall(watchdog, capture, clock)
        await advance(watchdog, clock, 30.0)

    assert notices.texts == [MICROPHONE_LOST, MICROPHONE_BACK] * 2


async def test_a_stopped_device_is_restarted_after_the_grace_period(
    watchdog: CaptureWatchdog, capture: FakeCapture, clock: FakeClock
) -> None:
    await watchdog.check()
    capture.alive = False
    watchdog.device_lost("device_stopped")
    await advance(watchdog, clock, CONFIG.lost_grace_seconds)

    assert capture.restarts == 1


async def test_a_stop_followed_by_frames_is_a_reroute_not_an_outage(
    watchdog: CaptureWatchdog, capture: FakeCapture, clock: FakeClock
) -> None:
    await watchdog.check()
    watchdog.device_lost("device_stopped")
    watchdog.device_rerouted()
    await advance(watchdog, clock, 5.0)

    assert capture.restarts == 0
    assert not watchdog.stalled


async def test_time_the_watchdog_itself_was_suspended_is_not_held_against_the_mic(
    watchdog: CaptureWatchdog, capture: FakeCapture, clock: FakeClock
) -> None:
    await watchdog.check()
    capture.alive = False
    clock.now += 13 * 60  # the machine slept; nothing ran, not even this loop
    with capture_logs() as logs:
        await watchdog.check()

    assert capture.restarts == 0
    assert any(e["event"] == "audio.watchdog.resumed_after_gap" for e in logs)

    await advance(watchdog, clock, CONFIG.stall_seconds)
    assert capture.restarts == 1


async def test_an_unexpected_restart_error_propagates(
    watchdog: CaptureWatchdog, capture: FakeCapture, clock: FakeClock
) -> None:
    capture.errors.append(ValueError("bug"))

    with pytest.raises(ValueError, match="bug"):
        await stall(watchdog, capture, clock)


async def test_a_failing_notifier_does_not_stop_recovery(
    capture: FakeCapture, clock: FakeClock
) -> None:
    async def broken(text: str) -> None:
        raise OSError("toast service down")

    watchdog = CaptureWatchdog(
        capture,
        config=CONFIG,
        notify=broken,
        clock=clock,
        restart_errors=(DeviceGoneError,),
    )
    capture.errors.extend(DeviceGoneError("no device") for _ in range(4))
    capture.heal_on_restart = True
    with capture_logs() as logs:
        await stall(watchdog, capture, clock)
        await advance(watchdog, clock, 30.0)

    assert not watchdog.stalled
    assert any(e["event"] == "audio.watchdog.notify_failed" for e in logs)


async def test_reset_forgets_an_outage(
    watchdog: CaptureWatchdog, capture: FakeCapture, clock: FakeClock
) -> None:
    await stall(watchdog, capture, clock)
    watchdog.reset()

    assert not watchdog.stalled
    await watchdog.check()
    assert capture.restarts == 1


async def test_run_checks_until_cancelled(capture: FakeCapture) -> None:
    fast = WatchdogConfig(stall_seconds=0.02, check_interval_seconds=0.001)
    watchdog = CaptureWatchdog(capture, config=fast)
    capture.alive = False

    task = asyncio.create_task(watchdog.run())
    async with asyncio.timeout(1.0):
        # The fake has no awaitable signal to wait on; polling is the test.
        while capture.restarts == 0:  # noqa: ASYNC110
            await asyncio.sleep(0.001)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


def test_backoff_doubles_and_caps() -> None:
    assert [CONFIG.backoff(n) for n in range(1, 7)] == [1.0, 2.0, 4.0, 8.0, 8.0, 8.0]
