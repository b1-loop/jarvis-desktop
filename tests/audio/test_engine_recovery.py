"""AudioEngine with capture recovery: device events, the watchdog task, state after a restart."""

import asyncio
from collections.abc import Callable

import pytest
from structlog.testing import capture_logs

from jarvis.audio.engine import AudioEngine, AudioEventKind, CaptureRecovery
from jarvis.audio.watchdog import WatchdogConfig
from tests.audio.fakes import FakeNativeEngine, native_event

POLL = 0.001
WAIT = 1.0
FAST = WatchdogConfig(
    stall_seconds=0.03,
    lost_grace_seconds=0.0,
    check_interval_seconds=0.001,
    initial_backoff_seconds=0.01,
    max_backoff_seconds=0.02,
)


def recovering(native: FakeNativeEngine) -> AudioEngine:
    return AudioEngine(native, poll_interval=POLL, recovery=CaptureRecovery(config=FAST))


async def eventually(condition: Callable[[], bool]) -> None:
    """Poll ``condition`` until it holds; the fakes expose no awaitable signal."""
    async with asyncio.timeout(WAIT):
        while not condition():  # noqa: ASYNC110
            await asyncio.sleep(POLL)


async def wait_for_restarts(native: FakeNativeEngine, count: int) -> None:
    await eventually(lambda: native.restarts >= count)


@pytest.mark.parametrize(
    "name",
    ["DEVICE_STOPPED", "DEVICE_REROUTED", "DEVICE_INTERRUPTED", "DEVICE_RESUMED"],
)
async def test_device_events_are_converted_and_queued(name: str) -> None:
    native = FakeNativeEngine()
    native.script(native_event(name))
    async with AudioEngine(native, poll_interval=POLL) as engine:
        event = await asyncio.wait_for(engine.next_event(), WAIT)

    assert event.kind is AudioEventKind[name]


async def test_a_stalled_capture_is_restarted_by_the_watchdog() -> None:
    native = FakeNativeEngine(delivery=0)

    async with recovering(native):
        await wait_for_restarts(native, 1)


async def test_a_live_capture_is_left_alone() -> None:
    native = FakeNativeEngine()

    async with recovering(native):
        await asyncio.sleep(FAST.stall_seconds * 5)

    assert native.restarts == 0


async def test_a_stopped_device_event_triggers_a_restart() -> None:
    native = FakeNativeEngine()
    native.script(native_event("DEVICE_STOPPED"))
    native.delivery = 0

    with capture_logs() as logs:
        async with recovering(native):
            await wait_for_restarts(native, 1)

    lost = next(e for e in logs if e["event"] == "audio.capture.device_lost")
    assert lost["reason"] == "device_stopped"


async def test_without_recovery_no_watchdog_runs() -> None:
    native = FakeNativeEngine(delivery=0)

    async with AudioEngine(native, poll_interval=POLL):
        await asyncio.sleep(FAST.stall_seconds * 5)

    assert native.restarts == 0


async def test_a_restart_forgets_speech_that_the_gap_cut_off() -> None:
    native = FakeNativeEngine()
    native.script(native_event("SPEECH_STARTED"))
    async with AudioEngine(native, poll_interval=POLL) as engine:
        await asyncio.wait_for(engine.next_event(AudioEventKind.SPEECH_STARTED), WAIT)
        assert engine.speech_active

        assert await engine.restart_capture()

    assert not engine.speech_active
    assert native.restarts == 1


async def test_stopping_the_engine_stops_the_watchdog() -> None:
    native = FakeNativeEngine()
    engine = recovering(native)
    await engine.start()
    await engine.stop()
    native.delivery = 0

    await asyncio.sleep(FAST.stall_seconds * 5)

    assert native.restarts == 0


async def test_a_crashed_watchdog_is_reported_when_the_engine_stops() -> None:
    native = FakeNativeEngine(delivery=0)
    native.restart_errors.append(ValueError("bug in recovery"))

    with capture_logs() as logs:
        engine = recovering(native)
        await engine.start()
        await eventually(lambda: any(e["event"] == "audio.watchdog.crashed" for e in logs))
        await engine.stop()

    assert any(e["event"] == "audio.watchdog.was_dead" for e in logs)
