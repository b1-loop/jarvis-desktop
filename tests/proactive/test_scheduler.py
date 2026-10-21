import asyncio
import random
from collections.abc import AsyncIterator
from datetime import UTC, datetime, time, timedelta
from zoneinfo import ZoneInfo

import pytest

from jarvis.proactive.scheduler import (
    Daily,
    Interval,
    OneShot,
    Scheduler,
    SchedulerError,
    SystemClock,
)
from tests.proactive.fakes import FakeClock

SOFIA = ZoneInfo("Europe/Sofia")
START = datetime(2026, 10, 21, 12, 0, tzinfo=SOFIA)  # a Wednesday
MINUTE = timedelta(minutes=1)


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock(START)


@pytest.fixture
async def scheduler(clock: FakeClock) -> AsyncIterator[Scheduler]:
    rng = random.Random(0)  # noqa: S311 - seeded so jitter is reproducible
    async with Scheduler(clock, rng=rng) as scheduler:
        await clock.settle()
        yield scheduler


class Recorder:
    """Callback that records when it ran, by the fake clock."""

    def __init__(self, clock: FakeClock) -> None:
        self._clock = clock
        self.ran_at: list[datetime] = []

    async def __call__(self) -> None:
        self.ran_at.append(self._clock.now())


# --- one-shot ---------------------------------------------------------------


async def test_one_shot_runs_at_its_time_and_retires(
    scheduler: Scheduler, clock: FakeClock
) -> None:
    recorder = Recorder(clock)
    job_id = scheduler.add_once("ping", START + 5 * MINUTE, recorder)

    await clock.advance(4 * MINUTE)
    assert recorder.ran_at == []

    await clock.advance(1 * MINUTE)
    assert recorder.ran_at == [START + 5 * MINUTE]
    assert scheduler.job(job_id) is None

    await clock.advance(60 * MINUTE)
    assert len(recorder.ran_at) == 1


async def test_one_shot_in_the_past_runs_on_next_tick(
    scheduler: Scheduler, clock: FakeClock
) -> None:
    recorder = Recorder(clock)
    scheduler.add_once("late", START - MINUTE, recorder)

    await clock.settle()
    assert recorder.ran_at == [START]


async def test_one_shot_rejects_naive_datetime(scheduler: Scheduler, clock: FakeClock) -> None:
    with pytest.raises(SchedulerError, match="timezone-aware"):
        scheduler.add_once("naive", datetime(2026, 1, 1, 8, 0), Recorder(clock))


# --- interval -----------------------------------------------------------------


async def test_interval_runs_every_period(scheduler: Scheduler, clock: FakeClock) -> None:
    recorder = Recorder(clock)
    scheduler.add_interval("tick", 10 * MINUTE, recorder)

    await clock.advance(35 * MINUTE)

    assert recorder.ran_at == [START + 10 * MINUTE, START + 20 * MINUTE, START + 30 * MINUTE]


async def test_interval_can_run_immediately(scheduler: Scheduler, clock: FakeClock) -> None:
    recorder = Recorder(clock)
    scheduler.add_interval("now", 10 * MINUTE, recorder, run_immediately=True)

    await clock.settle()
    assert recorder.ran_at == [START]

    await clock.advance(10 * MINUTE)
    assert recorder.ran_at == [START, START + 10 * MINUTE]


async def test_interval_skips_missed_periods_after_a_pause(
    scheduler: Scheduler, clock: FakeClock
) -> None:
    recorder = Recorder(clock)
    scheduler.add_interval("tick", 10 * MINUTE, recorder)

    # The machine slept through three periods; wake it 5 minutes into the fourth.
    clock.jump(START + 35 * MINUTE)
    await clock.advance(timedelta(0))
    assert recorder.ran_at == [START + 35 * MINUTE]

    await clock.advance(10 * MINUTE)
    # Anchored to the planned cadence (every 10 min from START), not to the late run.
    assert recorder.ran_at[-1] == START + 40 * MINUTE


async def test_interval_must_be_positive(scheduler: Scheduler, clock: FakeClock) -> None:
    with pytest.raises(SchedulerError, match="positive"):
        scheduler.add_interval("bad", timedelta(0), Recorder(clock))


# --- daily --------------------------------------------------------------------


async def test_daily_rolls_over_midnight(clock: FakeClock) -> None:
    clock.jump(datetime(2026, 10, 21, 23, 59, tzinfo=SOFIA))
    recorder = Recorder(clock)
    async with Scheduler(clock) as scheduler:
        await clock.settle()
        job_id = scheduler.add_daily("morning", time(7, 30), recorder, tz=SOFIA)
        job = scheduler.job(job_id)
        assert job is not None
        assert job.next_run == datetime(2026, 10, 22, 7, 30, tzinfo=SOFIA)

        await clock.advance(timedelta(hours=7, minutes=31))
        assert recorder.ran_at == [datetime(2026, 10, 22, 7, 30, tzinfo=SOFIA)]
        assert job.next_run == datetime(2026, 10, 23, 7, 30, tzinfo=SOFIA)


async def test_daily_honours_weekdays(scheduler: Scheduler, clock: FakeClock) -> None:
    friday_noon = datetime(2026, 10, 23, 12, 0, tzinfo=SOFIA)
    clock.jump(friday_noon)
    job_id = scheduler.add_daily(
        "standup", time(9, 0), Recorder(clock), weekdays=range(5), tz=SOFIA
    )

    job = scheduler.job(job_id)
    assert job is not None
    assert job.next_run == datetime(2026, 10, 26, 9, 0, tzinfo=SOFIA)
    assert job.next_run.weekday() == 0


@pytest.mark.parametrize(
    ("day_before", "expected_gap", "offset_before", "offset_after"),
    [
        (datetime(2026, 3, 28, 7, 0, tzinfo=SOFIA), 23, 2, 3),  # spring forward
        (datetime(2026, 10, 24, 7, 0, tzinfo=SOFIA), 25, 3, 2),  # fall back
    ],
)
async def test_daily_keeps_wall_clock_time_across_dst(
    day_before: datetime, expected_gap: int, offset_before: int, offset_after: int
) -> None:
    clock = FakeClock(day_before)
    recorder = Recorder(clock)
    async with Scheduler(clock) as scheduler:
        await clock.settle()
        scheduler.add_daily("morning", time(7, 30), recorder, tz=SOFIA)
        await clock.advance(timedelta(hours=49))

    first, second = recorder.ran_at[:2]
    assert (first.hour, first.minute) == (7, 30)
    assert (second.hour, second.minute) == (7, 30)
    # Subtract in UTC: same-zone datetimes subtract by wall clock in Python.
    assert second.astimezone(UTC) - first.astimezone(UTC) == timedelta(hours=expected_gap)
    assert first.utcoffset() == timedelta(hours=offset_before)
    assert second.utcoffset() == timedelta(hours=offset_after)


def test_daily_schedule_in_utc_is_plain_arithmetic() -> None:
    now = datetime(2026, 3, 28, 8, 0, tzinfo=UTC)
    schedule = Daily(time(7, 30), tz=UTC)

    assert schedule.next_run(now, None) == datetime(2026, 3, 29, 7, 30, tzinfo=UTC)


def test_daily_with_local_zone_is_aware() -> None:
    schedule = Daily(time(7, 30))
    nxt = schedule.next_run(datetime.now().astimezone(), None)

    assert nxt is not None
    assert nxt.utcoffset() is not None
    assert (nxt.hour, nxt.minute) == (7, 30)


def test_daily_with_impossible_weekdays_never_runs() -> None:
    with pytest.raises(SchedulerError, match="no allowed weekday"):
        Daily(time(7, 30), frozenset({9}), tz=UTC).next_run(START, None)


async def test_add_daily_validates_weekdays(scheduler: Scheduler, clock: FakeClock) -> None:
    with pytest.raises(SchedulerError, match="weekdays"):
        scheduler.add_daily("bad", time(7, 30), Recorder(clock), weekdays=[])
    with pytest.raises(SchedulerError, match="weekdays"):
        scheduler.add_daily("bad", time(7, 30), Recorder(clock), weekdays=[7])


# --- cancel, failures, jitter, lifecycle ---------------------------------------


async def test_cancel_prevents_future_runs(scheduler: Scheduler, clock: FakeClock) -> None:
    recorder = Recorder(clock)
    job_id = scheduler.add_interval("tick", MINUTE, recorder)
    await clock.advance(MINUTE)

    assert scheduler.cancel(job_id) is True
    assert scheduler.cancel(job_id) is False
    await clock.advance(5 * MINUTE)

    assert len(recorder.ran_at) == 1
    assert scheduler.jobs == []


async def test_failing_job_is_logged_counted_and_does_not_stop_others(
    scheduler: Scheduler, clock: FakeClock
) -> None:
    async def explode() -> None:
        raise RuntimeError("boom")

    recorder = Recorder(clock)
    bad_id = scheduler.add_interval("bad", MINUTE, explode)
    scheduler.add_interval("good", MINUTE, recorder)

    await clock.advance(3 * MINUTE)

    bad = scheduler.job(bad_id)
    assert bad is not None
    assert bad.runs == 3
    assert bad.failures == 3
    assert len(recorder.ran_at) == 3
    assert scheduler.running


async def test_jitter_delays_within_bounds(clock: FakeClock) -> None:
    rng = random.Random(42)  # noqa: S311 - seeded so jitter is reproducible
    scheduler = Scheduler(clock, rng=rng)
    job_id = scheduler.add_interval("jittery", 10 * MINUTE, Recorder(clock), jitter_seconds=60)

    job = scheduler.job(job_id)
    assert job is not None
    assert job.next_run is not None
    assert START + 10 * MINUTE <= job.next_run <= START + 11 * MINUTE


async def test_negative_jitter_is_rejected(scheduler: Scheduler, clock: FakeClock) -> None:
    with pytest.raises(SchedulerError, match="jitter"):
        scheduler.add_interval("bad", MINUTE, Recorder(clock), jitter_seconds=-1)


async def test_adding_a_sooner_job_wakes_the_loop(scheduler: Scheduler, clock: FakeClock) -> None:
    late = Recorder(clock)
    soon = Recorder(clock)
    scheduler.add_once("late", START + timedelta(hours=1), late)
    await clock.settle()

    scheduler.add_once("soon", START + MINUTE, soon)
    await clock.advance(MINUTE)

    assert soon.ran_at == [START + MINUTE]
    assert late.ran_at == []


async def test_stop_waits_for_in_flight_jobs(clock: FakeClock) -> None:
    gate = asyncio.Event()
    finished: list[bool] = []

    async def slow() -> None:
        await gate.wait()
        finished.append(True)

    scheduler = Scheduler(clock)
    scheduler.add_once("slow", START, slow)
    await scheduler.start()
    await clock.settle()

    stopping = asyncio.create_task(scheduler.stop())
    await clock.settle()
    assert not stopping.done()

    gate.set()
    await stopping
    assert finished == [True]
    assert not scheduler.running


async def test_start_and_stop_are_idempotent(clock: FakeClock) -> None:
    scheduler = Scheduler(clock)
    await scheduler.start()
    await scheduler.start()
    assert scheduler.running
    await scheduler.stop()
    await scheduler.stop()
    assert not scheduler.running


async def test_jobs_added_before_start_run_after_start(clock: FakeClock) -> None:
    scheduler = Scheduler(clock)
    recorder = Recorder(clock)
    scheduler.add_once("early", START + MINUTE, recorder)

    async with scheduler:
        await clock.advance(MINUTE)

    assert recorder.ran_at == [START + MINUTE]


async def test_idle_scheduler_does_not_sleep(scheduler: Scheduler, clock: FakeClock) -> None:
    assert scheduler.jobs == []
    assert clock.sleeping == 0


# --- schedules and the system clock ------------------------------------------


def test_one_shot_and_interval_policies() -> None:
    at = START + MINUTE
    assert OneShot(at).next_run(START, None) == at
    assert OneShot(at).next_run(START, at) is None

    every = Interval(10 * MINUTE)
    assert every.next_run(START, None) == START + 10 * MINUTE
    assert every.next_run(START + 25 * MINUTE, START) == START + 30 * MINUTE


async def test_system_clock_is_aware_and_sleeps() -> None:
    clock = SystemClock()
    assert clock.now().utcoffset() is not None
    assert SystemClock(UTC).now().tzinfo is UTC
    await clock.sleep(0)
