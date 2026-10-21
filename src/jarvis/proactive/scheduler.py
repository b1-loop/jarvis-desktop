"""Asyncio job scheduler with an injectable clock.

Everything proactive in Jarvis (trigger polling, timed routines, reminders)
runs on one :class:`Scheduler` living on the main event loop. Time is read
through the :class:`Clock` Protocol so the whole module is tested with a fake
clock that advances on demand: no test ever sleeps for real, and a daily job
can be walked across midnight or a DST transition in microseconds.
"""

from __future__ import annotations

import asyncio
import contextlib
import random
import uuid
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, time, timedelta, tzinfo
from typing import Protocol

from jarvis.utils.errors import JarvisError
from jarvis.utils.logging import get_logger

log = get_logger(__name__)

JobCallback = Callable[[], Awaitable[None]]
"""What a job runs: an async function taking no arguments."""

ALL_WEEKDAYS: frozenset[int] = frozenset(range(7))
"""Monday (0) to Sunday (6), ``datetime.weekday`` numbering."""

_DAYS_TO_SEARCH = 8
"""One week plus today covers every weekday set at least once."""


class SchedulerError(JarvisError):
    """A job could not be scheduled, e.g. a naive datetime or an empty weekday set."""


class Clock(Protocol):
    """Source of time for the scheduler; swapped for a fake in tests."""

    def now(self) -> datetime:
        """Current time, timezone-aware."""

    async def sleep(self, seconds: float) -> None:
        """Suspend the caller for ``seconds`` without blocking the loop."""


class SystemClock:
    """Wall-clock :class:`Clock` over ``datetime.now`` and ``asyncio.sleep``.

    Args:
        tz: Zone for ``now()``. ``None`` means the machine's local zone,
            resolved per call so a DST change is picked up without restart.
    """

    def __init__(self, tz: tzinfo | None = None) -> None:
        self._tz = tz

    def now(self) -> datetime:
        """Current time, aware, in the configured or local zone."""
        return datetime.now(self._tz) if self._tz else datetime.now().astimezone()

    async def sleep(self, seconds: float) -> None:
        """Delegate to ``asyncio.sleep``."""
        await asyncio.sleep(seconds)


class Schedule(Protocol):
    """Policy that answers "when does this job run next?"."""

    def next_run(self, now: datetime, previous: datetime | None) -> datetime | None:
        """Return the next run time after ``now``; ``None`` retires the job.

        Args:
            now: Current time from the scheduler's clock.
            previous: The planned time of the run that just happened, or
                ``None`` if the job has never run.
        """


@dataclass(frozen=True, slots=True)
class OneShot:
    """Run once at ``at`` and retire.

    A time already in the past runs on the next scheduler tick rather than
    being dropped: a reminder set for "now" must still fire.
    """

    at: datetime

    def next_run(self, now: datetime, previous: datetime | None) -> datetime | None:
        """``at`` the first time, then nothing."""
        del now
        return None if previous is not None else self.at


@dataclass(frozen=True, slots=True)
class Interval:
    """Run every ``every``, measured from the planned (not actual) run time.

    Anchoring to the planned time stops drift from accumulating. If the
    scheduler was stopped for longer than one period, the missed runs are
    skipped instead of fired in a burst.
    """

    every: timedelta
    run_immediately: bool = False

    def next_run(self, now: datetime, previous: datetime | None) -> datetime | None:
        """Next multiple of ``every`` after ``now``."""
        if previous is None:
            return now if self.run_immediately else now + self.every
        candidate = previous + self.every
        while candidate <= now:
            candidate += self.every
        return candidate


@dataclass(frozen=True, slots=True)
class Daily:
    """Run at a wall-clock time on the given weekdays, in a specific zone.

    The next run is computed by combining a calendar date with ``at`` in
    ``tz`` rather than by adding 24 hours, which is what makes it survive a
    DST transition: "07:30 Europe/Sofia" stays 07:30 on the clock whether the
    night was 23 or 25 hours long.

    Attributes:
        at: Wall-clock time.
        weekdays: ``datetime.weekday`` numbers the job may run on.
        tz: Zone the wall-clock time is expressed in; ``None`` means the
            machine's local zone as resolved by the C library for each date.
    """

    at: time
    weekdays: frozenset[int] = ALL_WEEKDAYS
    tz: tzinfo | None = None

    def next_run(self, now: datetime, previous: datetime | None) -> datetime | None:
        """First candidate strictly after ``now`` on an allowed weekday."""
        del previous
        local_now = now.astimezone(self.tz) if self.tz else now.astimezone()
        for offset in range(_DAYS_TO_SEARCH):
            candidate = self._at_wall_clock(local_now.date() + timedelta(days=offset))
            is_future = candidate.astimezone(UTC) > now.astimezone(UTC)
            if is_future and candidate.weekday() in self.weekdays:
                return candidate
        raise SchedulerError(f"daily job has no allowed weekday: {sorted(self.weekdays)}")

    def _at_wall_clock(self, day: date) -> datetime:
        naive = datetime.combine(day, self.at)
        # A naive datetime's astimezone() asks the C library for the offset
        # that applied on that date, so the local-zone path is DST-aware too.
        return naive.replace(tzinfo=self.tz) if self.tz else naive.astimezone()


@dataclass(slots=True)
class Job:
    """A scheduled callback and its bookkeeping.

    Attributes:
        id: Opaque handle returned by ``Scheduler.add`` and used to cancel.
        name: Human-readable label for logs.
        schedule: When to run.
        callback: What to run.
        jitter_seconds: Upper bound of a random delay added to each run so
            many jobs with the same cadence do not all fire in one burst.
        next_run: Planned time of the next run, or ``None`` once retired.
        runs: How many times the callback has started.
        failures: How many of those raised.
    """

    id: str
    name: str
    schedule: Schedule
    callback: JobCallback
    jitter_seconds: float = 0.0
    next_run: datetime | None = None
    runs: int = 0
    failures: int = 0
    _planned: datetime | None = field(default=None, repr=False)


class Scheduler:
    """Runs jobs on the asyncio loop, one task per run, surviving failures.

    Args:
        clock: Time source; defaults to the wall clock.
        rng: Random source for jitter; seed it in tests.
    """

    def __init__(self, clock: Clock | None = None, *, rng: random.Random | None = None) -> None:
        self._clock: Clock = clock or SystemClock()
        self._rng = rng or random.Random()  # noqa: S311 - jitter is not security-sensitive
        self._jobs: dict[str, Job] = {}
        self._running_tasks: set[asyncio.Task[None]] = set()
        self._wake = asyncio.Event()
        self._loop_task: asyncio.Task[None] | None = None

    @property
    def running(self) -> bool:
        """True between ``start`` and ``stop``."""
        return self._loop_task is not None

    @property
    def jobs(self) -> list[Job]:
        """Snapshot of the registered jobs, in registration order."""
        return list(self._jobs.values())

    def job(self, job_id: str) -> Job | None:
        """Look a job up by id; ``None`` if unknown or retired."""
        return self._jobs.get(job_id)

    def add(
        self, name: str, schedule: Schedule, callback: JobCallback, *, jitter_seconds: float = 0.0
    ) -> str:
        """Register ``callback`` under ``schedule`` and return its job id.

        Raises:
            SchedulerError: If the schedule can never produce a run time or
                jitter is negative.
        """
        if jitter_seconds < 0:
            raise SchedulerError(f"jitter must be >= 0, got {jitter_seconds}")
        job = Job(uuid.uuid4().hex, name, schedule, callback, jitter_seconds)
        first_run = self._plan(job)
        if first_run is None:
            raise SchedulerError(f"schedule for '{name}' never runs")
        self._jobs[job.id] = job
        self._wake.set()
        log.info("proactive.job.added", job=name, job_id=job.id, next_run=first_run.isoformat())
        return job.id

    def add_once(
        self, name: str, at: datetime, callback: JobCallback, *, jitter_seconds: float = 0.0
    ) -> str:
        """Run ``callback`` once at ``at`` (timezone-aware)."""
        _require_aware(at, name)
        return self.add(name, OneShot(at), callback, jitter_seconds=jitter_seconds)

    def add_interval(
        self,
        name: str,
        every: timedelta,
        callback: JobCallback,
        *,
        run_immediately: bool = False,
        jitter_seconds: float = 0.0,
    ) -> str:
        """Run ``callback`` every ``every``, optionally starting right away."""
        if every <= timedelta(0):
            raise SchedulerError(f"interval for '{name}' must be positive, got {every}")
        schedule = Interval(every, run_immediately)
        return self.add(name, schedule, callback, jitter_seconds=jitter_seconds)

    def add_daily(
        self,
        name: str,
        at: time,
        callback: JobCallback,
        *,
        weekdays: Iterable[int] = ALL_WEEKDAYS,
        tz: tzinfo | None = None,
        jitter_seconds: float = 0.0,
    ) -> str:
        """Run ``callback`` at wall-clock ``at`` on ``weekdays`` in zone ``tz``."""
        days = frozenset(weekdays)
        if not days or not days <= ALL_WEEKDAYS:
            raise SchedulerError(f"weekdays for '{name}' must be a non-empty subset of 0..6")
        schedule = Daily(at, days, tz)
        return self.add(name, schedule, callback, jitter_seconds=jitter_seconds)

    def cancel(self, job_id: str) -> bool:
        """Remove a job; returns False if it was unknown or already retired.

        A run that is already in flight is left to finish; cancelling only
        prevents future runs.
        """
        job = self._jobs.pop(job_id, None)
        if job is None:
            return False
        job.next_run = None
        self._wake.set()
        log.info("proactive.job.cancelled", job=job.name, job_id=job_id)
        return True

    async def __aenter__(self) -> Scheduler:
        await self.start()
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.stop()

    async def start(self) -> None:
        """Start dispatching; idempotent."""
        if self._loop_task is not None:
            return
        self._loop_task = asyncio.create_task(self._run_forever(), name="scheduler")
        log.info("proactive.scheduler.started", jobs=len(self._jobs))

    async def stop(self) -> None:
        """Stop dispatching and wait for in-flight runs to finish; idempotent."""
        if self._loop_task is None:
            return
        self._loop_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._loop_task
        self._loop_task = None
        if self._running_tasks:
            await asyncio.gather(*self._running_tasks)
        log.info("proactive.scheduler.stopped")

    async def _run_forever(self) -> None:
        try:
            while True:
                self._dispatch_due()
                await self._wait_for_next()
        except asyncio.CancelledError:
            raise
        except Exception:
            # The loop dying would silently end all proactive behaviour;
            # make the cause loud before the task carries it.
            log.exception("proactive.scheduler.crashed")
            raise

    def _dispatch_due(self) -> None:
        now = self._clock.now().astimezone(UTC)
        for job in list(self._jobs.values()):
            if job.next_run is not None and job.next_run.astimezone(UTC) <= now:
                self._spawn(job)
                if self._plan(job) is None:
                    del self._jobs[job.id]

    def _spawn(self, job: Job) -> None:
        job.runs += 1
        task = asyncio.create_task(self._run_job(job), name=f"job:{job.name}")
        self._running_tasks.add(task)
        task.add_done_callback(self._running_tasks.discard)

    async def _run_job(self, job: Job) -> None:
        log.debug("proactive.job.start", job=job.name, run=job.runs)
        try:
            await job.callback()
        except asyncio.CancelledError:
            raise
        except Exception:
            # Logged with traceback, counted, and the scheduler carries on:
            # one broken trigger must not take the morning routine with it.
            job.failures += 1
            log.exception("proactive.job.failed", job=job.name, failures=job.failures)

    def _plan(self, job: Job) -> datetime | None:
        """Compute and store the job's next run; ``None`` means it is retired."""
        planned = job.schedule.next_run(self._clock.now(), job._planned)
        job._planned = planned
        if planned is not None and job.jitter_seconds > 0:
            planned += timedelta(seconds=self._rng.uniform(0, job.jitter_seconds))
        job.next_run = planned
        return planned

    async def _wait_for_next(self) -> None:
        """Sleep until the earliest ``next_run`` or until a job is added/cancelled."""
        self._wake.clear()
        delay = self._seconds_until_next()
        waiter = asyncio.create_task(self._wake.wait())
        sleeper = asyncio.create_task(self._clock.sleep(delay)) if delay is not None else None
        pending = {waiter, sleeper} if sleeper else {waiter}
        try:
            await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)

    def _seconds_until_next(self) -> float | None:
        upcoming = [
            job.next_run.astimezone(UTC) for job in self._jobs.values() if job.next_run is not None
        ]
        if not upcoming:
            return None
        now = self._clock.now().astimezone(UTC)
        return max(0.0, (min(upcoming) - now).total_seconds())


def _require_aware(moment: datetime, name: str) -> None:
    if moment.tzinfo is None or moment.utcoffset() is None:
        raise SchedulerError(f"run time for '{name}' must be timezone-aware, got {moment!r}")


# Design notes:
# - One loop, many tasks: the loop only computes "what is due" and spawns a
#   task per run, so a slow job never delays the others and a raising job is
#   logged and counted rather than crashing the dispatcher.
# - The clock is a Protocol with both now() and sleep() because deterministic
#   tests need to control both: a fake that only moved now() would still
#   leave the loop blocked in a real asyncio.sleep.
# - _wait_for_next races a sleep against a wake-up Event instead of using
#   asyncio.wait_for(..., timeout) because wait_for reads the loop's monotonic
#   clock, which the fake clock cannot steer.
# - Instants are compared in UTC: Python compares and subtracts two datetimes
#   that share a tzinfo by their wall-clock fields, which is off by an hour
#   across a DST change. Converting first keeps "is it due?" exact.
# - Daily schedules recompute from calendar date + wall time on every plan,
#   never "previous + 24 h", which is the entire DST story. Non-existent
#   times on a spring-forward day resolve to the pre-transition offset (one
#   run, an hour later than the label); ambiguous fall-back times fire once.
# - Assumption: the composition root passes a ZoneInfo for daily jobs when
#   the user configures one; with tz=None the C library's local zone is used,
#   which is DST-aware for the system zone but not portable across machines.
