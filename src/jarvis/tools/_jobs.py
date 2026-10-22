"""``BackgroundJobs``: watches processes a tool started and left running.

start_dev_server spawns a process and returns immediately; somebody still
has to notice when it exits (a crashed dev server, a finished build). This
in-memory tracker awaits each process in its own task and queues the
outcome, and its ``pop_finished`` satisfies
``jarvis.proactive.triggers.CommandWatcher``, so the
``LongRunningCommandTrigger`` can announce "your dev server stopped after 40
minutes" without either side knowing about the other.

Private module (ignored by the tool registry); the composition root builds
one instance, passes it as ``ToolContext.jobs`` and to the trigger.
"""

from __future__ import annotations

import asyncio
import contextlib
import itertools
from dataclasses import dataclass
from datetime import timedelta
from typing import TYPE_CHECKING

from jarvis.proactive.triggers import CommandResult
from jarvis.utils.logging import get_logger
from jarvis.utils.timing import Clock, SystemClock

if TYPE_CHECKING:
    from jarvis.agent.tool import TrackedProcess

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class RunningJob:
    """A tracked process that has not exited yet, for the dashboard.

    Attributes:
        id: Job id handed out by :meth:`BackgroundJobs.track`.
        command: The command line as started.
        pid: Operating-system process id.
        started: Monotonic start time from the tracker's clock.
    """

    id: str
    command: str
    pid: int
    started: float


class BackgroundJobs:
    """In-memory registry of background processes and their outcomes.

    Args:
        clock: Monotonic time source for durations; a fake in tests.
    """

    def __init__(self, clock: Clock | None = None) -> None:
        self._clock: Clock = clock or SystemClock()
        self._ids = itertools.count(1)
        self._running: dict[str, RunningJob] = {}
        self._finished: list[CommandResult] = []
        self._watchers: set[asyncio.Task[None]] = set()

    @property
    def running(self) -> list[RunningJob]:
        """Jobs that are still running, oldest first."""
        return list(self._running.values())

    def track(self, command: str, process: TrackedProcess) -> str:
        """Start watching ``process`` and return its job id.

        Must be called from within the event loop (tools always are).
        """
        job = RunningJob(f"job-{next(self._ids)}", command, process.pid, self._clock.monotonic())
        self._running[job.id] = job
        watcher = asyncio.create_task(self._watch(job, process), name=f"watch-{job.id}")
        self._watchers.add(watcher)
        watcher.add_done_callback(self._watchers.discard)
        log.info("jobs.tracked", job_id=job.id, pid=job.pid, command=command)
        return job.id

    async def pop_finished(self) -> list[CommandResult]:
        """Remove and return jobs that exited since the previous call."""
        finished, self._finished = self._finished, []
        return finished

    async def join(self) -> None:
        """Wait until every tracked process has exited and been recorded.

        For orderly shutdown (wrapped in a timeout by the caller) and tests.
        """
        if self._watchers:
            await asyncio.gather(*self._watchers)

    async def close(self) -> None:
        """Stop watching (the processes themselves keep running)."""
        for watcher in list(self._watchers):
            watcher.cancel()
        for watcher in list(self._watchers):
            with contextlib.suppress(asyncio.CancelledError):
                await watcher

    async def _watch(self, job: RunningJob, process: TrackedProcess) -> None:
        exit_code = await process.wait()
        duration = timedelta(seconds=self._clock.monotonic() - job.started)
        self._running.pop(job.id, None)
        self._finished.append(CommandResult(job.id, job.command, exit_code, duration))
        log.info("jobs.finished", job_id=job.id, exit_code=exit_code, seconds=duration.seconds)


# Design notes:
# - One watcher task per process instead of polling: ``wait()`` costs
#   nothing while the process runs and reports the exit the moment it
#   happens. The task set keeps strong references so tasks are not
#   garbage-collected mid-wait.
# - close() cancels only the watchers. Stopping Jarvis should not kill the
#   user's dev server; it was started on their behalf and is theirs.
# - CancelledError is suppressed only for the watchers this method itself
#   cancelled, which is the documented way to await a cancelled task.
