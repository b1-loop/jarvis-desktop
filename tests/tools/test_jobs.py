import asyncio
from datetime import timedelta

from jarvis.agent.tool import BackgroundJobTracker
from jarvis.proactive.triggers import CommandWatcher, LongRunningCommandTrigger
from jarvis.tools._jobs import BackgroundJobs
from tests.fakes import FakeClock


class FakeProcess:
    def __init__(self, pid: int) -> None:
        self._pid = pid
        self._exit = asyncio.get_running_loop().create_future()

    @property
    def pid(self) -> int:
        return self._pid

    def exit(self, code: int) -> None:
        self._exit.set_result(code)

    async def wait(self) -> int:
        return int(await self._exit)


async def settle() -> None:
    for _ in range(3):
        await asyncio.sleep(0)


async def test_satisfies_both_protocols() -> None:
    jobs = BackgroundJobs()
    tracker: BackgroundJobTracker = jobs
    watcher: CommandWatcher = jobs

    assert tracker is watcher


async def test_reports_each_finished_job_once_with_its_duration() -> None:
    clock = FakeClock()
    jobs = BackgroundJobs(clock)
    process = FakeProcess(pid=77)

    job_id = jobs.track("npm run dev", process)
    assert [job.pid for job in jobs.running] == [77]
    assert await jobs.pop_finished() == []
    clock.advance(90)
    process.exit(1)
    await settle()

    finished = await jobs.pop_finished()
    assert [(r.id, r.command, r.exit_code, r.duration) for r in finished] == [
        (job_id, "npm run dev", 1, timedelta(seconds=90))
    ]
    assert jobs.running == []
    assert await jobs.pop_finished() == []


async def test_feeds_the_long_running_command_trigger() -> None:
    clock = FakeClock()
    jobs = BackgroundJobs(clock)
    process = FakeProcess(pid=5)
    jobs.track("cargo run", process)
    clock.advance(600)
    process.exit(0)
    await settle()

    notification = await LongRunningCommandTrigger(jobs).check()

    assert notification is not None
    assert notification.spoken_text == "Your command cargo run finished after about 10 minutes."


async def test_close_stops_watching_without_touching_the_process() -> None:
    jobs = BackgroundJobs()
    process = FakeProcess(pid=9)
    jobs.track("serve", process)

    await jobs.close()
    process.exit(0)
    await settle()

    assert await jobs.pop_finished() == []
