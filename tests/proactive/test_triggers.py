from collections.abc import AsyncIterator
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from jarvis.proactive.scheduler import Scheduler
from jarvis.proactive.triggers import (
    BatteryStatus,
    CalendarEvent,
    CommandResult,
    DueReminderTrigger,
    LongRunningCommandTrigger,
    LowBatteryTrigger,
    Notification,
    Priority,
    Reminder,
    TriggerRunner,
    UpcomingMeetingTrigger,
)
from tests.proactive.fakes import (
    FakeClock,
    FakeSink,
    ScriptedTrigger,
    StubBatterySource,
    StubCalendarSource,
    StubCommandWatcher,
    StubReminderSource,
)

SOFIA = ZoneInfo("Europe/Sofia")
START = datetime(2026, 10, 21, 9, 0, tzinfo=SOFIA)
MINUTE = timedelta(minutes=1)
POLL = timedelta(seconds=30)
COOLDOWN = timedelta(minutes=5)


def note(title: str, key: str | None = "k") -> Notification:
    return Notification(title=title, spoken_text=title, dedupe_key=key)


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock(START)


@pytest.fixture
def sink() -> FakeSink:
    return FakeSink()


@pytest.fixture
async def scheduler(clock: FakeClock) -> AsyncIterator[Scheduler]:
    async with Scheduler(clock) as scheduler:
        await clock.settle()
        yield scheduler


@pytest.fixture
def runner(scheduler: Scheduler, sink: FakeSink, clock: FakeClock) -> TriggerRunner:
    return TriggerRunner(scheduler, sink, clock=clock, poll=POLL, cooldown=COOLDOWN)


# --- runner -------------------------------------------------------------------


async def test_trigger_is_polled_immediately_and_then_every_period(
    runner: TriggerRunner, clock: FakeClock, sink: FakeSink
) -> None:
    trigger = ScriptedTrigger("t", [note("first", "a"), None, note("second", "b")])
    runner.add(trigger)

    await clock.settle()
    assert trigger.checks == 1
    assert [n.title for n in sink.received] == ["first"]

    await clock.advance(2 * POLL)
    assert trigger.checks == 3
    assert [n.title for n in sink.received] == ["first", "second"]
    assert runner.published == 2


async def test_same_key_is_suppressed_within_cooldown_and_published_after(
    runner: TriggerRunner, clock: FakeClock, sink: FakeSink
) -> None:
    trigger = ScriptedTrigger("t", [note("one"), note("two"), note("three"), note("four")])
    runner.add(trigger, poll=2 * MINUTE)

    await clock.advance(4 * MINUTE)  # polls at 0, 2 and 4 minutes
    assert [n.title for n in sink.received] == ["one"]
    assert runner.suppressed == 2

    await clock.advance(2 * MINUTE)  # 6 minutes: past the 5-minute cooldown
    assert [n.title for n in sink.received] == ["one", "four"]


async def test_notifications_without_key_are_never_suppressed(runner: TriggerRunner) -> None:
    trigger = ScriptedTrigger("t", [note("a", None), note("b", None)])
    runner.add(trigger)

    assert await runner.poll("t") is not None
    assert await runner.poll("t") is not None
    assert runner.suppressed == 0


async def test_failing_trigger_is_counted_and_others_keep_running(
    runner: TriggerRunner, clock: FakeClock, sink: FakeSink
) -> None:
    broken = ScriptedTrigger("broken", [RuntimeError("api down"), note("recovered", "x")])
    healthy = ScriptedTrigger("healthy", [None, note("fine", "y")])
    runner.add(broken)
    runner.add(healthy)

    await clock.advance(POLL)

    assert runner.failures("broken") == 1
    assert {n.title for n in sink.received} == {"recovered", "fine"}


async def test_duplicate_trigger_names_are_rejected(runner: TriggerRunner) -> None:
    runner.add(ScriptedTrigger("t"))
    with pytest.raises(ValueError, match="already registered"):
        runner.add(ScriptedTrigger("t"))


async def test_remove_cancels_polling(
    runner: TriggerRunner, scheduler: Scheduler, clock: FakeClock
) -> None:
    trigger = ScriptedTrigger("t")
    job_id = runner.add(trigger)
    await clock.settle()

    assert runner.remove("t") is True
    assert runner.remove("t") is False
    assert scheduler.job(job_id) is None
    assert runner.triggers == []

    await clock.advance(3 * POLL)
    assert trigger.checks == 1


# --- upcoming meeting -----------------------------------------------------------


def event(title: str, minutes_from_now: int, location: str | None = None) -> CalendarEvent:
    return CalendarEvent(f"id-{title}", title, START + minutes_from_now * MINUTE, location)


async def test_meeting_trigger_reports_the_earliest_future_event(clock: FakeClock) -> None:
    source = StubCalendarSource([event("Later", 9), event("Standup", 4, "Room 2")])
    trigger = UpcomingMeetingTrigger(source, lead=10 * MINUTE, clock=clock)

    notification = await trigger.check()

    assert notification is not None
    assert notification.title == "Standup in 4 min"
    assert "Room 2" in notification.spoken_text
    assert notification.priority is Priority.HIGH
    assert notification.dedupe_key == f"meeting:id-Standup:{(START + 4 * MINUTE).isoformat()}"
    assert source.requested_windows == [10 * MINUTE]
    assert [a.id for a in notification.actions] == ["snooze"]


async def test_meeting_trigger_ignores_events_that_already_started(clock: FakeClock) -> None:
    trigger = UpcomingMeetingTrigger(StubCalendarSource([event("Past", -1)]), clock=clock)
    assert await trigger.check() is None


async def test_meeting_trigger_is_quiet_with_empty_calendar(clock: FakeClock) -> None:
    trigger = UpcomingMeetingTrigger(StubCalendarSource(), clock=clock)
    assert await trigger.check() is None


# --- low battery --------------------------------------------------------------


async def test_battery_trigger_warns_once_per_threshold_and_rearms_on_charge() -> None:
    source = StubBatterySource(BatteryStatus(50, plugged_in=False))
    trigger = LowBatteryTrigger(source, thresholds=(20, 10))

    assert await trigger.check() is None

    source.reading = BatteryStatus(19, plugged_in=False)
    first = await trigger.check()
    assert first is not None
    assert first.dedupe_key == "battery:20"
    assert first.priority is Priority.NORMAL
    assert await trigger.check() is None

    source.reading = BatteryStatus(9, plugged_in=False)
    second = await trigger.check()
    assert second is not None
    assert second.dedupe_key == "battery:10"
    assert second.priority is Priority.URGENT

    source.reading = BatteryStatus(9, plugged_in=True)
    assert await trigger.check() is None
    source.reading = BatteryStatus(8, plugged_in=False)
    rearmed = await trigger.check()
    assert rearmed is not None
    assert rearmed.dedupe_key == "battery:10"


async def test_battery_trigger_skips_both_thresholds_in_one_drop() -> None:
    source = StubBatterySource(BatteryStatus(4, plugged_in=False))
    trigger = LowBatteryTrigger(source)

    notification = await trigger.check()

    assert notification is not None
    assert notification.dedupe_key == "battery:5"
    assert await trigger.check() is None


async def test_battery_trigger_rearms_when_level_recovers() -> None:
    source = StubBatterySource(BatteryStatus(15, plugged_in=False))
    trigger = LowBatteryTrigger(source, thresholds=(20,))
    assert await trigger.check() is not None

    source.reading = BatteryStatus(80, plugged_in=False)
    assert await trigger.check() is None
    source.reading = BatteryStatus(15, plugged_in=False)
    assert await trigger.check() is not None


async def test_battery_trigger_is_quiet_without_a_battery() -> None:
    trigger = LowBatteryTrigger(StubBatterySource(None))
    assert await trigger.check() is None


def test_battery_trigger_needs_a_threshold() -> None:
    with pytest.raises(ValueError, match="threshold"):
        LowBatteryTrigger(StubBatterySource(), thresholds=())


# --- due reminders ------------------------------------------------------------


async def test_reminder_trigger_pops_due_reminders(clock: FakeClock) -> None:
    source = StubReminderSource(
        [
            Reminder("r1", "call the dentist", START - MINUTE),
            Reminder("r2", "water the plants", START + 10 * MINUTE),
        ]
    )
    trigger = DueReminderTrigger(source, clock=clock)

    notification = await trigger.check()

    assert notification is not None
    assert notification.title == "call the dentist"
    assert notification.spoken_text == "Reminder: call the dentist."
    assert notification.dedupe_key == "reminder:r1"
    assert await trigger.check() is None
    assert [r.id for r in source.reminders] == ["r2"]


async def test_reminder_trigger_folds_several_into_one(clock: FakeClock) -> None:
    source = StubReminderSource([Reminder("a", "one", START), Reminder("b", "two", START)])
    trigger = DueReminderTrigger(source, clock=clock)

    notification = await trigger.check()

    assert notification is not None
    assert notification.title == "2 reminders"
    assert notification.spoken_text == "Reminder: one. Also: two."
    assert notification.dedupe_key == "reminder:a,b"


# --- long-running commands ------------------------------------------------------


async def test_command_trigger_announces_long_commands_only() -> None:
    watcher = StubCommandWatcher(
        [
            CommandResult("c1", "ls", 0, timedelta(seconds=1)),
            CommandResult("c2", "pytest", 0, timedelta(minutes=4)),
        ]
    )
    trigger = LongRunningCommandTrigger(watcher, min_duration=timedelta(seconds=30))

    notification = await trigger.check()

    assert notification is not None
    assert notification.title == "Command finished"
    assert "pytest finished after about 4 minutes" in notification.spoken_text
    assert notification.priority is Priority.NORMAL
    assert notification.dedupe_key == "command:c2"
    assert await trigger.check() is None


async def test_command_trigger_flags_failures() -> None:
    watcher = StubCommandWatcher([CommandResult("c3", "make", 2, timedelta(minutes=10))])
    trigger = LongRunningCommandTrigger(watcher)

    notification = await trigger.check()

    assert notification is not None
    assert notification.title == "Command failed with exit code 2"
    assert notification.priority is Priority.HIGH
