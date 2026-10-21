"""Triggers: conditions Jarvis watches so it can speak first.

A :class:`Trigger` is a tiny object with one async method, ``check``, that
returns a :class:`Notification` when something deserves the user's attention.
The :class:`TriggerRunner` polls every trigger on the shared scheduler,
suppresses repeats per dedupe key and hands what remains to a
:class:`NotificationSink`, which the orchestrator implements to speak and
the tray to show.

The concrete triggers depend only on the small source Protocols defined
here (calendar, battery, reminders, command watcher). The integrations
package implements those Protocols; this module never imports it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import IntEnum
from typing import Protocol

from jarvis.proactive.scheduler import Clock, JobCallback, Scheduler, SystemClock
from jarvis.utils.logging import get_logger

log = get_logger(__name__)

DEFAULT_POLL = timedelta(seconds=60)
DEFAULT_COOLDOWN = timedelta(minutes=15)
_SECONDS_PER_MINUTE = 60
URGENT_BATTERY_PERCENT = 10
"""At or below this level a battery warning is URGENT rather than NORMAL."""


class Priority(IntEnum):
    """How interruptive a notification may be; the sink decides what that means."""

    LOW = 0
    NORMAL = 1
    HIGH = 2
    URGENT = 3


@dataclass(frozen=True, slots=True)
class NotificationAction:
    """A follow-up the user can pick, e.g. "snooze" or "open".

    Attributes:
        id: Stable identifier the sink reports back when chosen.
        label: Short text for a button or a spoken option.
    """

    id: str
    label: str


@dataclass(frozen=True, slots=True)
class Notification:
    """Something Jarvis wants to tell the user unprompted.

    Attributes:
        title: One line for the tray/overlay.
        spoken_text: Full sentence for TTS.
        priority: Interruption level.
        dedupe_key: Notifications with the same key within the runner's
            cooldown are published once. ``None`` disables suppression.
        actions: Optional follow-ups.
    """

    title: str
    spoken_text: str
    priority: Priority = Priority.NORMAL
    dedupe_key: str | None = None
    actions: tuple[NotificationAction, ...] = ()


class NotificationSink(Protocol):
    """Where published notifications go (speech, tray, dashboard)."""

    async def notify(self, notification: Notification) -> None:
        """Deliver ``notification`` to the user."""


class Trigger(Protocol):
    """A watched condition."""

    @property
    def name(self) -> str:
        """Stable label for logs and job names."""

    async def check(self) -> Notification | None:
        """Inspect the world once; return a notification or ``None``."""


@dataclass(slots=True)
class _TriggerState:
    trigger: Trigger
    job_id: str | None = None
    failures: int = 0


class TriggerRunner:
    """Polls triggers on the scheduler and publishes de-duplicated notifications.

    Args:
        scheduler: Shared scheduler; each trigger becomes one interval job.
        sink: Receives notifications that pass the cooldown.
        clock: Time source for the cooldown; share it with the scheduler.
        poll: Default polling period per trigger.
        cooldown: Minimum gap between two notifications with the same key.
    """

    def __init__(
        self,
        scheduler: Scheduler,
        sink: NotificationSink,
        *,
        clock: Clock | None = None,
        poll: timedelta = DEFAULT_POLL,
        cooldown: timedelta = DEFAULT_COOLDOWN,
    ) -> None:
        self._scheduler = scheduler
        self._sink = sink
        self._clock: Clock = clock or SystemClock()
        self._poll = poll
        self._cooldown = cooldown
        self._triggers: dict[str, _TriggerState] = {}
        self._last_published: dict[str, datetime] = {}
        self._published = 0
        self._suppressed = 0

    @property
    def triggers(self) -> list[Trigger]:
        """Registered triggers, in registration order."""
        return [state.trigger for state in self._triggers.values()]

    @property
    def published(self) -> int:
        """Notifications handed to the sink so far."""
        return self._published

    @property
    def suppressed(self) -> int:
        """Notifications dropped by the cooldown so far."""
        return self._suppressed

    def failures(self, name: str) -> int:
        """How many times trigger ``name`` raised from ``check``."""
        return self._triggers[name].failures

    def add(self, trigger: Trigger, *, poll: timedelta | None = None) -> str:
        """Register ``trigger`` and start polling it every ``poll`` (default: runner's).

        Raises:
            ValueError: If a trigger with the same name is already registered.
        """
        if trigger.name in self._triggers:
            raise ValueError(f"trigger '{trigger.name}' is already registered")
        state = _TriggerState(trigger)
        state.job_id = self._scheduler.add_interval(
            f"trigger:{trigger.name}",
            poll or self._poll,
            self._job_for(trigger.name),
            run_immediately=True,
        )
        self._triggers[trigger.name] = state
        return state.job_id

    def _job_for(self, name: str) -> JobCallback:
        async def run() -> None:
            await self.poll(name)

        return run

    def remove(self, name: str) -> bool:
        """Stop polling trigger ``name``; False if it was not registered."""
        state = self._triggers.pop(name, None)
        if state is None:
            return False
        if state.job_id is not None:
            self._scheduler.cancel(state.job_id)
        return True

    async def poll(self, name: str) -> Notification | None:
        """Check one trigger now and publish its result through the cooldown.

        Exceptions from ``check`` are logged with the trigger name, counted
        and not re-raised so the scheduler keeps polling the others.

        Returns:
            The notification that was published, or ``None``.
        """
        state = self._triggers[name]
        try:
            notification = await state.trigger.check()
        except Exception:
            state.failures += 1
            log.exception("proactive.trigger.failed", trigger=name, failures=state.failures)
            return None
        if notification is None or self._is_suppressed(notification):
            return None
        await self._sink.notify(notification)
        self._published += 1
        log.info("proactive.notification", trigger=name, title=notification.title)
        return notification

    def _is_suppressed(self, notification: Notification) -> bool:
        key = notification.dedupe_key
        if key is None:
            return False
        now = self._clock.now()
        last = self._last_published.get(key)
        if last is not None and now - last < self._cooldown:
            self._suppressed += 1
            log.debug("proactive.notification.suppressed", key=key)
            return True
        self._last_published[key] = now
        return False


# ---------------------------------------------------------------------------
# Source Protocols and the concrete triggers built on them
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CalendarEvent:
    """The slice of a calendar entry the trigger needs."""

    id: str
    title: str
    start: datetime
    location: str | None = None


class CalendarSource(Protocol):
    """Read access to upcoming calendar events."""

    async def upcoming_events(self, within: timedelta) -> Sequence[CalendarEvent]:
        """Events starting between now and ``now + within``, earliest first."""


class UpcomingMeetingTrigger:
    """Warns ``lead`` before the next calendar event.

    Args:
        source: Calendar read access.
        lead: How far ahead to look; the default matches the Phase 2 brief.
        clock: Time source; share it with the scheduler.
    """

    name = "upcoming_meeting"

    def __init__(
        self,
        source: CalendarSource,
        *,
        lead: timedelta = timedelta(minutes=10),
        clock: Clock | None = None,
    ) -> None:
        self._source = source
        self._lead = lead
        self._clock: Clock = clock or SystemClock()

    async def check(self) -> Notification | None:
        """Notify about the earliest event that has not started yet."""
        now = self._clock.now()
        events = await self._source.upcoming_events(self._lead)
        future = [event for event in events if event.start > now]
        if not future:
            return None
        event = min(future, key=lambda item: item.start)
        return self._describe(event, event.start - now)

    @staticmethod
    def _describe(event: CalendarEvent, until: timedelta) -> Notification:
        minutes = max(1, round(until.total_seconds() / _SECONDS_PER_MINUTE))
        where = f" in {event.location}" if event.location else ""
        return Notification(
            title=f"{event.title} in {minutes} min",
            spoken_text=f"Heads up: {event.title} starts in {minutes} minutes{where}.",
            priority=Priority.HIGH,
            dedupe_key=f"meeting:{event.id}:{event.start.isoformat()}",
            actions=(NotificationAction("snooze", "Remind me in 5 minutes"),),
        )


@dataclass(frozen=True, slots=True)
class BatteryStatus:
    """Battery reading; ``None`` from the source means no battery present."""

    percent: int
    plugged_in: bool


class BatterySource(Protocol):
    """Read access to the battery."""

    async def status(self) -> BatteryStatus | None:
        """Current reading, or ``None`` on a machine without a battery."""


class LowBatteryTrigger:
    """Warns once per threshold as the battery drains on battery power.

    Thresholds are re-armed when the charger is plugged in or the level
    climbs back above the highest threshold, so a laptop that bounces
    around 20 % is not nagged every poll.

    Args:
        source: Battery read access.
        thresholds: Percent levels that each produce one warning, any order.
    """

    name = "low_battery"

    def __init__(self, source: BatterySource, *, thresholds: Sequence[int] = (20, 10, 5)) -> None:
        if not thresholds:
            raise ValueError("at least one battery threshold is required")
        self._source = source
        self._thresholds = tuple(sorted(thresholds, reverse=True))
        self._warned: set[int] = set()

    async def check(self) -> Notification | None:
        """Warn for the lowest newly crossed threshold, if any."""
        status = await self._source.status()
        if status is None:
            return None
        if status.plugged_in or status.percent > self._thresholds[0]:
            self._warned.clear()
            return None
        crossed = [level for level in self._thresholds if status.percent <= level]
        pending = [level for level in crossed if level not in self._warned]
        if not pending:
            return None
        self._warned.update(crossed)
        return self._describe(status.percent, min(pending))

    @staticmethod
    def _describe(percent: int, level: int) -> Notification:
        priority = Priority.URGENT if level <= URGENT_BATTERY_PERCENT else Priority.NORMAL
        return Notification(
            title=f"Battery at {percent}%",
            spoken_text=f"Battery is at {percent} percent. You might want to plug in.",
            priority=priority,
            dedupe_key=f"battery:{level}",
        )


@dataclass(frozen=True, slots=True)
class Reminder:
    """A reminder the user asked for earlier."""

    id: str
    text: str
    due: datetime


class ReminderSource(Protocol):
    """Reminder storage with pop semantics so each reminder is delivered once."""

    async def pop_due(self, now: datetime) -> Sequence[Reminder]:
        """Remove and return every reminder with ``due <= now``."""


class DueReminderTrigger:
    """Speaks reminders as they come due, several at once if they pile up."""

    name = "due_reminder"

    def __init__(self, source: ReminderSource, *, clock: Clock | None = None) -> None:
        self._source = source
        self._clock: Clock = clock or SystemClock()

    async def check(self) -> Notification | None:
        """Pop everything due and fold it into one notification."""
        due = list(await self._source.pop_due(self._clock.now()))
        if not due:
            return None
        texts = [reminder.text for reminder in due]
        spoken = "Reminder: " + ". Also: ".join(texts) + "."
        title = texts[0] if len(texts) == 1 else f"{len(texts)} reminders"
        return Notification(
            title=title,
            spoken_text=spoken,
            priority=Priority.HIGH,
            dedupe_key="reminder:" + ",".join(reminder.id for reminder in due),
        )


@dataclass(frozen=True, slots=True)
class CommandResult:
    """Outcome of a shell command the user started through Jarvis."""

    id: str
    command: str
    exit_code: int
    duration: timedelta


class CommandWatcher(Protocol):
    """Tracks background commands and hands over the ones that finished."""

    async def pop_finished(self) -> Sequence[CommandResult]:
        """Remove and return commands that finished since the previous call."""


class LongRunningCommandTrigger:
    """Announces finished commands the user has probably stopped watching.

    Args:
        watcher: Source of finished commands.
        min_duration: Commands shorter than this are not announced; the user
            saw them finish.
    """

    name = "long_running_command"

    def __init__(
        self, watcher: CommandWatcher, *, min_duration: timedelta = timedelta(seconds=30)
    ) -> None:
        self._watcher = watcher
        self._min_duration = min_duration

    async def check(self) -> Notification | None:
        """Report the first long-enough finished command."""
        finished = await self._watcher.pop_finished()
        long_ones = [result for result in finished if result.duration >= self._min_duration]
        if not long_ones:
            return None
        return self._describe(long_ones[0])

    @staticmethod
    def _describe(result: CommandResult) -> Notification:
        outcome = (
            "finished" if result.exit_code == 0 else f"failed with exit code {result.exit_code}"
        )
        minutes = round(result.duration.total_seconds() / _SECONDS_PER_MINUTE)
        return Notification(
            title=f"Command {outcome}",
            spoken_text=f"Your command {result.command} {outcome} after about {minutes} minutes.",
            priority=Priority.NORMAL if result.exit_code == 0 else Priority.HIGH,
            dedupe_key=f"command:{result.id}",
        )


# Design notes:
# - One interval job per trigger, not one job polling all: a slow calendar
#   API must not delay the battery check, and each trigger can have its own
#   period (battery every minute, calendar every five).
# - Exceptions from check() are caught in the runner, not left to the
#   scheduler, so the log line carries the trigger name and the runner can
#   expose a failure count to the dashboard. They are never silent.
# - Dedupe is time-based (cooldown per key) rather than "once ever" because
#   the same condition legitimately recurs: a meeting notification has the
#   event start in its key, a battery warning its threshold.
# - The source Protocols are the minimum each trigger needs, not a full
#   calendar/battery API, so the integrations layer can implement them with
#   a thin adapter and tests can fake them in a few lines.
# - Assumption: pop semantics for reminders and commands (the source removes
#   what it returns) so a notification is never delivered twice even if the
#   runner restarts.
