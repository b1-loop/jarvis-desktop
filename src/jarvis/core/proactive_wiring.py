"""Adapters and builders that connect the proactive module to the running assistant.

``jarvis.proactive`` defines a scheduler, triggers that watch the world and
YAML routines, all against small Protocols. This module is the composition
code that satisfies those Protocols with the real services:

* battery readings from psutil (the battery tool's probe) off the loop;
* due reminders from the ``MemoryManager``;
* finished background commands from the shared ``BackgroundJobs``;
* calendar events through the calendar tools' ``CalendarTriggerSource``;
* notifications spoken (voice) or printed (text) through the front end's
  sink once the assistant is idle, plus a system notification;
* routines: phrase routines answered inside a turn instead of the agent,
  timed routines run by the scheduler, their tool steps going through the
  central confirmation policy like any agent tool call.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Final, Protocol

from jarvis.agent.chaining import TurnReport
from jarvis.agent.confirmation import ConfirmationPolicy
from jarvis.agent.loop import SentenceSink
from jarvis.agent.registry import ToolRegistry
from jarvis.agent.registry import UnknownToolError as RegistryUnknownToolError
from jarvis.agent.tool import ToolContext
from jarvis.core.events import EventBus, StateChanged
from jarvis.core.orchestrator import OutputSink, TurnRunner
from jarvis.core.state_machine import AssistantState
from jarvis.proactive import triggers
from jarvis.proactive.routines import (
    RoutineRegistry,
    RoutineRunner,
    UnknownToolError,
    load_routines,
)
from jarvis.proactive.scheduler import Clock as SchedulerClock
from jarvis.proactive.scheduler import Scheduler
from jarvis.proactive.scheduler import SystemClock as SchedulerSystemClock
from jarvis.utils.logging import get_logger

log = get_logger(__name__)

ROUTINE_SUCCESS_SPOKEN: Final = "Done."


class BatteryReadingLike(Protocol):
    """``tools.battery_status.BatteryReading``: what the psutil probe returns."""

    @property
    def percent(self) -> float:
        """Charge in percent."""

    @property
    def plugged_in(self) -> bool:
        """Mains power connected."""


class BlockingBatteryProbe(Protocol):
    """``tools.battery_status.BatteryProbe``: a blocking read."""

    def read(self) -> BatteryReadingLike | None:
        """Current reading, ``None`` without a battery."""


class ProbeBatterySource:
    """``triggers.BatterySource`` over the battery tool's blocking probe."""

    def __init__(self, probe: BlockingBatteryProbe) -> None:
        self._probe = probe

    async def status(self) -> triggers.BatteryStatus | None:
        """Read the battery in a worker thread (psutil may touch WMI)."""
        reading = await asyncio.to_thread(self._probe.read)
        if reading is None:
            return None
        return triggers.BatteryStatus(round(reading.percent), reading.plugged_in)


class StoredReminder(Protocol):
    """``memory.models.Reminder``, the fields the trigger needs."""

    @property
    def id(self) -> int:
        """Row id."""

    @property
    def text(self) -> str:
        """What to say."""

    @property
    def due_at(self) -> datetime:
        """When to say it."""


class ReminderStore(Protocol):
    """The slice of ``MemoryManager`` that hands out due reminders once."""

    async def pop_due_reminders(self, now: datetime | None = None) -> Sequence[StoredReminder]:
        """Mark due reminders fired and return them."""


class MemoryReminderSource:
    """``triggers.ReminderSource`` over the memory store."""

    def __init__(self, store: ReminderStore) -> None:
        self._store = store

    async def pop_due(self, now: datetime) -> list[triggers.Reminder]:
        """Due reminders in the trigger's shape; each is returned only once."""
        due = await self._store.pop_due_reminders(now)
        return [triggers.Reminder(str(item.id), item.text, item.due_at) for item in due]


class RegistryToolInvoker:
    """``routines.ToolInvoker`` over the tool registry and the confirmation policy.

    Routine steps run exactly like agent tool calls: validated, confirmed
    when the policy requires it, and reported by their spoken result.
    """

    def __init__(
        self, registry: ToolRegistry, policy: ConfirmationPolicy, context: ToolContext
    ) -> None:
        self._registry = registry
        self._policy = policy
        self._context = context

    async def invoke(self, name: str, args: Mapping[str, Any]) -> str:
        """Run ``name`` and return its spoken result.

        Raises:
            UnknownToolError: If no tool has that name.
        """
        try:
            tool = self._registry.get(name)
        except RegistryUnknownToolError as error:
            raise UnknownToolError(name) from error
        result = await self._policy.execute_with_policy(tool, args, self._context)
        return result.spoken


class _SinkSpeaker:
    """``routines.Speaker`` over a turn's sentence sink."""

    def __init__(self, sink: SentenceSink) -> None:
        self._sink = sink
        self.said: list[str] = []

    async def say(self, text: str) -> None:
        self.said.append(text)
        await self._sink(text)


class RoutineAwareTurns:
    """``TurnRunner`` that answers a routine's phrase with the routine, else asks the agent.

    Args:
        agent: The agent that answers everything else.
        routines: Phrase lookup.
        invoker: Runs the routine's tool steps.
    """

    def __init__(
        self, agent: TurnRunner, routines: RoutineRegistry, invoker: RegistryToolInvoker
    ) -> None:
        self._agent = agent
        self._routines = routines
        self._invoker = invoker

    async def respond(self, user_text: str, sink: SentenceSink) -> TurnReport:
        """Run the matching routine or delegate to the agent."""
        routine = self._routines.by_phrase(user_text)
        if routine is None:
            return await self._agent.respond(user_text, sink)
        log.info("proactive.routine.phrase", routine=routine.name)
        speaker = _SinkSpeaker(sink)
        report = await RoutineRunner(self._invoker, speaker).run(routine)
        if not speaker.said:
            await speaker.say(ROUTINE_SUCCESS_SPOKEN)
        error = None if report.ok else f"routine {routine.name} did not complete"
        return TurnReport(user_text=user_text, reply=" ".join(speaker.said), error=error)


Notifier = Callable[[str, str], Awaitable[None]]
"""Shows a system notification (``Platform.notify``)."""


class IdleAnnouncer:
    """Speaks unprompted text through the front end, but only between turns.

    Implements both ``triggers.NotificationSink`` and ``routines.Speaker``.
    Speaking in the middle of a turn would interleave with the answer (and,
    in voice mode, be heard by the microphone while listening), so every
    announcement waits for IDLE and announcements are serialised.

    Args:
        sink: The front end's output (TTS in voice mode, stdout in text mode).
        bus: Source of ``StateChanged``.
        notify: System notification for each proactive notification.
    """

    def __init__(self, sink: OutputSink, bus: EventBus, notify: Notifier) -> None:
        self._sink = sink
        self._notify = notify
        self._idle = asyncio.Event()
        self._idle.set()
        self._lock = asyncio.Lock()
        self._unsubscribe = bus.subscribe_callback(StateChanged, self._on_state)

    def close(self) -> None:
        """Stop following the state machine."""
        self._unsubscribe()

    async def notify(self, notification: triggers.Notification) -> None:
        """Say the notification and show it as a system notification."""
        await self.say(notification.spoken_text)
        await self._notify(notification.title, notification.spoken_text)

    async def say(self, text: str) -> None:
        """Speak ``text`` once the assistant is idle, then wait until it was delivered."""
        async with self._lock:
            await self._idle.wait()
            await self._sink.say(text)
            await self._sink.flush()

    def _on_state(self, event: StateChanged) -> None:
        if event.new is AssistantState.IDLE:
            self._idle.set()
        else:
            self._idle.clear()


@dataclass(frozen=True, slots=True)
class ProactiveSources:
    """What the standard triggers watch.

    Attributes:
        battery: Battery readings; ``None`` skips the battery trigger.
        reminders: Due reminders.
        commands: Finished background commands (``ToolServices.jobs``).
        calendar: Upcoming calendar events.
    """

    battery: triggers.BatterySource | None
    reminders: triggers.ReminderSource
    commands: triggers.CommandWatcher
    calendar: triggers.CalendarSource


def build_trigger_runner(
    scheduler: Scheduler,
    sink: triggers.NotificationSink,
    sources: ProactiveSources,
    *,
    clock: SchedulerClock | None = None,
    poll: timedelta = triggers.DEFAULT_POLL,
) -> triggers.TriggerRunner:
    """A runner polling the standard triggers on ``scheduler`` every ``poll``."""
    clock = clock or SchedulerSystemClock()
    runner = triggers.TriggerRunner(scheduler, sink, clock=clock, poll=poll)
    if sources.battery is not None:
        runner.add(triggers.LowBatteryTrigger(sources.battery))
    runner.add(triggers.DueReminderTrigger(sources.reminders, clock=clock))
    runner.add(triggers.LongRunningCommandTrigger(sources.commands))
    runner.add(triggers.UpcomingMeetingTrigger(sources.calendar, clock=clock))
    return runner


def load_routine_registry(directory: Path) -> RoutineRegistry:
    """Every routine under ``directory`` (``Settings.routines_dir``)."""
    return RoutineRegistry(load_routines(directory))


# Design notes:
# - The announcer waits for IDLE instead of interrupting: Phase 3 asks for
#   "upcoming meeting, low battery" style notices, none of which is worth
#   cutting the user's own request short. An URGENT priority could pre-empt
#   in future; today the priority only reaches the system notification.
# - Phrase routines run inside the orchestrator's turn task, so they get
#   the same states, barge-in and confirmation as an agent answer. They are
#   matched before the agent so "good morning" is deterministic and costs
#   no model round trip.
# - A routine step's output is the tool's spoken result, the same sentence
#   the agent would have heard; the routine YAML decides what to say with it.
