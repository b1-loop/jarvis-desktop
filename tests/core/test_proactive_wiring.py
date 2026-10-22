import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from jarvis.agent.confirmation import ConfirmationPolicy
from jarvis.agent.registry import ToolRegistry
from jarvis.agent.tool import ToolContext
from jarvis.config import Settings
from jarvis.core.events import EventBus
from jarvis.core.proactive_wiring import (
    ROUTINE_SUCCESS_SPOKEN,
    IdleAnnouncer,
    MemoryReminderSource,
    ProactiveSources,
    ProbeBatterySource,
    RegistryToolInvoker,
    RoutineAwareTurns,
    build_trigger_runner,
    load_routine_registry,
)
from jarvis.core.state_machine import AssistantState, StateMachine
from jarvis.proactive.routines import UnknownToolError
from jarvis.proactive.scheduler import Scheduler
from jarvis.proactive.triggers import BatteryStatus, CalendarEvent, Notification
from jarvis.tools._jobs import BackgroundJobs
from tests.agent.llm_fakes import LookupContactTool, SentenceCollector
from tests.core.fakes import FakeAgent, RecordingOutput
from tests.core.voice_fakes import eventually

MakeSettings = Callable[..., Settings]

ROUTINE_YAML = """
name: greeting
trigger: { kind: phrase, phrase: good morning }
intro: Good morning.
steps:
  - tool: lookup_contact
    args: { name: Anna }
    speak: "Found: {result}"
  - tool: no_such_tool
"""

SILENT_ROUTINE_YAML = """
name: quiet
trigger: { kind: phrase, phrase: lights out }
steps:
  - tool: lookup_contact
    args: { name: Bo }
"""


@dataclass(frozen=True)
class Reading:
    percent: float
    plugged_in: bool


@dataclass
class Probe:
    reading: Reading | None

    def read(self) -> Reading | None:
        return self.reading


async def test_battery_source_reads_the_probe_off_the_loop() -> None:
    assert await ProbeBatterySource(Probe(Reading(17.6, False))).status() == BatteryStatus(
        18, False
    )
    assert await ProbeBatterySource(Probe(None)).status() is None


@dataclass(frozen=True)
class StoredReminder:
    id: int
    text: str
    due_at: datetime


@dataclass
class ReminderStore:
    due: list[StoredReminder] = field(default_factory=list)

    async def pop_due_reminders(self, now: datetime | None = None) -> list[StoredReminder]:
        due, self.due = self.due, []
        return due


async def test_reminders_come_from_memory_once() -> None:
    due = datetime(2026, 10, 22, 9, 0).astimezone()
    store = ReminderStore([StoredReminder(7, "Call Anna", due)])
    source = MemoryReminderSource(store)

    first = await source.pop_due(due)
    assert [(r.id, r.text, r.due) for r in first] == [("7", "Call Anna", due)]
    assert await source.pop_due(due) == []


def tool_context(
    make_settings: MakeSettings,
) -> tuple[ToolRegistry, ConfirmationPolicy, ToolContext]:
    settings = make_settings()
    registry = ToolRegistry()
    registry.register(LookupContactTool())

    async def never(_prompt: str) -> bool:
        return False

    context = ToolContext(settings=settings, bus=EventBus(), ask_confirmation=never)
    return registry, ConfirmationPolicy.from_settings(settings), context


async def test_invoker_runs_tools_through_the_policy(make_settings: MakeSettings) -> None:
    invoker = RegistryToolInvoker(*tool_context(make_settings))

    assert "Anna" in await invoker.invoke("lookup_contact", {"name": "Anna"})
    with pytest.raises(UnknownToolError):
        await invoker.invoke("no_such_tool", {})


async def test_a_routine_phrase_is_answered_by_the_routine(
    make_settings: MakeSettings, tmp_path: Path
) -> None:
    (tmp_path / "greeting.yaml").write_text(ROUTINE_YAML, encoding="utf-8")
    (tmp_path / "quiet.yaml").write_text(SILENT_ROUTINE_YAML, encoding="utf-8")
    agent = FakeAgent()
    turns = RoutineAwareTurns(
        agent, load_routine_registry(tmp_path), RegistryToolInvoker(*tool_context(make_settings))
    )
    spoken = SentenceCollector()

    report = await turns.respond("Good morning, Jarvis!", spoken)
    quiet = await turns.respond("lights out", SentenceCollector())
    other = await turns.respond("What time is it?", SentenceCollector())

    assert spoken.sentences[0] == "Good morning."
    assert spoken.sentences[1].startswith("Found:")
    assert "no_such_tool" in spoken.sentences[2]
    assert report.error == "routine greeting did not complete"
    assert quiet.reply == ROUTINE_SUCCESS_SPOKEN
    assert agent.requests == ["What time is it?"]
    assert other.reply == "Certainly, sir."


async def test_announcements_wait_for_idle_and_notify_the_system() -> None:
    bus = EventBus()
    state = StateMachine(bus)
    output = RecordingOutput()
    shown: list[tuple[str, str]] = []

    async def notify(title: str, body: str) -> None:
        shown.append((title, body))

    announcer = IdleAnnouncer(output, bus, notify)
    state.transition(AssistantState.LISTENING, "busy")
    pending = asyncio.create_task(
        announcer.notify(
            Notification(title="Battery at 9%", spoken_text="Battery is at 9 percent.")
        )
    )
    await asyncio.sleep(0.02)
    assert output.said == []  # not while the user is talking

    state.transition(AssistantState.IDLE, "done")
    await asyncio.wait_for(pending, 2)
    announcer.close()

    assert output.said == ["Battery is at 9 percent."]
    assert output.flushes == 1
    assert shown == [("Battery at 9%", "Battery is at 9 percent.")]


@dataclass
class Calendar:
    async def upcoming_events(self, within: timedelta) -> list[CalendarEvent]:
        return []


@dataclass
class Reminders:
    async def pop_due(self, now: datetime) -> list[object]:
        return []


async def test_trigger_runner_polls_the_standard_triggers() -> None:
    scheduler = Scheduler()
    output = RecordingOutput()

    class Sink:
        async def notify(self, notification: Notification) -> None:
            output.said.append(notification.spoken_text)

    with_battery = ProactiveSources(
        battery=ProbeBatterySource(Probe(Reading(4, False))),
        reminders=Reminders(),  # type: ignore[arg-type]
        commands=BackgroundJobs(),
        calendar=Calendar(),
    )
    runner = build_trigger_runner(scheduler, Sink(), with_battery, poll=timedelta(hours=1))
    without_battery = build_trigger_runner(
        Scheduler(),
        Sink(),
        ProactiveSources(None, Reminders(), BackgroundJobs(), Calendar()),  # type: ignore[arg-type]
    )

    assert [t.name for t in runner.triggers] == [
        "low_battery",
        "due_reminder",
        "long_running_command",
        "upcoming_meeting",
    ]
    assert "low_battery" not in [t.name for t in without_battery.triggers]
    async with scheduler:
        await eventually(lambda: output.said)
    assert "4 percent" in output.said[0]
