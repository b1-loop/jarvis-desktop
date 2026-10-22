import asyncio
import dataclasses
import re
from datetime import UTC, datetime, timedelta

import pytest

from jarvis.core.events import (
    AssistantSaid,
    ConfirmationRequested,
    ConfirmationResolved,
    EventBus,
    LatencySample,
    StateChanged,
    ToolCallFinished,
    ToolCallStarted,
    TranscriptReady,
    WakeWordHeard,
)
from jarvis.core.state_machine import AssistantState
from jarvis.memory.models import Fact, Note, Reminder, Todo
from jarvis.ui.view_model import (
    EMPTY_SECTION,
    NO_VALUE,
    OUTCOME_COLOURS,
    STATE_STYLES,
    Confirmation,
    DashboardModel,
    MemoryStatus,
    MemoryView,
    ModelLimits,
    Speaker,
    ToolOutcome,
    format_clock,
    format_duration,
    state_style,
    tray_tooltip,
    truncate,
)
from jarvis.utils.timing import LatencyTracker

NOON = datetime(2026, 10, 22, 12, 0, 0, tzinfo=UTC).timestamp()
HEX_COLOUR = re.compile(r"^#[0-9A-F]{6}$")


class FakeMemoryInspector:
    """In-memory ``MemoryInspector`` with fixed rows; counts reads."""

    def __init__(self) -> None:
        stamp = datetime(2026, 10, 22, 9, 0, tzinfo=UTC)
        self.facts = [
            _fact("profile.name", "Ada", stamp),
            _fact("profile.preference.units", "metric", stamp),
            _fact("home.city", "Sofia", stamp, source="llm", confidence=0.7),
            _fact("work.editor", "VS Code", stamp + timedelta(hours=1)),
        ]
        self.notes = [Note(id=i, title="", content=f"n{i}", created_at=stamp) for i in range(3)]
        self.todos = [Todo(id=1, text="Ship   step\n13", done=False, created_at=stamp)]
        self.reminders = [
            Reminder(id=1, text="Stand-up", due_at=stamp, fired=False, created_at=stamp)
        ]
        self.reads = 0
        self.gate: asyncio.Event | None = None

    async def list_facts(self, prefix: str = "") -> list[Fact]:
        self.reads += 1
        if self.gate is not None:
            await self.gate.wait()
        return [fact for fact in self.facts if fact.key.startswith(prefix)]

    async def list_notes(self, limit: int = 50) -> list[Note]:
        return self.notes[:limit]

    async def list_todos(self, *, include_done: bool = False) -> list[Todo]:
        return self.todos

    async def list_reminders(self, *, include_fired: bool = False) -> list[Reminder]:
        return self.reminders


class BrokenMemoryInspector(FakeMemoryInspector):
    async def list_facts(self, prefix: str = "") -> list[Fact]:
        raise RuntimeError("database is locked")


def _fact(
    key: str, value: str, at: datetime, *, source: str = "user", confidence: float = 1.0
) -> Fact:
    return Fact(key, value, source, confidence, created_at=at, updated_at=at)


@pytest.fixture
def bus() -> EventBus:
    return EventBus()


@pytest.fixture
def model(bus: EventBus) -> DashboardModel:
    model = DashboardModel(bus, tz=UTC, now=lambda: NOON)
    model.attach()
    return model


def _state(old: AssistantState, new: AssistantState, reason: str = "test") -> StateChanged:
    return StateChanged(old=old, new=new, reason=reason, timestamp=NOON)


# -- presentation helpers -------------------------------------------------------


def test_every_state_has_a_label_and_a_hex_colour() -> None:
    assert set(STATE_STYLES) == set(AssistantState)
    for state in AssistantState:
        style = state_style(state)
        assert HEX_COLOUR.match(style.colour)
        assert style.label == state.value.title()
    assert len({style.colour for style in STATE_STYLES.values()}) == len(AssistantState)


def test_every_outcome_has_a_colour() -> None:
    assert set(OUTCOME_COLOURS) == set(ToolOutcome)


def test_truncate_flattens_whitespace_and_marks_the_cut() -> None:
    assert truncate("a\n b\t c", 10) == "a b c"
    assert truncate("abcdefghij", 5) == "abcd…"
    assert len(truncate("x" * 50, 8)) == 8


def test_durations_switch_to_seconds_at_one_second() -> None:
    assert format_duration(12.4) == "12 ms"
    assert format_duration(999.4) == "999 ms"
    assert format_duration(1250) == "1.2 s"


def test_clock_uses_the_given_zone() -> None:
    assert format_clock(NOON, UTC) == "12:00:00"


def test_tray_tooltip_mentions_state_and_mute() -> None:
    assert tray_tooltip(AssistantState.THINKING, microphone_muted=False) == "Jarvis - Thinking"
    assert tray_tooltip(AssistantState.IDLE, microphone_muted=True).endswith("(microphone muted)")


# -- state and counters -------------------------------------------------------------


def test_initial_snapshot_is_idle_and_empty(model: DashboardModel) -> None:
    snapshot = model.snapshot()

    assert snapshot.state is AssistantState.IDLE
    assert snapshot.state_since == "12:00:00"
    assert snapshot.transcript == ()
    assert snapshot.tool_calls == ()
    assert snapshot.latency == ()
    assert snapshot.memory.status is MemoryStatus.UNAVAILABLE


def test_state_change_updates_header_and_notifies(bus: EventBus, model: DashboardModel) -> None:
    calls: list[AssistantState] = []
    model.add_listener(lambda: calls.append(model.snapshot().state))

    bus.publish(_state(AssistantState.IDLE, AssistantState.LISTENING, "wake word"))

    snapshot = model.snapshot()
    assert calls == [AssistantState.LISTENING]
    assert snapshot.style == STATE_STYLES[AssistantState.LISTENING]
    assert snapshot.header_detail() == "since 12:00:00 (wake word)"
    assert snapshot.revisions.header == 1


def test_wake_words_and_events_are_counted(bus: EventBus, model: DashboardModel) -> None:
    bus.publish(WakeWordHeard(score=0.9))
    bus.publish(WakeWordHeard())
    bus.publish(AssistantSaid(text="hi"))

    snapshot = model.snapshot()
    assert snapshot.wake_words == 2
    assert snapshot.events_seen == 3
    assert snapshot.counters_line() == "wake words 2  |  events 3"


def test_removed_listener_is_not_called(bus: EventBus, model: DashboardModel) -> None:
    calls: list[int] = []
    remove = model.add_listener(lambda: calls.append(1))
    remove()
    remove()

    bus.publish(WakeWordHeard())

    assert calls == []


def test_detach_stops_updates_and_attach_is_idempotent(
    bus: EventBus, model: DashboardModel
) -> None:
    model.attach()
    bus.publish(WakeWordHeard())
    assert model.snapshot().events_seen == 1

    model.detach()
    bus.publish(WakeWordHeard())

    assert not model.attached
    assert model.snapshot().events_seen == 1


def test_snapshots_are_immutable(model: DashboardModel) -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        model.snapshot().wake_words = 3  # type: ignore[misc]


# -- transcript ---------------------------------------------------------------------


def test_transcript_records_both_speakers_in_order(bus: EventBus, model: DashboardModel) -> None:
    bus.publish(TranscriptReady(text="what time is it", latency_ms=120, timestamp=NOON))
    bus.publish(AssistantSaid(text="It is noon.", timestamp=NOON + 1))

    lines = model.snapshot().transcript
    assert [(line.speaker, line.text, line.time) for line in lines] == [
        (Speaker.USER, "what time is it", "12:00:00"),
        (Speaker.ASSISTANT, "It is noon.", "12:00:01"),
    ]
    assert model.snapshot().revisions.transcript == 2


def test_transcript_is_capped_and_truncated(bus: EventBus) -> None:
    model = DashboardModel(bus, limits=ModelLimits(transcript_lines=2, line_chars=6), tz=UTC)
    model.attach()
    for text in ("one", "two", "three\nthree"):
        bus.publish(AssistantSaid(text=text))

    assert [line.text for line in model.snapshot().transcript] == ["two", "three…"]


# -- tool calls -----------------------------------------------------------------------


def test_successful_tool_call_is_one_row(bus: EventBus, model: DashboardModel) -> None:
    bus.publish(ToolCallStarted(name="open_app", args_summary='{"name": "code"}', timestamp=NOON))
    running = model.snapshot().tool_calls
    bus.publish(
        ToolCallFinished(name="open_app", args_summary='{"name": "code"}', duration_ms=42, ok=True)
    )

    assert running[0].outcome is ToolOutcome.RUNNING
    (row,) = model.snapshot().tool_calls
    assert (row.time, row.name, row.args, row.duration) == (
        "12:00:00",
        "open_app",
        '{"name": "code"}',
        "42 ms",
    )
    assert row.outcome is ToolOutcome.OK
    assert row.confirmation is Confirmation.NOT_ASKED
    assert row.error == ""


def test_failed_tool_call_shows_the_error(bus: EventBus, model: DashboardModel) -> None:
    bus.publish(ToolCallStarted(name="shell", args_summary="{}"))
    bus.publish(
        ToolCallFinished(
            name="shell", args_summary="{}", duration_ms=1500, ok=False, error="exit\ncode 1"
        )
    )

    (row,) = model.snapshot().tool_calls
    assert row.outcome is ToolOutcome.ERROR
    assert row.duration == "1.5 s"
    assert row.error == "exit code 1"


def test_finish_without_start_still_produces_a_row(bus: EventBus, model: DashboardModel) -> None:
    bus.publish(ToolCallFinished(name="late", args_summary="{}", duration_ms=5, ok=True))

    (row,) = model.snapshot().tool_calls
    assert (row.name, row.outcome) == ("late", ToolOutcome.OK)


def test_approved_confirmation_and_run_share_a_row(bus: EventBus, model: DashboardModel) -> None:
    bus.publish(ConfirmationRequested(tool_name="send_email", prompt="Send it to Anna?"))
    waiting = model.snapshot().tool_calls[0]
    bus.publish(ConfirmationResolved(tool_name="send_email", approved=True))
    bus.publish(ToolCallStarted(name="send_email", args_summary='{"to": "anna"}'))
    bus.publish(
        ToolCallFinished(name="send_email", args_summary='{"to": "anna"}', duration_ms=9, ok=True)
    )

    assert waiting.outcome is ToolOutcome.AWAITING_CONFIRMATION
    assert waiting.confirmation is Confirmation.PENDING
    assert waiting.args == "Send it to Anna?"
    (row,) = model.snapshot().tool_calls
    assert row.confirmation is Confirmation.APPROVED
    assert row.outcome is ToolOutcome.OK
    assert row.args == '{"to": "anna"}'


def test_declined_confirmation_ends_the_row(bus: EventBus, model: DashboardModel) -> None:
    bus.publish(ConfirmationRequested(tool_name="delete", prompt="Delete?"))
    bus.publish(ConfirmationResolved(tool_name="delete", approved=False))

    (row,) = model.snapshot().tool_calls
    assert row.confirmation is Confirmation.DECLINED
    assert row.outcome is ToolOutcome.DECLINED
    assert row.duration == NO_VALUE


def test_unmatched_confirmation_answer_is_ignored(bus: EventBus, model: DashboardModel) -> None:
    bus.publish(ConfirmationResolved(tool_name="ghost", approved=True))

    assert model.snapshot().tool_calls == ()


def test_tool_log_is_capped(bus: EventBus) -> None:
    model = DashboardModel(bus, limits=ModelLimits(tool_calls=2))
    model.attach()
    for name in ("a", "b", "c"):
        bus.publish(ToolCallStarted(name=name, args_summary="{}"))

    assert [row.name for row in model.snapshot().tool_calls] == ["b", "c"]


def test_long_arguments_are_truncated(bus: EventBus) -> None:
    model = DashboardModel(bus, limits=ModelLimits(args_chars=5))
    model.attach()
    bus.publish(ToolCallStarted(name="t", args_summary="abcdefgh"))

    assert model.snapshot().tool_calls[0].args == "abcd…"


# -- latency --------------------------------------------------------------------------


def test_own_tracker_turns_samples_into_stats(bus: EventBus, model: DashboardModel) -> None:
    for ms in (100.0, 200.0, 300.0, 1400.0):
        bus.publish(LatencySample(stage="stt", ms=ms))

    (row,) = model.snapshot().latency
    assert (row.stage, row.count) == ("stt", 4)
    assert (row.mean, row.p50, row.p95, row.last) == ("500 ms", "200 ms", "1.4 s", "1.4 s")
    assert model.snapshot().revisions.latency == 4


def test_injected_tracker_is_read_not_fed(bus: EventBus) -> None:
    tracker = LatencyTracker()
    tracker.attach(bus)
    model = DashboardModel(bus, latency=tracker)
    model.attach()

    bus.publish(LatencySample(stage="llm", ms=50.0))

    assert model.snapshot().latency[0].count == 1


# -- memory -----------------------------------------------------------------------------


async def test_refresh_reads_and_formats_memory(bus: EventBus) -> None:
    inspector = FakeMemoryInspector()
    model = DashboardModel(bus, memory=inspector, tz=UTC, now=lambda: NOON)
    assert model.snapshot().memory.status is MemoryStatus.NOT_LOADED

    await model.refresh_memory()

    memory = model.snapshot().memory
    assert memory.status is MemoryStatus.READY
    assert memory.status_line() == "ready (refreshed 12:00:00)"
    assert memory.profile == ("Name: Ada", "Preferences: units: metric")
    assert memory.facts == ("work.editor: VS Code (user, 100%)", "home.city: Sofia (llm, 70%)")
    assert memory.notes_count == "3"
    assert memory.todos == ("Ship step 13",)
    assert memory.reminders == ("Thu 22 Oct 09:00  Stand-up",)


async def test_note_count_shows_the_cap(bus: EventBus) -> None:
    model = DashboardModel(bus, memory=FakeMemoryInspector(), limits=ModelLimits(notes_count_cap=2))

    await model.refresh_memory()

    assert model.snapshot().memory.notes_count == "2+"


async def test_memory_sections_have_placeholders(bus: EventBus) -> None:
    inspector = FakeMemoryInspector()
    inspector.facts, inspector.todos, inspector.reminders = [], [], []
    model = DashboardModel(bus, memory=inspector)

    await model.refresh_memory()

    sections = dict(model.snapshot().memory.sections())
    assert sections["Profile"] == (EMPTY_SECTION,)
    assert sections["Open todos"] == (EMPTY_SECTION,)
    assert sections["Notes"] == ("3 stored",)


async def test_failed_refresh_is_shown_not_raised(bus: EventBus) -> None:
    model = DashboardModel(bus, memory=BrokenMemoryInspector())

    await model.refresh_memory()

    memory = model.snapshot().memory
    assert memory.status is MemoryStatus.FAILED
    assert memory.status_line() == "failed: database is locked"
    assert memory.sections() == ()


async def test_refresh_without_inspector_is_a_no_op(model: DashboardModel) -> None:
    await model.refresh_memory()

    assert model.snapshot().memory == MemoryView(MemoryStatus.UNAVAILABLE)
    assert model.snapshot().memory.status_line() == "no memory inspector"


async def test_requested_refreshes_do_not_overlap(bus: EventBus) -> None:
    inspector = FakeMemoryInspector()
    inspector.gate = asyncio.Event()
    model = DashboardModel(bus, memory=inspector)

    model.request_memory_refresh()
    await asyncio.sleep(0)
    model.request_memory_refresh()
    assert model.snapshot().memory.status is MemoryStatus.LOADING
    inspector.gate.set()
    await model.close()

    assert inspector.reads == 1
    assert model.snapshot().memory.status is MemoryStatus.READY
