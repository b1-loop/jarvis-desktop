"""Toolkit-free view model shared by the dashboard window and the tray icon.

``DashboardModel`` subscribes to the event bus and folds every event into
the state a human wants to look at: what Jarvis is doing, what was said,
which tools ran and how long they took, how fast each pipeline stage is,
and what the memory holds. It produces immutable :class:`DashboardSnapshot`
values; a renderer (Tk, a test, a future web view) only ever draws a
snapshot and never reads the model's internals.

Every presentation decision lives here as plain data: truncation, time and
duration formats, the colour and label of each assistant state, the tray
tooltip. That keeps the toolkit modules thin enough that the logic worth
testing is tested without a display.
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, tzinfo
from enum import StrEnum
from typing import Final, Protocol

from jarvis.core.events import (
    AssistantSaid,
    ConfirmationRequested,
    ConfirmationResolved,
    Event,
    EventBus,
    LatencySample,
    StateChanged,
    ToolCallFinished,
    ToolCallStarted,
    TranscriptReady,
    Unsubscribe,
    WakeWordHeard,
)
from jarvis.core.state_machine import AssistantState
from jarvis.memory.models import Fact, Note, Reminder, Todo
from jarvis.memory.user_profile import PROFILE_PREFIX, UserProfile
from jarvis.utils.logging import get_logger
from jarvis.utils.timing import LatencyTracker, StageStats

log = get_logger(__name__)

ELLIPSIS: Final = "…"
CLOCK_FORMAT: Final = "%H:%M:%S"
DUE_FORMAT: Final = "%a %d %b %H:%M"
NO_VALUE: Final = "-"
EMPTY_SECTION: Final = "(nothing yet)"
_MS_PER_SECOND: Final = 1000.0


@dataclass(frozen=True, slots=True)
class ModelLimits:
    """Caps that keep the model, and every snapshot of it, small.

    Attributes:
        transcript_lines: Transcript lines kept; older ones scroll away.
        tool_calls: Tool-call rows kept.
        line_chars: Longest transcript line shown.
        args_chars: Longest tool-argument summary shown.
        error_chars: Longest tool error shown.
        memory_items: Facts, todos and reminders shown per section.
        notes_count_cap: Notes counted before the count reads ``N+``.
    """

    transcript_lines: int = 200
    tool_calls: int = 100
    line_chars: int = 400
    args_chars: int = 80
    error_chars: int = 120
    memory_items: int = 20
    notes_count_cap: int = 500


# -- presentation helpers -----------------------------------------------------


@dataclass(frozen=True, slots=True)
class StateStyle:
    """How one assistant state is shown.

    Attributes:
        label: Title-case name for headers and tooltips.
        colour: ``#RRGGBB``; the same palette as the native overlay.
    """

    label: str
    colour: str


STATE_STYLES: Final[dict[AssistantState, StateStyle]] = {
    AssistantState.IDLE: StateStyle("Idle", "#8C99B3"),
    AssistantState.LISTENING: StateStyle("Listening", "#4DD973"),
    AssistantState.THINKING: StateStyle("Thinking", "#FABF40"),
    AssistantState.SPEAKING: StateStyle("Speaking", "#59A6FF"),
}
"""Hex versions of ``Theme`` in ``native/overlay/include/jarvis/overlay/layout.hpp``."""

BACKGROUND_COLOUR: Final = "#12141C"
TEXT_COLOUR: Final = "#EDF0F5"
MUTED_TEXT_COLOUR: Final = "#99A1B3"
ERROR_COLOUR: Final = "#F26B6B"
"""Not in the overlay theme (it never shows errors); a red that reads on the dark background."""


def state_style(state: AssistantState) -> StateStyle:
    """Label and colour for ``state``."""
    return STATE_STYLES[state]


def truncate(text: str, limit: int) -> str:
    """Collapse whitespace to single spaces and cut to ``limit`` characters.

    Rows are one line tall, so newlines inside a transcript or a tool error
    would break the table; collapsing them keeps every row readable.
    """
    flat = " ".join(text.split())
    if len(flat) <= limit:
        return flat
    return flat[: max(limit - len(ELLIPSIS), 0)] + ELLIPSIS


def format_clock(timestamp: float, tz: tzinfo | None = None) -> str:
    """``HH:MM:SS`` for an epoch timestamp, in ``tz`` (local time when ``None``)."""
    return datetime.fromtimestamp(timestamp, tz).strftime(CLOCK_FORMAT)


def format_duration(ms: float) -> str:
    """Milliseconds below one second, seconds with one decimal above."""
    if ms < _MS_PER_SECOND:
        return f"{ms:.0f} ms"
    return f"{ms / _MS_PER_SECOND:.1f} s"


def tray_tooltip(state: AssistantState, *, microphone_muted: bool) -> str:
    """Tooltip text for the tray icon."""
    suffix = " (microphone muted)" if microphone_muted else ""
    return f"Jarvis - {state_style(state).label}{suffix}"


# -- snapshot types -------------------------------------------------------------


class Speaker(StrEnum):
    """Who a transcript line belongs to."""

    USER = "You"
    ASSISTANT = "Jarvis"


@dataclass(frozen=True, slots=True)
class TranscriptLine:
    """One line of the rolling transcript."""

    time: str
    speaker: Speaker
    text: str


class ToolOutcome(StrEnum):
    """Lifecycle of one tool call as the dashboard shows it."""

    AWAITING_CONFIRMATION = "awaiting confirmation"
    DECLINED = "declined"
    RUNNING = "running"
    OK = "ok"
    ERROR = "error"


OUTCOME_COLOURS: Final[dict[ToolOutcome, str]] = {
    ToolOutcome.AWAITING_CONFIRMATION: STATE_STYLES[AssistantState.SPEAKING].colour,
    ToolOutcome.DECLINED: MUTED_TEXT_COLOUR,
    ToolOutcome.RUNNING: STATE_STYLES[AssistantState.THINKING].colour,
    ToolOutcome.OK: STATE_STYLES[AssistantState.LISTENING].colour,
    ToolOutcome.ERROR: ERROR_COLOUR,
}
"""Row colour per outcome, reusing the state palette so the window reads as one theme."""


class Confirmation(StrEnum):
    """Whether, and how, the user was asked before a tool ran."""

    NOT_ASKED = "not asked"
    PENDING = "pending"
    APPROVED = "approved"
    DECLINED = "declined"


@dataclass(frozen=True, slots=True)
class ToolCallRow:
    """One row of the tool-call table; every field is display-ready text."""

    time: str
    name: str
    args: str
    duration: str
    outcome: ToolOutcome
    confirmation: Confirmation
    error: str


@dataclass(frozen=True, slots=True)
class LatencyRow:
    """Per-stage latency statistics, formatted."""

    stage: str
    count: int
    mean: str
    p50: str
    p95: str
    last: str


class MemoryStatus(StrEnum):
    """Where the memory inspector's data stands."""

    UNAVAILABLE = "no memory inspector"
    NOT_LOADED = "not loaded"
    LOADING = "loading"
    READY = "ready"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class MemoryView:
    """What the memory inspector tab shows."""

    status: MemoryStatus
    refreshed_at: str = NO_VALUE
    error: str = ""
    profile: tuple[str, ...] = ()
    facts: tuple[str, ...] = ()
    notes_count: str = NO_VALUE
    todos: tuple[str, ...] = ()
    reminders: tuple[str, ...] = ()

    def status_line(self) -> str:
        """One line for the top of the tab: state, time of the last read, error."""
        if self.status is MemoryStatus.FAILED:
            return f"{self.status.value}: {self.error}"
        if self.refreshed_at == NO_VALUE:
            return self.status.value
        return f"{self.status.value} (refreshed {self.refreshed_at})"

    def sections(self) -> tuple[tuple[str, tuple[str, ...]], ...]:
        """Heading and lines per section, with placeholders for empty ones."""
        if self.status is not MemoryStatus.READY:
            return ()
        return (
            ("Profile", self.profile or (EMPTY_SECTION,)),
            ("Recent facts", self.facts or (EMPTY_SECTION,)),
            ("Notes", (f"{self.notes_count} stored",)),
            ("Open todos", self.todos or (EMPTY_SECTION,)),
            ("Upcoming reminders", self.reminders or (EMPTY_SECTION,)),
        )


@dataclass(frozen=True, slots=True)
class Revisions:
    """Change counters per section, so a renderer redraws only what changed."""

    header: int = 0
    transcript: int = 0
    tool_calls: int = 0
    latency: int = 0
    memory: int = 0


@dataclass(frozen=True, slots=True)
class DashboardSnapshot:
    """Everything the dashboard draws, immutable and cheap to hand across threads."""

    state: AssistantState
    style: StateStyle
    state_reason: str
    state_since: str
    wake_words: int
    events_seen: int
    transcript: tuple[TranscriptLine, ...]
    tool_calls: tuple[ToolCallRow, ...]
    latency: tuple[LatencyRow, ...]
    memory: MemoryView
    revisions: Revisions

    def header_detail(self) -> str:
        """Second header line: when and why the current state began."""
        return f"since {self.state_since} ({self.state_reason})"

    def counters_line(self) -> str:
        """Right-hand header text: session counters."""
        return f"wake words {self.wake_words}  |  events {self.events_seen}"


# -- memory ---------------------------------------------------------------------


class MemoryInspector(Protocol):
    """Read-only view of the memory store the inspector tab needs.

    ``jarvis.memory.sqlite_store.SqliteStore`` satisfies it structurally, so
    the composition root passes the store it already owns and the UI never
    imports the concrete class.
    """

    async def list_facts(self, prefix: str = "") -> list[Fact]:
        """Facts whose key starts with ``prefix``."""

    async def list_notes(self, limit: int = 50) -> list[Note]:
        """Notes, newest first."""

    async def list_todos(self, *, include_done: bool = False) -> list[Todo]:
        """Todos, open ones only unless asked."""

    async def list_reminders(self, *, include_fired: bool = False) -> list[Reminder]:
        """Reminders by due time, pending ones only unless asked."""


async def build_memory_view(
    inspector: MemoryInspector, limits: ModelLimits, tz: tzinfo | None, now: float
) -> MemoryView:
    """Read the memory store and format it for the inspector tab."""
    facts = await inspector.list_facts()
    notes = await inspector.list_notes(limits.notes_count_cap)
    todos = await inspector.list_todos()
    reminders = await inspector.list_reminders()
    return MemoryView(
        status=MemoryStatus.READY,
        refreshed_at=format_clock(now, tz),
        profile=_profile_lines(facts),
        facts=_recent_fact_lines(facts, limits),
        notes_count=_notes_count(len(notes), limits.notes_count_cap),
        todos=tuple(
            truncate(todo.text, limits.line_chars) for todo in todos[: limits.memory_items]
        ),
        reminders=_reminder_lines(reminders, limits, tz),
    )


def _profile_lines(facts: Sequence[Fact]) -> tuple[str, ...]:
    rendered = UserProfile.from_facts(facts).render()
    # render() is prompt text: a "User profile:" heading, then one "- " bullet per line.
    return tuple(line.removeprefix("- ") for line in rendered.splitlines()[1:] if line)


def _recent_fact_lines(facts: Sequence[Fact], limits: ModelLimits) -> tuple[str, ...]:
    loose = [fact for fact in facts if not fact.key.startswith(PROFILE_PREFIX)]
    loose.sort(key=lambda fact: fact.updated_at, reverse=True)
    return tuple(
        truncate(
            f"{fact.key}: {fact.value} ({fact.source}, {fact.confidence:.0%})", limits.line_chars
        )
        for fact in loose[: limits.memory_items]
    )


def _notes_count(count: int, cap: int) -> str:
    return f"{count}+" if count >= cap else str(count)


def _reminder_lines(
    reminders: Sequence[Reminder], limits: ModelLimits, tz: tzinfo | None
) -> tuple[str, ...]:
    return tuple(
        f"{reminder.due_at.astimezone(tz).strftime(DUE_FORMAT)}  "
        f"{truncate(reminder.text, limits.line_chars)}"
        for reminder in reminders[: limits.memory_items]
    )


# -- the model --------------------------------------------------------------------


@dataclass(slots=True)
class _ToolCallEntry:
    """Mutable while the call is in flight; frozen into a row for each snapshot."""

    started: float
    name: str
    args: str = ""
    duration: str = NO_VALUE
    outcome: ToolOutcome = ToolOutcome.RUNNING
    confirmation: Confirmation = Confirmation.NOT_ASKED
    error: str = ""

    def row(self, tz: tzinfo | None) -> ToolCallRow:
        return ToolCallRow(
            time=format_clock(self.started, tz),
            name=self.name,
            args=self.args,
            duration=self.duration,
            outcome=self.outcome,
            confirmation=self.confirmation,
            error=self.error,
        )


@dataclass(slots=True)
class _Counters:
    header: int = 0
    transcript: int = 0
    tool_calls: int = 0
    latency: int = 0
    memory: int = 0
    events: int = 0
    wake_words: int = 0

    def revisions(self) -> Revisions:
        return Revisions(self.header, self.transcript, self.tool_calls, self.latency, self.memory)


ChangeListener = Callable[[], None]
"""Called on the event loop after the model changed; read ``snapshot()`` from it."""


@dataclass(slots=True)
class _StateView:
    state: AssistantState = AssistantState.IDLE
    reason: str = "start"
    since: float = field(default_factory=time.time)


class DashboardModel:
    """Folds bus events into display state; lives on the asyncio loop.

    Args:
        bus: Event bus to observe. Nothing is ever published back.
        memory: Source for the memory inspector; ``None`` shows the tab as
            unavailable instead of failing.
        latency: Tracker whose statistics the latency table shows. When
            ``None`` the model creates its own and feeds it from the bus;
            when given, its owner is responsible for feeding it (with
            ``LatencyTracker.attach`` or ``timed(tracker=...)``), so samples
            are never counted twice.
        limits: Caps and truncation lengths.
        tz: Time zone for displayed times; local time when ``None``.
        now: Wall-clock source, injectable for tests.
    """

    def __init__(
        self,
        bus: EventBus,
        *,
        memory: MemoryInspector | None = None,
        latency: LatencyTracker | None = None,
        limits: ModelLimits | None = None,
        tz: tzinfo | None = None,
        now: Callable[[], float] = time.time,
    ) -> None:
        self._bus = bus
        self._memory = memory
        self._owns_tracker = latency is None
        self._tracker = latency if latency is not None else LatencyTracker()
        self._limits = limits or ModelLimits()
        self._tz = tz
        self._now = now
        self._state = _StateView(since=now())
        self._transcript: deque[TranscriptLine] = deque(maxlen=self._limits.transcript_lines)
        self._tool_calls: deque[_ToolCallEntry] = deque(maxlen=self._limits.tool_calls)
        self._memory_view = MemoryView(
            MemoryStatus.NOT_LOADED if memory is not None else MemoryStatus.UNAVAILABLE
        )
        self._counters = _Counters()
        self._listeners: list[ChangeListener] = []
        self._unsubscribers: list[Unsubscribe] = []
        self._refresh: asyncio.Task[None] | None = None

    @property
    def attached(self) -> bool:
        """True between :meth:`attach` and :meth:`detach`."""
        return bool(self._unsubscribers)

    def attach(self) -> None:
        """Start observing the bus. Idempotent."""
        if self.attached:
            return
        handlers: list[tuple[type[Event], Callable[[Event], None]]] = [
            (Event, self._count_event),
            (StateChanged, self._typed(StateChanged, self._on_state_changed)),
            (WakeWordHeard, self._typed(WakeWordHeard, self._on_wake_word)),
            (TranscriptReady, self._typed(TranscriptReady, self._on_transcript)),
            (AssistantSaid, self._typed(AssistantSaid, self._on_assistant_said)),
            (ToolCallStarted, self._typed(ToolCallStarted, self._on_tool_started)),
            (ToolCallFinished, self._typed(ToolCallFinished, self._on_tool_finished)),
            (ConfirmationRequested, self._typed(ConfirmationRequested, self._on_confirm_asked)),
            (ConfirmationResolved, self._typed(ConfirmationResolved, self._on_confirm_resolved)),
            (LatencySample, self._typed(LatencySample, self._on_latency)),
        ]
        self._unsubscribers = [self._bus.subscribe_callback(t, h) for t, h in handlers]
        log.debug("dashboard.model.attached", owns_tracker=self._owns_tracker)

    def detach(self) -> None:
        """Stop observing the bus. Idempotent."""
        for unsubscribe in self._unsubscribers:
            unsubscribe()
        self._unsubscribers = []

    def add_listener(self, listener: ChangeListener) -> Unsubscribe:
        """Call ``listener`` after every change; returns a remover."""
        self._listeners.append(listener)

        def remove() -> None:
            if listener in self._listeners:
                self._listeners.remove(listener)

        return remove

    def snapshot(self) -> DashboardSnapshot:
        """The current display state as an immutable value."""
        return DashboardSnapshot(
            state=self._state.state,
            style=state_style(self._state.state),
            state_reason=self._state.reason,
            state_since=format_clock(self._state.since, self._tz),
            wake_words=self._counters.wake_words,
            events_seen=self._counters.events,
            transcript=tuple(self._transcript),
            tool_calls=tuple(entry.row(self._tz) for entry in self._tool_calls),
            latency=self._latency_rows(),
            memory=self._memory_view,
            revisions=self._counters.revisions(),
        )

    def request_memory_refresh(self) -> None:
        """Schedule :meth:`refresh_memory` unless one is already running.

        Must be called on the event loop (the dashboard hops onto it with
        ``call_soon_threadsafe``); the task reference is kept so it cannot
        be garbage-collected mid-flight.
        """
        if self._refresh is not None and not self._refresh.done():
            return
        self._refresh = asyncio.get_running_loop().create_task(self.refresh_memory())

    async def refresh_memory(self) -> None:
        """Re-read the memory store into the inspector view.

        A failing store is logged with its traceback and shown on the tab as
        ``FAILED`` with the error text; the dashboard is a diagnostic tool,
        so it reports the failure rather than raising into a UI callback.
        """
        if self._memory is None:
            return
        self._set_memory(
            MemoryView(MemoryStatus.LOADING, refreshed_at=self._memory_view.refreshed_at)
        )
        try:
            view = await build_memory_view(self._memory, self._limits, self._tz, self._now())
        except Exception as error:
            log.exception("dashboard.memory.refresh_failed")
            view = MemoryView(
                MemoryStatus.FAILED, error=truncate(str(error), self._limits.error_chars)
            )
        self._set_memory(view)

    async def close(self) -> None:
        """Detach and wait for an in-flight memory refresh."""
        self.detach()
        if self._refresh is not None:
            await asyncio.gather(self._refresh, return_exceptions=True)

    # -- event handlers ---------------------------------------------------------

    @staticmethod
    def _typed[E: Event](
        event_type: type[E], handler: Callable[[E], None]
    ) -> Callable[[Event], None]:
        def dispatch(event: Event) -> None:
            if isinstance(event, event_type):
                handler(event)

        return dispatch

    def _count_event(self, _event: Event) -> None:
        self._counters.events += 1

    def _on_state_changed(self, event: StateChanged) -> None:
        self._state = _StateView(event.new, event.reason, event.timestamp)
        self._counters.header += 1
        self._notify()

    def _on_wake_word(self, _event: WakeWordHeard) -> None:
        self._counters.wake_words += 1
        self._counters.header += 1
        self._notify()

    def _on_transcript(self, event: TranscriptReady) -> None:
        self._add_line(event, Speaker.USER, event.text)

    def _on_assistant_said(self, event: AssistantSaid) -> None:
        self._add_line(event, Speaker.ASSISTANT, event.text)

    def _add_line(self, event: Event, speaker: Speaker, text: str) -> None:
        line = TranscriptLine(
            format_clock(event.timestamp, self._tz),
            speaker,
            truncate(text, self._limits.line_chars),
        )
        self._transcript.append(line)
        self._counters.transcript += 1
        self._notify()

    def _on_confirm_asked(self, event: ConfirmationRequested) -> None:
        entry = _ToolCallEntry(
            started=event.timestamp,
            name=event.tool_name,
            args=truncate(event.prompt, self._limits.args_chars),
            outcome=ToolOutcome.AWAITING_CONFIRMATION,
            confirmation=Confirmation.PENDING,
        )
        self._tool_calls.append(entry)
        self._tools_changed()

    def _on_confirm_resolved(self, event: ConfirmationResolved) -> None:
        entry = self._find_entry(event.tool_name, ToolOutcome.AWAITING_CONFIRMATION)
        if entry is None:
            log.debug("dashboard.confirmation.unmatched", tool=event.tool_name)
            return
        if event.approved:
            entry.confirmation = Confirmation.APPROVED
        else:
            entry.confirmation = Confirmation.DECLINED
            entry.outcome = ToolOutcome.DECLINED
        self._tools_changed()

    def _on_tool_started(self, event: ToolCallStarted) -> None:
        args = truncate(event.args_summary, self._limits.args_chars)
        entry = self._approved_entry(event.name)
        if entry is None:
            entry = _ToolCallEntry(started=event.timestamp, name=event.name)
            self._tool_calls.append(entry)
        entry.args = args
        entry.outcome = ToolOutcome.RUNNING
        self._tools_changed()

    def _on_tool_finished(self, event: ToolCallFinished) -> None:
        entry = self._find_entry(event.name, ToolOutcome.RUNNING)
        if entry is None:
            entry = _ToolCallEntry(started=event.timestamp, name=event.name)
            self._tool_calls.append(entry)
        entry.args = truncate(event.args_summary, self._limits.args_chars)
        entry.duration = format_duration(event.duration_ms)
        entry.outcome = ToolOutcome.OK if event.ok else ToolOutcome.ERROR
        entry.error = truncate(event.error or "", self._limits.error_chars)
        self._tools_changed()

    def _on_latency(self, event: LatencySample) -> None:
        if self._owns_tracker:
            self._tracker.record(event.stage, event.ms)
        self._counters.latency += 1
        self._notify()

    # -- helpers --------------------------------------------------------------------

    def _find_entry(self, name: str, outcome: ToolOutcome) -> _ToolCallEntry | None:
        """Newest entry for ``name`` in ``outcome``; tools of one turn run in order."""
        for entry in reversed(self._tool_calls):
            if entry.name == name and entry.outcome is outcome:
                return entry
        return None

    def _approved_entry(self, name: str) -> _ToolCallEntry | None:
        for entry in reversed(self._tool_calls):
            if (
                entry.name == name
                and entry.confirmation is Confirmation.APPROVED
                and entry.outcome is ToolOutcome.AWAITING_CONFIRMATION
            ):
                return entry
        return None

    def _tools_changed(self) -> None:
        self._counters.tool_calls += 1
        self._notify()

    def _set_memory(self, view: MemoryView) -> None:
        self._memory_view = view
        self._counters.memory += 1
        self._notify()

    def _latency_rows(self) -> tuple[LatencyRow, ...]:
        return tuple(_latency_row(stats) for stats in self._tracker.snapshot().values())

    def _notify(self) -> None:
        for listener in list(self._listeners):
            listener()


def _latency_row(stats: StageStats) -> LatencyRow:
    return LatencyRow(
        stage=stats.stage,
        count=stats.count,
        mean=format_duration(stats.mean_ms),
        p50=format_duration(stats.p50_ms),
        p95=format_duration(stats.p95_ms),
        last=format_duration(stats.last_ms),
    )


# Design notes:
# - The model is the only place that knows what an event "means" for the
#   screen. Renderers get frozen snapshots plus per-section revision
#   counters, so a Tk view on another thread never touches mutable state
#   and redraws a table only when its counter moved.
# - Confirmation and tool events are correlated by tool name, newest first.
#   The agent runs one turn's tools sequentially and publishes
#   ConfirmationRequested -> ConfirmationResolved -> ToolCallStarted ->
#   ToolCallFinished for each, so that is unambiguous today. If the agent
#   ever runs tools concurrently, the events need a call id; until then an
#   unmatched event still produces a row instead of being dropped.
# - When the user approves, the confirmation row is reused for the run, so
#   one row tells the whole story ("approved", "ok", "1.2 s"). An approval
#   whose ToolCallStarted never arrives (invalid arguments) stays visible
#   as "awaiting confirmation / approved".
# - Provider and integration status is not shown: no event carries it yet.
#   It belongs in core/events.py as a status event, at which point it is
#   one more handler and one more snapshot field here.
# - Colours are duplicated from the C++ Theme rather than read from the
#   native module, because the dashboard and tray must work when the native
#   build is missing. The hex values are rounded from the float RGBA there.
