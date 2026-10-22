"""Dashboard window: live state, transcript, tool calls, latency, memory.

The window is a standard-library ``tkinter`` application running on a
thread of its own with its own ``mainloop``. It never shares Tk objects
with the asyncio loop: the loop side pushes immutable
:class:`~jarvis.ui.view_model.DashboardSnapshot` values and commands into a
thread-safe ``queue.Queue``, and the Tk thread drains that queue from an
``after()`` timer. The only traffic the other way is the memory refresh
button, which hops onto the loop with ``call_soon_threadsafe``.

All formatting lives in :mod:`jarvis.ui.view_model`; this module only maps
snapshot fields onto widgets. :class:`StubDashboard` stands in when there
is no display, and :func:`create_dashboard` picks between the two.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import gc
import queue
import threading
import time
import tkinter
import tkinter.font
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from tkinter import ttk
from typing import Final, Protocol

from jarvis.core.events import EventBus, Unsubscribe
from jarvis.core.state_machine import AssistantState
from jarvis.ui.view_model import (
    BACKGROUND_COLOUR,
    MUTED_TEXT_COLOUR,
    OUTCOME_COLOURS,
    STATE_STYLES,
    TEXT_COLOUR,
    DashboardModel,
    DashboardSnapshot,
    LatencyRow,
    MemoryInspector,
    MemoryView,
    Speaker,
    ToolCallRow,
    TranscriptLine,
)
from jarvis.utils.errors import JarvisError
from jarvis.utils.logging import get_logger
from jarvis.utils.timing import LatencyTracker

log = get_logger(__name__)

START_TIMEOUT_S: Final = 5.0
STOP_TIMEOUT_S: Final = 5.0
_INDICATOR_SIZE: Final = 18
_PAD: Final = 8
_ROW_PADDING: Final = 6
_BASE_DPI: Final = 96.0
"""Treeview rows are a fixed height; derive it from the font so high-DPI text fits."""
_FONT: Final = ("Segoe UI", 10)
_HEADER_FONT: Final = ("Segoe UI Semibold", 14)
_MONO_FONT: Final = ("Consolas", 10)
_SECTION_FONT: Final = ("Segoe UI Semibold", 11)
_SURFACE_COLOUR: Final = "#1B1E29"
"""Panels a shade lighter than the overlay background, so tabs read as cards."""
_IDLE_COLOUR: Final = STATE_STYLES[AssistantState.IDLE].colour
_USER_COLOUR: Final = STATE_STYLES[AssistantState.LISTENING].colour
_ASSISTANT_COLOUR: Final = STATE_STYLES[AssistantState.SPEAKING].colour

_TOOL_COLUMNS: Final = (
    ("time", "Time", 80),
    ("name", "Tool", 140),
    ("args", "Arguments", 280),
    ("duration", "Duration", 80),
    ("confirmation", "Confirmation", 110),
    ("outcome", "Result", 170),
    ("error", "Error", 260),
)
_LATENCY_COLUMNS: Final = (
    ("stage", "Stage", 140),
    ("count", "Count", 60),
    ("mean", "Mean", 75),
    ("p50", "p50", 75),
    ("p95", "p95", 75),
    ("last", "Last", 75),
)


_TK_START_ATTEMPTS: Final = 3
_TK_RETRY_SECONDS: Final = 0.05
_TK_TRANSIENT_MARKERS: Final = ("couldn't read file", "tcl_findLibrary", "init.tcl")
"""Ways a half-read Tcl library shows up; all clear on the next attempt."""


def create_tk_root() -> tkinter.Tk:
    """Create the Tk root, retrying a transient failure to read Tcl's own scripts.

    On Windows, Tcl occasionally fails to read ``init.tcl`` with "couldn't
    read file ...: No error" while another process (an antivirus scan, a
    parallel build) holds the file; the next attempt succeeds. Any other
    TclError, such as no display, is raised at once. A partially read
    library can also surface as ``invalid command name "tcl_findLibrary"``.
    """
    for attempt in range(1, _TK_START_ATTEMPTS + 1):
        try:
            return tkinter.Tk()
        except tkinter.TclError as error:
            transient = any(marker in str(error) for marker in _TK_TRANSIENT_MARKERS)
            if not transient or attempt == _TK_START_ATTEMPTS:
                raise
            log.warning("dashboard.tk.retry", attempt=attempt)
            time.sleep(_TK_RETRY_SECONDS)
    raise AssertionError("unreachable")  # pragma: no cover


class DashboardError(JarvisError):
    """The dashboard window could not be created (no display, no Tk)."""


@dataclass(frozen=True, slots=True)
class DashboardConfig:
    """Window geometry and behaviour.

    Attributes:
        title: Window title.
        width: Initial width in pixels.
        height: Initial height in pixels.
        poll_ms: How often the Tk thread drains the snapshot queue. 100 ms
            is faster than a person reads and slower than anything costs.
        start_visible: Show the window on start; the composition root sets
            False and lets the tray's "Show dashboard" open it.
    """

    title: str = "Jarvis dashboard"
    width: int = 1120
    height: int = 720
    poll_ms: int = 100
    start_visible: bool = True


class _Command(Enum):
    SHOW = "show"
    HIDE = "hide"
    STOP = "stop"


_InboxItem = DashboardSnapshot | _Command


class DashboardSurface(Protocol):
    """What the composition root holds: a :class:`Dashboard` or a :class:`StubDashboard`."""

    @property
    def model(self) -> DashboardModel:
        """The view model feeding the surface."""

    @property
    def running(self) -> bool:
        """Whether the surface is active."""

    async def start(self) -> None:
        """Attach the model and open the window."""

    async def stop(self) -> None:
        """Close the window and detach the model."""

    def show(self) -> None:
        """Bring the window to the front."""

    def hide(self) -> None:
        """Hide the window without closing it."""


# -- Tk side: everything below until Dashboard runs only on the Tk thread -----------


class _DashboardView:
    """The widgets. Created, used and destroyed on the Tk thread only."""

    def __init__(self, config: DashboardConfig, on_refresh: Callable[[], None]) -> None:
        self.root = create_tk_root()
        self.root.title(config.title)
        self._scale = self.root.winfo_fpixels("1i") / _BASE_DPI
        self.root.geometry(f"{self._px(config.width)}x{self._px(config.height)}")
        self.root.configure(background=BACKGROUND_COLOUR)
        self.root.report_callback_exception = _log_tk_exception
        self.root.protocol("WM_DELETE_WINDOW", self.root.withdraw)
        _apply_theme(self.root)
        self._seen = (-1, -1, -1, -1, -1)
        self._build_header()
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill="both", expand=True, padx=_PAD, pady=(0, _PAD))
        self._build_live_tab(notebook)
        self._memory_status, self._memory_text = self._build_memory(notebook, on_refresh)
        if not config.start_visible:
            self.root.withdraw()

    def _build_header(self) -> None:
        header = ttk.Frame(self.root, padding=_PAD)
        header.pack(fill="x")
        self._indicator = tkinter.Canvas(
            header,
            width=_INDICATOR_SIZE,
            height=_INDICATOR_SIZE,
            background=BACKGROUND_COLOUR,
            highlightthickness=0,
        )
        self._indicator.pack(side="left", padx=(0, _PAD))
        self._dot = self._indicator.create_oval(
            1, 1, _INDICATOR_SIZE - 1, _INDICATOR_SIZE - 1, fill=_IDLE_COLOUR, width=0
        )
        self._state_label = ttk.Label(header, style="Header.TLabel")
        self._state_label.pack(side="left")
        self._detail_label = ttk.Label(header, style="Muted.TLabel")
        self._detail_label.pack(side="left", padx=_PAD)
        self._counters_label = ttk.Label(header, style="Muted.TLabel")
        self._counters_label.pack(side="right")

    def _build_live_tab(self, notebook: ttk.Notebook) -> None:
        """Transcript beside latency on top, the tool-call log across the bottom."""
        rows = ttk.PanedWindow(notebook, orient="vertical")
        notebook.add(rows, text="Live")
        top = ttk.PanedWindow(rows, orient="horizontal")
        rows.add(top, weight=3)
        self._transcript = _build_transcript(_section(top, "Transcript", weight=3))
        latency_pane = _section(top, "Latency per stage", weight=2)
        self._latency = _build_table(latency_pane, _LATENCY_COLUMNS, self._px)
        self._tools = _build_table(_section(rows, "Tool calls", weight=2), _TOOL_COLUMNS, self._px)
        for outcome, colour in OUTCOME_COLOURS.items():
            self._tools.tag_configure(outcome.value, foreground=colour)

    def _build_memory(
        self, notebook: ttk.Notebook, on_refresh: Callable[[], None]
    ) -> tuple[ttk.Label, tkinter.Text]:
        frame = ttk.Frame(notebook, padding=_PAD)
        notebook.add(frame, text="Memory")
        bar = ttk.Frame(frame)
        bar.pack(fill="x", pady=(0, _PAD))
        ttk.Button(bar, text="Refresh", command=on_refresh).pack(side="left")
        status = ttk.Label(bar, style="Muted.TLabel")
        status.pack(side="left", padx=_PAD)
        text = _text_widget(frame)
        text.pack(fill="both", expand=True)
        text.tag_configure("heading", font=_SECTION_FONT)
        return status, text

    def _px(self, logical: int) -> int:
        """Logical (96 DPI) pixels to device pixels; Tk geometry is in device pixels."""
        return round(logical * self._scale)

    def run(self, poll_ms: int, drain: Callable[[_DashboardView], bool]) -> None:
        """Run the Tk mainloop, draining the inbox every ``poll_ms`` until told to stop."""

        def poll() -> None:
            if drain(self):
                self.root.after(poll_ms, poll)
            else:
                self.root.quit()

        self.root.after(0, poll)
        self.root.mainloop()

    def show(self) -> None:
        self.root.deiconify()
        self.root.lift()
        self.root.focus_force()

    def hide(self) -> None:
        self.root.withdraw()

    def render(self, snapshot: DashboardSnapshot) -> None:
        """Redraw the sections whose revision moved since the last render."""
        revisions = snapshot.revisions
        current = (
            revisions.header,
            revisions.transcript,
            revisions.tool_calls,
            revisions.latency,
            revisions.memory,
        )
        renderers = (
            self._render_header,
            self._render_transcript,
            self._render_tools,
            self._render_latency,
            self._render_memory,
        )
        for seen, now, render in zip(self._seen, current, renderers, strict=True):
            if seen != now:
                render(snapshot)
        self._seen = current
        self._counters_label.configure(text=snapshot.counters_line())

    def _render_header(self, snapshot: DashboardSnapshot) -> None:
        self._indicator.itemconfigure(self._dot, fill=snapshot.style.colour)
        self._state_label.configure(text=snapshot.style.label, foreground=snapshot.style.colour)
        self._detail_label.configure(text=snapshot.header_detail())

    def _render_transcript(self, snapshot: DashboardSnapshot) -> None:
        _replace_text(self._transcript, lambda text: _write_transcript(text, snapshot.transcript))
        self._transcript.see("end")

    def _render_tools(self, snapshot: DashboardSnapshot) -> None:
        rows = [(_tool_values(row), row.outcome.value) for row in snapshot.tool_calls]
        _replace_rows(self._tools, rows)

    def _render_latency(self, snapshot: DashboardSnapshot) -> None:
        _replace_rows(self._latency, [(_latency_values(row), "") for row in snapshot.latency])

    def _render_memory(self, snapshot: DashboardSnapshot) -> None:
        self._memory_status.configure(text=snapshot.memory.status_line())
        _replace_text(self._memory_text, lambda text: _write_memory(text, snapshot.memory))

    def destroy(self) -> None:
        self.root.destroy()


def _apply_theme(root: tkinter.Tk) -> None:
    style = ttk.Style(root)
    style.theme_use("clam")
    style.configure(".", background=BACKGROUND_COLOUR, foreground=TEXT_COLOUR, font=_FONT)
    style.configure("Header.TLabel", font=_HEADER_FONT)
    style.configure("Section.TLabel", font=_SECTION_FONT, foreground=MUTED_TEXT_COLOUR)
    style.configure("Muted.TLabel", foreground=MUTED_TEXT_COLOUR)
    style.configure("TNotebook", background=BACKGROUND_COLOUR, borderwidth=0)
    style.configure("TNotebook.Tab", background=_SURFACE_COLOUR, padding=(12, 4))
    style.map("TNotebook.Tab", background=[("selected", BACKGROUND_COLOUR)])
    style.configure(
        "Treeview",
        background=_SURFACE_COLOUR,
        fieldbackground=_SURFACE_COLOUR,
        foreground=TEXT_COLOUR,
        borderwidth=0,
        rowheight=tkinter.font.Font(root=root, font=_FONT).metrics("linespace") + _ROW_PADDING,
    )
    style.configure("Treeview.Heading", background=BACKGROUND_COLOUR, foreground=MUTED_TEXT_COLOUR)
    style.configure("TButton", background=_SURFACE_COLOUR)


def _text_widget(parent: tkinter.Misc) -> tkinter.Text:
    return tkinter.Text(
        parent,
        wrap="word",
        state="disabled",
        background=_SURFACE_COLOUR,
        foreground=TEXT_COLOUR,
        insertbackground=TEXT_COLOUR,
        borderwidth=0,
        padx=_PAD,
        pady=_PAD,
        font=_MONO_FONT,
    )


def _section(paned: ttk.PanedWindow, title: str, *, weight: int) -> ttk.Frame:
    """A titled pane; the caller fills it with one widget."""
    frame = ttk.Frame(paned, padding=(0, _PAD, _PAD, 0))
    ttk.Label(frame, text=title, style="Section.TLabel").pack(anchor="w", pady=(0, _PAD // 2))
    paned.add(frame, weight=weight)
    return frame


def _build_transcript(parent: ttk.Frame) -> tkinter.Text:
    text = _text_widget(parent)
    text.pack(fill="both", expand=True)
    text.tag_configure("time", foreground=MUTED_TEXT_COLOUR)
    text.tag_configure(Speaker.USER.value, foreground=_USER_COLOUR)
    text.tag_configure(Speaker.ASSISTANT.value, foreground=_ASSISTANT_COLOUR)
    return text


def _build_table(
    parent: ttk.Frame, columns: tuple[tuple[str, str, int], ...], px: Callable[[int], int]
) -> ttk.Treeview:
    table = ttk.Treeview(parent, columns=[key for key, _, _ in columns], show="headings")
    for key, heading, width in columns:
        table.heading(key, text=heading, anchor="w")
        table.column(key, width=px(width), minwidth=px(width) // 2, anchor="w")
    table.pack(fill="both", expand=True)
    return table


def _replace_text(widget: tkinter.Text, write: Callable[[tkinter.Text], None]) -> None:
    widget.configure(state="normal")
    widget.delete("1.0", "end")
    write(widget)
    widget.configure(state="disabled")


def _write_transcript(widget: tkinter.Text, lines: tuple[TranscriptLine, ...]) -> None:
    for line in lines:
        widget.insert("end", f"{line.time}  ", "time")
        widget.insert("end", f"{line.speaker.value}: ", line.speaker.value)
        widget.insert("end", f"{line.text}\n")


def _write_memory(widget: tkinter.Text, memory: MemoryView) -> None:
    for heading, lines in memory.sections():
        widget.insert("end", f"{heading}\n", "heading")
        for line in lines:
            widget.insert("end", f"  {line}\n")
        widget.insert("end", "\n")


def _replace_rows(table: ttk.Treeview, rows: list[tuple[tuple[object, ...], str]]) -> None:
    table.delete(*table.get_children())
    for values, tag in rows:
        table.insert("", "end", values=values, tags=(tag,) if tag else ())
    children = table.get_children()
    if children:
        table.see(children[-1])


def _tool_values(row: ToolCallRow) -> tuple[object, ...]:
    return (
        row.time,
        row.name,
        row.args,
        row.duration,
        row.confirmation.value,
        row.outcome.value,
        row.error,
    )


def _latency_values(row: LatencyRow) -> tuple[object, ...]:
    return (row.stage, row.count, row.mean, row.p50, row.p95, row.last)


def _log_tk_exception(*exc_info: object) -> None:
    """Route exceptions raised in Tk callbacks to structlog instead of stderr."""
    error = exc_info[1]
    log.error("dashboard.tk.callback_failed", error=repr(error), exc_info=error)


# -- loop side -------------------------------------------------------------------------


class Dashboard:
    """Owns the Tk thread and feeds it snapshots of a :class:`DashboardModel`.

    Every public method is called from the asyncio loop and returns without
    waiting on Tk; :meth:`start` and :meth:`stop` await the Tk thread
    through futures and ``asyncio.to_thread`` rather than blocking.

    Args:
        model: The view model to display. :meth:`start` attaches it to the
            bus and :meth:`stop` closes it.
        loop: The loop the model lives on; the refresh button schedules
            work there.
        config: Window geometry and behaviour.
    """

    def __init__(
        self,
        model: DashboardModel,
        loop: asyncio.AbstractEventLoop,
        config: DashboardConfig | None = None,
    ) -> None:
        self._model = model
        self._loop = loop
        self._config = config or DashboardConfig()
        self._inbox: queue.Queue[_InboxItem] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._remove_listener: Unsubscribe | None = None

    @property
    def model(self) -> DashboardModel:
        """The view model feeding the window."""
        return self._model

    @property
    def running(self) -> bool:
        """True while the Tk thread is alive."""
        return self._thread is not None and self._thread.is_alive()

    async def start(self) -> None:
        """Create the window on its own thread and start feeding it.

        Raises:
            DashboardError: If Tk cannot open a window (no display) or the
                window does not appear within ``START_TIMEOUT_S``.
        """
        if self._thread is not None:
            return
        ready: concurrent.futures.Future[None] = concurrent.futures.Future()
        self._thread = threading.Thread(
            target=self._run_ui, args=(ready,), name="jarvis-dashboard", daemon=True
        )
        self._thread.start()
        try:
            # Shielded so a timeout does not cancel ``ready`` under the UI thread,
            # which would make its later set_result raise InvalidStateError.
            await asyncio.wait_for(asyncio.shield(asyncio.wrap_future(ready)), START_TIMEOUT_S)
        except TimeoutError as error:
            self._inbox.put(_Command.STOP)
            raise DashboardError("dashboard window did not open in time") from error
        self._model.attach()
        self._remove_listener = self._model.add_listener(self._publish_snapshot)
        self._publish_snapshot()
        self._model.request_memory_refresh()
        log.info("dashboard.started")

    async def stop(self) -> None:
        """Close the window, join the Tk thread and close the model. Idempotent."""
        if self._remove_listener is not None:
            self._remove_listener()
            self._remove_listener = None
        thread, self._thread = self._thread, None
        if thread is not None:
            self._inbox.put(_Command.STOP)
            await asyncio.to_thread(thread.join, STOP_TIMEOUT_S)
            if thread.is_alive():
                log.warning("dashboard.stop.timeout", timeout_s=STOP_TIMEOUT_S)
        await self._model.close()
        log.info("dashboard.stopped")

    def show(self) -> None:
        """Bring the window to the front and refresh the memory tab."""
        self._inbox.put(_Command.SHOW)
        self._model.request_memory_refresh()

    def hide(self) -> None:
        """Hide the window; it keeps updating underneath."""
        self._inbox.put(_Command.HIDE)

    def _publish_snapshot(self) -> None:
        self._inbox.put(self._model.snapshot())

    def _request_refresh_from_ui(self) -> None:
        """Refresh button handler, on the Tk thread: hop onto the loop."""
        self._loop.call_soon_threadsafe(self._model.request_memory_refresh)

    def _run_ui(self, ready: concurrent.futures.Future[None]) -> None:
        """Tk thread body: build, run and tear down the window on this thread."""
        try:
            view = _DashboardView(self._config, self._request_refresh_from_ui)
        except tkinter.TclError as error:
            failure = DashboardError(f"cannot open the dashboard window: {error}")
            failure.__cause__ = error
            ready.set_exception(failure)
            return
        ready.set_result(None)
        try:
            view.run(self._config.poll_ms, self._drain)
        finally:
            view.destroy()
            del view
            # Tk objects hold reference cycles; collect them here so the Tcl
            # interpreter is never finalised on another thread, which aborts.
            gc.collect()

    def _drain(self, view: _DashboardView) -> bool:
        """Apply queued commands and the newest snapshot; False once told to stop."""
        latest: DashboardSnapshot | None = None
        while True:
            try:
                item = self._inbox.get_nowait()
            except queue.Empty:
                break
            if item is _Command.STOP:
                return False
            if item is _Command.SHOW:
                view.show()
            elif item is _Command.HIDE:
                view.hide()
            elif isinstance(item, DashboardSnapshot):
                latest = item
        if latest is not None:
            view.render(latest)
        return True


class StubDashboard:
    """Stand-in for :class:`Dashboard` when there is no display (headless, text mode).

    Keeps the model attached so its snapshot stays current and anything
    that inspects it (tests, a future text renderer) still works, and logs
    show/hide at debug level. Nothing is drawn; this is a stub, not a
    fallback renderer.
    """

    def __init__(self, model: DashboardModel) -> None:
        self._model = model
        self._running = False
        self.show_requests = 0

    @property
    def model(self) -> DashboardModel:
        """The view model the stub keeps current."""
        return self._model

    @property
    def running(self) -> bool:
        """True between :meth:`start` and :meth:`stop`."""
        return self._running

    async def start(self) -> None:
        """Attach the model to the bus."""
        self._model.attach()
        self._running = True

    async def stop(self) -> None:
        """Detach and close the model. Idempotent."""
        self._running = False
        await self._model.close()

    def show(self) -> None:
        """Count the request; there is no window to show."""
        self.show_requests += 1
        log.debug("dashboard.stub.show", state=self._model.snapshot().state.value)

    def hide(self) -> None:
        """Nothing to hide."""
        log.debug("dashboard.stub.hide")


async def create_dashboard(
    bus: EventBus,
    memory_inspector: MemoryInspector | None,
    latency_tracker: LatencyTracker | None,
    *,
    config: DashboardConfig | None = None,
    enabled: bool = True,
) -> DashboardSurface:
    """Start the Tk dashboard when a display is available, otherwise a :class:`StubDashboard`.

    Must be awaited on the loop that publishes on ``bus``. See
    :class:`~jarvis.ui.view_model.DashboardModel` for who feeds
    ``latency_tracker``.
    """
    model = DashboardModel(bus, memory=memory_inspector, latency=latency_tracker)
    if enabled:
        dashboard = Dashboard(model, asyncio.get_running_loop(), config)
        try:
            await dashboard.start()
        except DashboardError as error:
            log.warning("dashboard.unavailable", reason=str(error))
        else:
            return dashboard
    stub = StubDashboard(model)
    await stub.start()
    return stub


# Design notes:
# - tkinter over Qt, Dear ImGui or a local web page: it ships with the
#   interpreter (no dependency), and the dashboard redraws a few tables a
#   few times a second, so the GIL argument that put the overlay in C++
#   (ADR-0011) does not apply. ADR-0015 records the comparison.
# - Tk is thread-affine: the interpreter, every widget and their
#   destruction stay on the thread that created them. The loop side only
#   touches queue.Queue, and the gc.collect() at the end of the thread
#   makes sure widget reference cycles are not finalised on another thread.
# - The queue is unbounded but self-limiting: the Tk thread drains it every
#   poll_ms and renders only the newest snapshot, and each section is
#   redrawn only when its revision counter moved.
# - Closing the window hides it (WM_DELETE_WINDOW -> withdraw): the
#   dashboard is a view the tray reopens, not an application to quit.
# - DashboardSurface keeps the composition root independent of whether a
#   window exists; a stub keeps the model current either way.
