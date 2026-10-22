import asyncio
import threading
import time
import tkinter
from collections.abc import Callable, Iterator
from datetime import UTC
from tkinter import ttk

import pytest
from structlog.testing import CapturingLogger

from jarvis.core.events import (
    AssistantSaid,
    EventBus,
    LatencySample,
    StateChanged,
    ToolCallFinished,
    ToolCallStarted,
    TranscriptReady,
)
from jarvis.core.state_machine import AssistantState
from jarvis.ui import dashboard as dashboard_module
from jarvis.ui.dashboard import (
    Dashboard,
    DashboardConfig,
    DashboardError,
    StubDashboard,
    _Command,
    _DashboardView,
    _log_tk_exception,
    create_dashboard,
)
from jarvis.ui.view_model import DashboardModel, DashboardSnapshot, MemoryStatus
from tests.ui.test_view_model import FakeMemoryInspector

HIDDEN = DashboardConfig(start_visible=False)


class FakeView:
    """Records what ``Dashboard._drain`` asks of the Tk view."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.rendered: list[DashboardSnapshot] = []

    def show(self) -> None:
        self.calls.append("show")

    def hide(self) -> None:
        self.calls.append("hide")

    def render(self, snapshot: DashboardSnapshot) -> None:
        self.calls.append("render")
        self.rendered.append(snapshot)


@pytest.fixture
def bus() -> EventBus:
    return EventBus()


@pytest.fixture
def model(bus: EventBus) -> DashboardModel:
    return DashboardModel(bus, memory=FakeMemoryInspector(), tz=UTC)


@pytest.fixture
def display(capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Skip unless Tk can open a window here (never on the headless CI runner).

    Output capture is suspended for the test: on Windows, Tcl intermittently
    fails to read its own init scripts ("couldn't read file ...: No error")
    while pytest has redirected file descriptors 0-2. The module logger is
    replaced so the uncaptured terminal stays quiet.
    """
    monkeypatch.setattr(dashboard_module, "log", CapturingLogger())
    with capsys.disabled():
        try:
            root = tkinter.Tk()
        except tkinter.TclError as error:
            pytest.skip(f"no display: {error}")
        root.destroy()
        yield


def _publish_turn(bus: EventBus) -> None:
    bus.publish(StateChanged(old=AssistantState.IDLE, new=AssistantState.LISTENING, reason="t"))
    bus.publish(TranscriptReady(text="open code", latency_ms=90))
    bus.publish(ToolCallStarted(name="open_app", args_summary="{}"))
    bus.publish(ToolCallFinished(name="open_app", args_summary="{}", duration_ms=12, ok=True))
    bus.publish(AssistantSaid(text="Done."))
    bus.publish(LatencySample(stage="stt", ms=90))


# -- stub and factory ---------------------------------------------------------------------


async def test_stub_keeps_the_model_current(bus: EventBus, model: DashboardModel) -> None:
    stub = StubDashboard(model)
    await stub.start()

    _publish_turn(bus)
    stub.show()
    stub.hide()

    assert stub.running
    assert stub.model is model
    assert stub.show_requests == 1
    assert len(model.snapshot().transcript) == 2
    await stub.stop()
    assert not stub.running
    assert not model.attached


async def test_disabled_dashboard_is_a_stub(bus: EventBus) -> None:
    surface = await create_dashboard(bus, None, None, enabled=False)

    assert isinstance(surface, StubDashboard)
    assert surface.model.snapshot().memory.status is MemoryStatus.UNAVAILABLE
    await surface.stop()


async def test_no_display_degrades_to_the_stub(
    bus: EventBus, monkeypatch: pytest.MonkeyPatch
) -> None:
    def no_display(*_args: object) -> None:
        raise tkinter.TclError("no display name and no $DISPLAY environment variable")

    logger = CapturingLogger()
    monkeypatch.setattr(dashboard_module, "_DashboardView", no_display)
    monkeypatch.setattr(dashboard_module, "log", logger)

    surface = await create_dashboard(bus, FakeMemoryInspector(), None)

    assert isinstance(surface, StubDashboard)
    assert surface.running
    assert ("warning", "dashboard.unavailable") in [
        (call.method_name, call.args[0]) for call in logger.calls
    ]
    await surface.stop()


class SlowView:
    """Stands in for a window that takes too long to appear, then behaves."""

    opened = threading.Event()

    def __init__(self, *_args: object) -> None:
        self.opened.wait(5)

    def run(self, _poll_ms: int, drain: Callable[[object], bool]) -> None:
        while drain(self):
            time.sleep(0.01)

    def destroy(self) -> None:
        """Nothing to destroy."""


async def test_window_that_never_opens_times_out(
    model: DashboardModel, monkeypatch: pytest.MonkeyPatch
) -> None:
    SlowView.opened.clear()
    monkeypatch.setattr(dashboard_module, "START_TIMEOUT_S", 0.05)
    monkeypatch.setattr(dashboard_module, "_DashboardView", SlowView)
    dashboard = Dashboard(model, asyncio.get_running_loop(), HIDDEN)

    with pytest.raises(DashboardError, match="in time"):
        await dashboard.start()
    SlowView.opened.set()
    await dashboard.stop()

    assert not dashboard.running


# -- loop-side plumbing without Tk -----------------------------------------------------------


async def test_drain_applies_commands_and_renders_only_the_newest_snapshot(
    bus: EventBus, model: DashboardModel
) -> None:
    dashboard = Dashboard(model, asyncio.get_running_loop(), HIDDEN)
    view = FakeView()
    model.attach()
    first = model.snapshot()
    _publish_turn(bus)
    newest = model.snapshot()
    for item in (first, _Command.SHOW, newest, _Command.HIDE):
        dashboard._inbox.put(item)

    keep_running = dashboard._drain(view)  # type: ignore[arg-type]

    assert keep_running
    assert view.calls == ["show", "hide", "render"]
    assert view.rendered == [newest]


async def test_drain_stops_on_the_stop_command(model: DashboardModel) -> None:
    dashboard = Dashboard(model, asyncio.get_running_loop(), HIDDEN)
    dashboard._inbox.put(_Command.STOP)

    assert not dashboard._drain(FakeView())  # type: ignore[arg-type]


async def test_refresh_button_runs_the_refresh_on_the_loop(model: DashboardModel) -> None:
    dashboard = Dashboard(model, asyncio.get_running_loop(), HIDDEN)
    button = threading.Thread(target=dashboard._request_refresh_from_ui)

    button.start()
    button.join()
    await asyncio.sleep(0)
    await model.close()

    assert model.snapshot().memory.status is MemoryStatus.READY


def test_tk_callback_errors_are_logged(monkeypatch: pytest.MonkeyPatch) -> None:
    logger = CapturingLogger()
    monkeypatch.setattr(dashboard_module, "log", logger)
    error = ValueError("bad widget")

    _log_tk_exception(ValueError, error, None)

    (call,) = logger.calls
    assert (call.method_name, call.args) == ("error", ("dashboard.tk.callback_failed",))
    assert call.kwargs["exc_info"] is error


# -- real Tk (skipped without a display) -------------------------------------------------------


@pytest.mark.usefixtures("display")
async def test_view_renders_a_snapshot(bus: EventBus, model: DashboardModel) -> None:
    model.attach()
    _publish_turn(bus)
    await model.refresh_memory()
    view = _DashboardView(HIDDEN, lambda: None)
    try:
        view.render(model.snapshot())
        transcript = view._transcript.get("1.0", "end")
        tools = [view._tools.item(row, "values") for row in view._tools.get_children()]
        latency = view._latency.get_children()
        memory = view._memory_text.get("1.0", "end")
        state = view._state_label.cget("text")
    finally:
        view.destroy()

    assert "You: open code" in transcript
    assert "Jarvis: Done." in transcript
    assert [values[1] for values in tools] == ["open_app"]
    assert len(latency) == 1
    assert "Recent facts" in memory
    assert state == "Listening"


@pytest.mark.usefixtures("display")
async def test_dashboard_runs_on_its_own_thread_and_stops_cleanly(
    bus: EventBus, model: DashboardModel
) -> None:
    dashboard = Dashboard(model, asyncio.get_running_loop(), HIDDEN)

    await dashboard.start()
    await dashboard.start()
    _publish_turn(bus)
    dashboard.show()
    dashboard.hide()
    await asyncio.sleep(0.3)
    assert dashboard.running
    assert model.attached

    await dashboard.stop()
    await dashboard.stop()

    assert not dashboard.running
    assert not model.attached
    assert not any(t.name == "jarvis-dashboard" for t in threading.enumerate())


@pytest.mark.usefixtures("display")
async def test_create_dashboard_opens_the_window(bus: EventBus) -> None:
    surface = await create_dashboard(bus, FakeMemoryInspector(), None, config=HIDDEN)

    assert isinstance(surface, Dashboard)
    await surface.stop()


def test_notebook_has_live_and_memory_tabs(display: None) -> None:
    view = _DashboardView(HIDDEN, lambda: None)
    try:
        notebook = next(w for w in view.root.winfo_children() if isinstance(w, ttk.Notebook))
        tabs = [notebook.tab(tab, "text") for tab in notebook.tabs()]
    finally:
        view.destroy()

    assert tabs == ["Live", "Memory"]


def test_tk_start_retries_a_transient_tcl_read_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    attempts: list[int] = []

    def flaky_tk() -> object:
        attempts.append(1)
        if len(attempts) == 1:
            raise tkinter.TclError('couldn\'t read file "init.tcl": No error')
        return "root"

    monkeypatch.setattr(dashboard_module.tkinter, "Tk", flaky_tk)
    monkeypatch.setattr(dashboard_module.time, "sleep", lambda _s: None)

    assert dashboard_module.create_tk_root() == "root"
    assert len(attempts) == 2


def test_tk_start_does_not_retry_a_missing_display(monkeypatch: pytest.MonkeyPatch) -> None:
    def no_display() -> object:
        raise tkinter.TclError("no display name and no $DISPLAY environment variable")

    monkeypatch.setattr(dashboard_module.tkinter, "Tk", no_display)

    with pytest.raises(tkinter.TclError, match="no display"):
        dashboard_module.create_tk_root()


def test_tk_start_retries_a_half_loaded_tcl_library(monkeypatch: pytest.MonkeyPatch) -> None:
    attempts: list[int] = []

    def flaky_tk() -> object:
        attempts.append(1)
        if len(attempts) == 1:
            raise tkinter.TclError('invalid command name "tcl_findLibrary"')
        return "root"

    monkeypatch.setattr(dashboard_module.tkinter, "Tk", flaky_tk)
    monkeypatch.setattr(dashboard_module.time, "sleep", lambda _s: None)

    assert dashboard_module.create_tk_root() == "root"
