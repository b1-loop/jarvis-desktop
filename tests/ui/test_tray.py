import asyncio
import sys
import threading
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import pytest
from PIL import Image, ImageColor

from jarvis.core.events import EventBus, StateChanged
from jarvis.core.state_machine import AssistantState
from jarvis.ui import tray as tray_module
from jarvis.ui.tray import (
    ICON_SIZE,
    TOOLTIP_MAX_CHARS,
    StubTray,
    TrayAction,
    TrayError,
    TrayIcon,
    TrayMenuEntry,
    create_tray,
    pystray_backend_factory,
    render_state_icon,
    render_state_icons,
    with_notice,
)
from jarvis.ui.view_model import STATE_STYLES


class FakeBackend:
    """``TrayBackend`` that runs a blocking loop on the tray thread until stopped."""

    def __init__(
        self,
        entries: tuple[TrayMenuEntry, ...],
        sink: Callable[[TrayAction], None],
        image: Image.Image,
        tooltip: str,
    ) -> None:
        self.entries = entries
        self.sink = sink
        self.images = [image]
        self.tooltips = [tooltip]
        self.menu_refreshes = 0
        self.run_thread: threading.Thread | None = None
        self._stopped = threading.Event()

    def run(self, on_ready: Callable[[], None]) -> None:
        self.run_thread = threading.current_thread()
        on_ready()
        self._stopped.wait()

    def stop(self) -> None:
        self._stopped.set()

    def set_image(self, image: Image.Image) -> None:
        self.images.append(image)

    def set_tooltip(self, text: str) -> None:
        self.tooltips.append(text)

    def refresh_menu(self) -> None:
        self.menu_refreshes += 1


class FailingBackend(FakeBackend):
    def run(self, on_ready: Callable[[], None]) -> None:
        raise OSError("Shell_NotifyIcon failed")


class SilentBackend(FakeBackend):
    """Never becomes ready, like a shell that is not running."""

    def run(self, on_ready: Callable[[], None]) -> None:
        self._stopped.wait()


class FakeLoop:
    """Records what the tray hands to ``call_soon_threadsafe``."""

    def __init__(self) -> None:
        self.scheduled: list[Callable[[], object]] = []

    def call_soon_threadsafe(self, callback: Callable[[], object], /) -> object:
        self.scheduled.append(callback)
        return None


class RecordingCallbacks:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.threads: list[threading.Thread] = []

    def _record(self, name: str) -> None:
        self.calls.append(name)
        self.threads.append(threading.current_thread())

    def show_dashboard(self) -> None:
        self._record("show_dashboard")

    def toggle_overlay(self) -> None:
        self._record("toggle_overlay")

    def toggle_microphone(self) -> None:
        self._record("toggle_microphone")

    def quit(self) -> None:
        self._record("quit")


@pytest.fixture
def bus() -> EventBus:
    return EventBus()


@pytest.fixture
def callbacks() -> RecordingCallbacks:
    return RecordingCallbacks()


def _backend_of(tray: TrayIcon) -> FakeBackend:
    backend = tray._backend
    assert isinstance(backend, FakeBackend)
    return backend


def _changed(new: AssistantState) -> StateChanged:
    return StateChanged(old=AssistantState.IDLE, new=new, reason="test")


# -- icons ------------------------------------------------------------------------------


@pytest.mark.parametrize("state", list(AssistantState))
def test_icon_is_a_disc_in_the_state_colour(state: AssistantState) -> None:
    image = render_state_icon(state)
    centre = ICON_SIZE // 2

    assert image.size == (ICON_SIZE, ICON_SIZE)
    assert image.mode == "RGBA"
    assert image.getpixel((0, 0)) == (0, 0, 0, 0)
    assert image.getpixel((centre, centre)) == (
        *ImageColor.getrgb(STATE_STYLES[state].colour),
        255,
    )


def test_one_distinct_icon_per_state() -> None:
    icons = render_state_icons(size=16)

    assert set(icons) == set(AssistantState)
    assert len({icon.tobytes() for icon in icons.values()}) == len(AssistantState)


# -- TrayIcon ------------------------------------------------------------------------------


async def test_menu_lists_the_four_actions(bus: EventBus, callbacks: RecordingCallbacks) -> None:
    tray = TrayIcon(FakeBackend, bus, callbacks, FakeLoop())

    entries = _backend_of(tray).entries

    assert [entry.action for entry in entries] == list(TrayAction)
    assert [entry.default for entry in entries] == [True, False, False, False]


async def test_menu_actions_are_dispatched_to_the_loop(
    bus: EventBus, callbacks: RecordingCallbacks
) -> None:
    loop = FakeLoop()
    tray = TrayIcon(FakeBackend, bus, callbacks, loop)

    for action in TrayAction:
        tray.activate(action)

    assert callbacks.calls == []
    for scheduled in loop.scheduled:
        scheduled()
    assert callbacks.calls == ["show_dashboard", "toggle_overlay", "toggle_microphone", "quit"]


async def test_menu_click_on_the_tray_thread_runs_on_the_loop_thread(
    bus: EventBus, callbacks: RecordingCallbacks
) -> None:
    tray = TrayIcon(FakeBackend, bus, callbacks, asyncio.get_running_loop())
    clicker = threading.Thread(target=tray.activate, args=(TrayAction.QUIT,))

    clicker.start()
    clicker.join()
    await asyncio.sleep(0)

    assert callbacks.calls == ["quit"]
    assert callbacks.threads == [threading.current_thread()]


async def test_start_runs_the_backend_on_its_own_thread_and_follows_state(
    bus: EventBus, callbacks: RecordingCallbacks
) -> None:
    tray = TrayIcon(FakeBackend, bus, callbacks, FakeLoop())
    backend = _backend_of(tray)

    await tray.start()
    bus.publish(_changed(AssistantState.SPEAKING))

    assert tray.running
    assert backend.run_thread is not None
    assert backend.run_thread is not threading.current_thread()
    assert tray.state is AssistantState.SPEAKING
    assert backend.images[-1].tobytes() == render_state_icon(AssistantState.SPEAKING).tobytes()
    assert backend.tooltips[-1] == "Jarvis - Speaking"
    await tray.stop()


async def test_stop_joins_the_thread_and_unsubscribes(
    bus: EventBus, callbacks: RecordingCallbacks
) -> None:
    tray = TrayIcon(FakeBackend, bus, callbacks, FakeLoop())
    await tray.start()
    await tray.start()

    await tray.stop()
    await tray.stop()
    bus.publish(_changed(AssistantState.THINKING))

    assert not tray.running
    assert tray.state is AssistantState.IDLE


async def test_mute_updates_tooltip_and_check_mark(
    bus: EventBus, callbacks: RecordingCallbacks
) -> None:
    tray = TrayIcon(FakeBackend, bus, callbacks, FakeLoop())
    backend = _backend_of(tray)
    mute_entry = backend.entries[2]
    assert mute_entry.checked is not None
    assert not mute_entry.checked()

    tray.set_microphone_muted(True)

    assert tray.microphone_muted
    assert mute_entry.checked()
    assert backend.tooltips[-1] == "Jarvis - Idle (microphone muted)"
    assert backend.menu_refreshes == 1


async def test_backend_failure_surfaces_from_start(
    bus: EventBus, callbacks: RecordingCallbacks
) -> None:
    tray = TrayIcon(FailingBackend, bus, callbacks, FakeLoop())

    with pytest.raises(OSError, match="Shell_NotifyIcon"):
        await tray.start()


async def test_backend_that_never_shows_times_out(
    bus: EventBus, callbacks: RecordingCallbacks, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(tray_module, "START_TIMEOUT_S", 0.05)
    tray = TrayIcon(SilentBackend, bus, callbacks, FakeLoop())

    with pytest.raises(TrayError):
        await tray.start()
    await tray.stop()


# -- factory and stub -------------------------------------------------------------------


async def test_create_tray_returns_the_real_icon_when_it_starts(
    bus: EventBus, callbacks: RecordingCallbacks
) -> None:
    tray = await create_tray(bus, callbacks, backend_factory=FakeBackend)

    assert isinstance(tray, TrayIcon)
    assert tray.running
    await tray.stop()


async def test_create_tray_degrades_when_the_backend_fails(
    bus: EventBus, callbacks: RecordingCallbacks
) -> None:
    tray = await create_tray(bus, callbacks, backend_factory=FailingBackend)

    assert isinstance(tray, StubTray)
    assert tray.running


async def test_create_tray_degrades_without_pystray(
    bus: EventBus, callbacks: RecordingCallbacks
) -> None:
    def missing(*_args: Any) -> FakeBackend:
        raise ImportError("this platform is not supported")

    tray = await create_tray(bus, callbacks, backend_factory=missing)

    assert isinstance(tray, StubTray)


async def test_disabled_tray_is_a_stub(bus: EventBus, callbacks: RecordingCallbacks) -> None:
    tray = await create_tray(bus, callbacks, enabled=False, backend_factory=FakeBackend)

    assert isinstance(tray, StubTray)


async def test_stub_tracks_state_and_mute(bus: EventBus) -> None:
    stub = StubTray(bus)
    await stub.start()
    await stub.start()

    bus.publish(_changed(AssistantState.LISTENING))
    stub.set_microphone_muted(True)

    assert stub.state is AssistantState.LISTENING
    assert stub.microphone_muted
    assert stub.tooltip == "Jarvis - Listening (microphone muted)"
    await stub.stop()
    await stub.stop()
    assert not stub.running


async def test_notice_is_shown_under_the_state_and_cleared(
    bus: EventBus, callbacks: RecordingCallbacks
) -> None:
    tray = TrayIcon(FakeBackend, bus, callbacks, FakeLoop())
    backend = _backend_of(tray)

    tray.set_notice("Not listening: no microphone")
    assert backend.tooltips[-1] == "Jarvis - Idle\nNot listening: no microphone"

    tray.set_notice(None)
    assert backend.tooltips[-1] == "Jarvis - Idle"


def test_stub_records_the_notice(bus: EventBus) -> None:
    stub = StubTray(bus)

    stub.set_notice("LLM offline")

    assert stub.tooltip == "Jarvis - Idle\nLLM offline"


def test_long_notices_are_cut_to_what_windows_displays() -> None:
    text = with_notice("Jarvis - Idle", "x" * 500)

    assert len(text) == TOOLTIP_MAX_CHARS
    assert text.endswith("...")
    assert with_notice("Jarvis - Idle", None) == "Jarvis - Idle"


# -- pystray adapter ---------------------------------------------------------------------


class FakePystrayIcon:
    def __init__(self, name: str, icon: Image.Image, title: str, menu: Any) -> None:
        self.name, self.icon, self.title, self.menu = name, icon, title, menu
        self.visible = False
        self.stopped = False
        self.menu_updates = 0

    def run(self, setup: Callable[[Any], None]) -> None:
        setup(self)

    def stop(self) -> None:
        self.stopped = True

    def update_menu(self) -> None:
        self.menu_updates += 1


class FakeMenuItem:
    def __init__(self, text: str, action: Callable[..., None], **options: Any) -> None:
        self.text, self.action, self.options = text, action, options


@pytest.fixture
def fake_pystray(monkeypatch: pytest.MonkeyPatch) -> None:
    module = SimpleNamespace(
        Icon=FakePystrayIcon, Menu=lambda *items: list(items), MenuItem=FakeMenuItem
    )
    monkeypatch.setitem(sys.modules, "pystray", module)


@pytest.mark.usefixtures("fake_pystray")
def test_pystray_adapter_builds_menu_and_forwards_calls() -> None:
    clicked: list[TrayAction] = []
    muted = [False]
    entries = (
        TrayMenuEntry(TrayAction.SHOW_DASHBOARD, default=True),
        TrayMenuEntry(TrayAction.TOGGLE_MICROPHONE, checked=lambda: muted[0]),
    )
    image = render_state_icon(AssistantState.IDLE, size=16)
    ready: list[bool] = []

    backend = pystray_backend_factory(entries, clicked.append, image, "Jarvis - Idle")
    icon: FakePystrayIcon = backend._icon  # type: ignore[attr-defined]
    show, mute = icon.menu
    show.action(icon, show)
    muted[0] = True
    backend.run(lambda: ready.append(True))
    backend.set_tooltip("new")
    backend.set_image(image)
    backend.refresh_menu()
    backend.stop()

    assert [item.text for item in icon.menu] == ["Show dashboard", "Mute microphone"]
    assert show.options == {"checked": None, "default": True}
    assert mute.options["checked"](mute) is True
    assert clicked == [TrayAction.SHOW_DASHBOARD]
    assert icon.visible
    assert ready == [True]
    assert (icon.title, icon.menu_updates, icon.stopped) == ("new", 1, True)
