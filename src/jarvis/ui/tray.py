"""System tray icon reflecting the assistant state.

The icon is a coloured disc, one colour per ``AssistantState`` in the same
palette as the native overlay, with a tooltip naming the state. Its menu
(Show dashboard, Toggle overlay, Mute microphone, Quit) contains no logic:
each entry hops back onto the asyncio loop with ``call_soon_threadsafe``
and calls one method of the injected :class:`TrayCallbacks`. The tray only
observes the bus; it never publishes and never drives the pipeline.

``pystray`` picks its platform backend when it is imported, which fails on
a headless Linux runner, so it is imported lazily inside
:func:`pystray_backend_factory`. Everything else here imports anywhere.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import importlib
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Final, Protocol

from PIL import Image, ImageDraw

from jarvis.core.events import EventBus, StateChanged, Unsubscribe
from jarvis.core.state_machine import AssistantState
from jarvis.ui.view_model import BACKGROUND_COLOUR, state_style, tray_tooltip
from jarvis.utils.errors import JarvisError
from jarvis.utils.logging import get_logger

log = get_logger(__name__)

ICON_SIZE: Final = 64
"""Rendered size in pixels; Windows downsamples to the 16-32 px it shows."""
_RING_FRACTION: Final = 0.08
_DISC_INSET_FRACTION: Final = 0.2
START_TIMEOUT_S: Final = 5.0
STOP_TIMEOUT_S: Final = 5.0
_APP_NAME: Final = "jarvis"


class TrayError(JarvisError):
    """The tray icon could not be created or did not start in time."""


class TrayAction(StrEnum):
    """What a menu entry asks the application to do."""

    SHOW_DASHBOARD = "Show dashboard"
    TOGGLE_OVERLAY = "Toggle overlay"
    TOGGLE_MICROPHONE = "Mute microphone"
    QUIT = "Quit"


class TrayCallbacks(Protocol):
    """Application actions behind the menu; every method runs on the asyncio loop."""

    def show_dashboard(self) -> None:
        """Bring the dashboard window to the front."""

    def toggle_overlay(self) -> None:
        """Show the overlay if hidden, hide it if shown."""

    def toggle_microphone(self) -> None:
        """Mute or unmute the microphone; report back with ``set_microphone_muted``."""

    def quit(self) -> None:
        """Shut the application down."""


class LoopScheduler(Protocol):
    """The one method of ``asyncio.AbstractEventLoop`` the tray needs."""

    def call_soon_threadsafe(self, callback: Callable[[], object], /) -> object:
        """Run ``callback`` on the loop's thread soon."""


@dataclass(frozen=True, slots=True)
class TrayMenuEntry:
    """One menu entry, toolkit-neutral.

    Attributes:
        action: What it does; its value is the label.
        checked: Evaluated by the backend when the menu is drawn; ``None``
            for entries without a check mark.
        default: The entry a double-click on the icon triggers.
    """

    action: TrayAction
    checked: Callable[[], bool] | None = None
    default: bool = False


class TrayBackend(Protocol):
    """What :class:`TrayIcon` needs from a tray implementation (pystray in practice)."""

    def run(self, on_ready: Callable[[], None]) -> None:
        """Show the icon and block in its message loop; call ``on_ready`` once shown."""

    def stop(self) -> None:
        """End the message loop from any thread."""

    def set_image(self, image: Image.Image) -> None:
        """Replace the icon image."""

    def set_tooltip(self, text: str) -> None:
        """Replace the tooltip."""

    def refresh_menu(self) -> None:
        """Re-evaluate dynamic menu properties such as check marks."""


BackendFactory = Callable[
    [tuple[TrayMenuEntry, ...], Callable[[TrayAction], None], Image.Image, str], TrayBackend
]
"""Builds a backend from the menu, the action sink, the first image and tooltip."""


def render_state_icon(state: AssistantState, size: int = ICON_SIZE) -> Image.Image:
    """Draw the tray image for ``state``: a dark ring around a coloured disc.

    The dark ring keeps the disc visible on both light and dark taskbars.
    """
    image = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    ring = max(round(size * _RING_FRACTION), 1)
    inset = round(size * _DISC_INSET_FRACTION)
    draw.ellipse((0, 0, size - 1, size - 1), fill=BACKGROUND_COLOUR)
    draw.ellipse((ring, ring, size - 1 - ring, size - 1 - ring), fill=state_style(state).colour)
    draw.ellipse(
        (inset, inset, size - 1 - inset, size - 1 - inset), outline=BACKGROUND_COLOUR, width=ring
    )
    return image


def render_state_icons(size: int = ICON_SIZE) -> dict[AssistantState, Image.Image]:
    """Pre-render one image per state so a transition only swaps a reference."""
    return {state: render_state_icon(state, size) for state in AssistantState}


class TrayIcon:
    """Keeps a tray backend in sync with the bus and routes menu clicks to the loop.

    Args:
        backend_factory: Builds the platform backend; see
            :func:`pystray_backend_factory`.
        bus: Event bus to observe for ``StateChanged``.
        callbacks: Application actions behind the menu.
        loop: Loop that the callbacks run on.
    """

    def __init__(
        self,
        backend_factory: BackendFactory,
        bus: EventBus,
        callbacks: TrayCallbacks,
        loop: LoopScheduler,
    ) -> None:
        self._bus = bus
        self._callbacks = callbacks
        self._loop = loop
        self._icons = render_state_icons()
        self._state = AssistantState.IDLE
        self._muted = False
        self._actions: Mapping[TrayAction, Callable[[], None]] = {
            TrayAction.SHOW_DASHBOARD: callbacks.show_dashboard,
            TrayAction.TOGGLE_OVERLAY: callbacks.toggle_overlay,
            TrayAction.TOGGLE_MICROPHONE: callbacks.toggle_microphone,
            TrayAction.QUIT: callbacks.quit,
        }
        self._backend = backend_factory(
            self.menu_entries(), self.activate, self._icons[self._state], self.tooltip
        )
        self._thread: threading.Thread | None = None
        self._unsubscribe: Unsubscribe | None = None

    @property
    def state(self) -> AssistantState:
        """The state the icon currently shows."""
        return self._state

    @property
    def tooltip(self) -> str:
        """The tooltip the icon currently shows."""
        return tray_tooltip(self._state, microphone_muted=self._muted)

    @property
    def microphone_muted(self) -> bool:
        """Whether the menu shows the microphone as muted."""
        return self._muted

    @property
    def running(self) -> bool:
        """True while the tray thread is alive."""
        return self._thread is not None and self._thread.is_alive()

    def menu_entries(self) -> tuple[TrayMenuEntry, ...]:
        """The menu, top to bottom."""
        return (
            TrayMenuEntry(TrayAction.SHOW_DASHBOARD, default=True),
            TrayMenuEntry(TrayAction.TOGGLE_OVERLAY),
            TrayMenuEntry(TrayAction.TOGGLE_MICROPHONE, checked=lambda: self._muted),
            TrayMenuEntry(TrayAction.QUIT),
        )

    async def start(self) -> None:
        """Subscribe to state changes and run the backend on its own thread.

        Raises:
            TrayError: If the backend fails or is not shown within
                ``START_TIMEOUT_S``.
        """
        if self._thread is not None:
            return
        ready: concurrent.futures.Future[None] = concurrent.futures.Future()
        self._thread = threading.Thread(
            target=self._run_backend, args=(ready,), name="jarvis-tray", daemon=True
        )
        self._thread.start()
        try:
            # Shielded so a timeout does not cancel ``ready`` under the UI thread,
            # which would make its later set_result raise InvalidStateError.
            await asyncio.wait_for(asyncio.shield(asyncio.wrap_future(ready)), START_TIMEOUT_S)
        except TimeoutError as error:
            self._backend.stop()
            raise TrayError("tray icon did not start in time") from error
        self._unsubscribe = self._bus.subscribe_callback(StateChanged, self._on_state_changed)
        log.info("tray.started")

    async def stop(self) -> None:
        """Remove the icon and join the tray thread without blocking the loop. Idempotent."""
        if self._unsubscribe is not None:
            self._unsubscribe()
            self._unsubscribe = None
        thread, self._thread = self._thread, None
        if thread is None:
            return
        self._backend.stop()
        await asyncio.to_thread(thread.join, STOP_TIMEOUT_S)
        if thread.is_alive():
            log.warning("tray.stop.timeout", timeout_s=STOP_TIMEOUT_S)
        log.info("tray.stopped")

    def set_microphone_muted(self, muted: bool) -> None:
        """Reflect the real microphone state, after the application changed it."""
        self._muted = muted
        self._backend.set_tooltip(self.tooltip)
        self._backend.refresh_menu()

    def activate(self, action: TrayAction) -> None:
        """Menu sink, called on the tray thread: forward ``action`` to the loop."""
        log.debug("tray.menu", action=action.value)
        self._loop.call_soon_threadsafe(self._actions[action])

    def _on_state_changed(self, event: StateChanged) -> None:
        self._state = event.new
        self._backend.set_image(self._icons[event.new])
        self._backend.set_tooltip(self.tooltip)

    def _run_backend(self, ready: concurrent.futures.Future[None]) -> None:
        """Tray thread body: the backend's message loop, with failures reported."""
        try:
            self._backend.run(lambda: _resolve(ready))
        except Exception as error:
            if ready.done():
                log.exception("tray.backend.crashed")
            else:
                ready.set_exception(error)


def _resolve(ready: concurrent.futures.Future[None]) -> None:
    if not ready.done():
        ready.set_result(None)


class StubTray:
    """Stand-in for :class:`TrayIcon` when there is no tray (headless, text mode).

    Tracks the state it would show and logs transitions at debug level so a
    terminal session can still see what the icon would have displayed.
    Nothing is drawn; this is a stub, not a fallback renderer.
    """

    def __init__(self, bus: EventBus) -> None:
        self._bus = bus
        self._state = AssistantState.IDLE
        self._muted = False
        self._unsubscribe: Unsubscribe | None = None

    @property
    def state(self) -> AssistantState:
        """The state the icon would show."""
        return self._state

    @property
    def tooltip(self) -> str:
        """The tooltip the icon would show."""
        return tray_tooltip(self._state, microphone_muted=self._muted)

    @property
    def microphone_muted(self) -> bool:
        """The last value passed to :meth:`set_microphone_muted`."""
        return self._muted

    @property
    def running(self) -> bool:
        """True between :meth:`start` and :meth:`stop`."""
        return self._unsubscribe is not None

    async def start(self) -> None:
        """Begin tracking state changes."""
        if self._unsubscribe is None:
            self._unsubscribe = self._bus.subscribe_callback(StateChanged, self._on_state_changed)

    async def stop(self) -> None:
        """Stop tracking. Idempotent."""
        if self._unsubscribe is not None:
            self._unsubscribe()
            self._unsubscribe = None

    def set_microphone_muted(self, muted: bool) -> None:
        """Record the microphone state."""
        self._muted = muted

    def _on_state_changed(self, event: StateChanged) -> None:
        self._state = event.new
        log.debug("tray.stub.state", state=event.new.value)


class TraySurface(Protocol):
    """What the composition root holds: a :class:`TrayIcon` or a :class:`StubTray`."""

    @property
    def state(self) -> AssistantState:
        """The state the icon shows."""

    @property
    def running(self) -> bool:
        """Whether the tray is active."""

    async def start(self) -> None:
        """Show the icon."""

    async def stop(self) -> None:
        """Remove the icon."""

    def set_microphone_muted(self, muted: bool) -> None:
        """Reflect the microphone state."""


class _PystrayBackend:
    """:class:`TrayBackend` over a ``pystray.Icon``."""

    def __init__(self, icon: Any) -> None:
        self._icon = icon

    def run(self, on_ready: Callable[[], None]) -> None:
        def setup(icon: Any) -> None:
            icon.visible = True
            on_ready()

        self._icon.run(setup=setup)

    def stop(self) -> None:
        self._icon.stop()

    def set_image(self, image: Image.Image) -> None:
        self._icon.icon = image

    def set_tooltip(self, text: str) -> None:
        self._icon.title = text

    def refresh_menu(self) -> None:
        self._icon.update_menu()


def pystray_backend_factory(
    entries: tuple[TrayMenuEntry, ...],
    sink: Callable[[TrayAction], None],
    image: Image.Image,
    tooltip: str,
) -> TrayBackend:
    """Build a pystray icon; pystray is imported here, not at module import.

    Raises:
        ImportError: If pystray is not installed or has no backend for this
            desktop (e.g. a headless Linux session).
    """
    pystray = importlib.import_module("pystray")
    items = [_pystray_item(pystray, entry, sink) for entry in entries]
    icon = pystray.Icon(_APP_NAME, image, tooltip, menu=pystray.Menu(*items))
    return _PystrayBackend(icon)


def _pystray_item(pystray: Any, entry: TrayMenuEntry, sink: Callable[[TrayAction], None]) -> Any:
    checked = entry.checked
    return pystray.MenuItem(
        entry.action.value,
        lambda _icon, _item: sink(entry.action),
        checked=(lambda _item: checked()) if checked is not None else None,
        default=entry.default,
    )


async def create_tray(
    bus: EventBus,
    callbacks: TrayCallbacks,
    *,
    enabled: bool = True,
    backend_factory: BackendFactory = pystray_backend_factory,
) -> TraySurface:
    """Start the real tray icon when possible, otherwise a :class:`StubTray`.

    Must be awaited on the loop the callbacks should run on. A missing
    pystray, an unsupported desktop or a backend that fails to start logs a
    warning and the assistant carries on without an icon.
    """
    if enabled:
        try:
            tray = TrayIcon(backend_factory, bus, callbacks, asyncio.get_running_loop())
            await tray.start()
        except (ImportError, OSError, TrayError) as error:
            log.warning("tray.unavailable", reason=str(error))
        else:
            return tray
    stub = StubTray(bus)
    await stub.start()
    return stub


# Design notes:
# - The backend sits behind a Protocol so the TrayIcon's real logic (state
#   to image, menu to loop) is unit-tested with a fake on any OS, and the
#   pystray adapter stays a dozen lines that only a desktop can exercise.
# - The tray thread is a daemon: pystray's message loop is ended with
#   stop(), but if a shell hang ever kept it alive, a non-daemon thread
#   would keep the process alive after Quit.
# - set_image and set_tooltip run on the event loop thread. pystray's Win32
#   backend turns them into Shell_NotifyIcon calls against the icon's
#   window handle, which are safe from any thread; nothing is redrawn in
#   Python except once per state change.
# - Mute is shown, not owned: the menu entry asks the application to toggle
#   the microphone and the application reports the result through
#   set_microphone_muted, so the check mark can never disagree with the
#   audio engine.
# - Icons are pre-rendered at construction (four 64x64 images), so a state
#   transition costs a reference swap, not a Pillow draw on the loop.
