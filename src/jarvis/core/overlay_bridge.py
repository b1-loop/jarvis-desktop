"""Keeps the status overlay in step with the voice pipeline, and the HUD in step with Jarvis.

The overlay (ADR-0011) has cheap, thread-safe setters and never calls
back. This bridge is the only code that feeds it: state changes, the wake
pulse and the transcript arrive as bus events; response lines come from the
voice output as each sentence starts; and a small task samples the
microphone level while LISTENING and the playback level while SPEAKING at
about 20 Hz for the waveform.

The interactive HUD (ADR-0024) adds two directions. Every user line, reply
and tool call on the bus is appended to the expanded view's history, and a
second task drains the HUD's UI events every ~30 ms on the event loop and
dispatches each to a small handler (:class:`HudHandlers`, implemented in
``core/hud_wiring.py``). The native UI thread therefore never runs Python.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Callable
from typing import Final, Protocol

from jarvis.core.events import (
    AssistantSaid,
    EventBus,
    StateChanged,
    ToolCallFinished,
    TranscriptReady,
    Unsubscribe,
    WakeWordHeard,
)
from jarvis.core.state_machine import AssistantState
from jarvis.ui.hud_protocol import (
    CloseRequested,
    HistoryRole,
    Minimized,
    OpenLog,
    PlacementChanged,
    QuitRequested,
    RestartRequested,
    Restored,
    SetExpanded,
    SettingChanged,
    SettingsOpened,
    SubmitText,
    ToggleMute,
    UiEvent,
)
from jarvis.ui.overlay import OverlayState
from jarvis.utils.errors import JarvisError
from jarvis.utils.logging import get_logger

log = get_logger(__name__)

LEVEL_INTERVAL_SECONDS: Final = 0.05
"""20 Hz: smooth enough for a waveform, negligible work on the loop."""

UI_EVENT_INTERVAL_SECONDS: Final = 0.03
"""How often the HUD's UI events are drained; well under a perceptible delay."""

TOOL_LINE: Final = "{name} · {ms:.0f} ms"
TOOL_FAILED_LINE: Final = "{name} failed · {error}"

LevelSource = Callable[[], float]
"""Returns the current RMS level in ``[0, 1]``."""


class OverlayView(Protocol):
    """The slice of ``ui.overlay.Overlay`` the bridge drives."""

    def set_state(self, state: OverlayState) -> None:
        """Switch the indicator."""

    def push_level(self, level: float) -> None:
        """Append one waveform sample."""

    def set_transcript(self, text: str) -> None:
        """Show what the user said."""

    def set_response(self, text: str) -> None:
        """Show what Jarvis is saying."""

    def flash_wake_word(self) -> None:
        """Pulse the indicator."""

    def add_history(self, role: HistoryRole, text: str) -> None:
        """Append one line to the expanded view's conversation."""

    def poll_ui_events(self) -> list[UiEvent]:
        """What the user did in the HUD since the last poll."""


class HudHandlers(Protocol):
    """What each HUD event does; ``core/hud_wiring.HudActions`` is the real one."""

    def submit(self, text: str) -> None:
        """A typed request."""

    def close(self) -> None:
        """Hide the HUD."""

    def quit(self) -> None:
        """Quit Jarvis (confirmed in the HUD)."""

    async def toggle_mute(self) -> None:
        """Mute or unmute the microphone."""

    def expanded(self, expanded: bool) -> None:
        """The HUD expanded or collapsed."""

    async def setting_changed(self, key: str, value: str) -> None:
        """A setting was edited."""

    async def settings_opened(self) -> None:
        """The settings page opened."""

    def open_log(self) -> None:
        """Open the log file."""

    def restart(self) -> None:
        """Restart Jarvis."""

    async def placement_changed(self, event: PlacementChanged) -> None:
        """The HUD was moved or reset."""

    async def minimized(self, automatic: bool) -> None:
        """The HUD shrank to the orb (``automatic``: by auto-minimize)."""

    async def restored(self, automatic: bool) -> None:
        """The orb grew back into the HUD (``automatic``: by auto-minimize)."""


class OverlayBridge:
    """Subscribes to the bus, samples audio levels and dispatches HUD events.

    Args:
        overlay: The panel (or its stub).
        bus: Source of state, wake, transcript, reply and tool events.
        input_level: Microphone RMS, sampled while LISTENING.
        output_level: Playback RMS, sampled while SPEAKING.
        interval: Seconds between level samples.
        ui_interval: Seconds between drains of the HUD's UI events.
    """

    def __init__(
        self,
        overlay: OverlayView,
        bus: EventBus,
        *,
        input_level: LevelSource,
        output_level: LevelSource,
        interval: float = LEVEL_INTERVAL_SECONDS,
        ui_interval: float = UI_EVENT_INTERVAL_SECONDS,
    ) -> None:
        self._overlay = overlay
        self._bus = bus
        self._levels: dict[AssistantState, LevelSource] = {
            AssistantState.LISTENING: input_level,
            AssistantState.SPEAKING: output_level,
        }
        self._interval = interval
        self._ui_interval = ui_interval
        self._state = AssistantState.IDLE
        self._unsubscribe: list[Unsubscribe] = []
        self._tasks: list[asyncio.Task[None]] = []
        self._handlers: HudHandlers | None = None

    def attach_handlers(self, handlers: HudHandlers) -> None:
        """Route the HUD's events to ``handlers`` (built after the bridge)."""
        self._handlers = handlers

    def start(self) -> None:
        """Begin forwarding events, sampling levels and polling the HUD."""
        self._unsubscribe = [
            self._bus.subscribe_callback(StateChanged, self._on_state),
            self._bus.subscribe_callback(WakeWordHeard, self._on_wake),
            self._bus.subscribe_callback(TranscriptReady, self._on_transcript),
            self._bus.subscribe_callback(AssistantSaid, self._on_reply),
            self._bus.subscribe_callback(ToolCallFinished, self._on_tool),
        ]
        self._tasks = [
            asyncio.create_task(self._pump_levels(), name="overlay-levels"),
            asyncio.create_task(self._pump_ui_events(), name="overlay-ui-events"),
        ]

    async def aclose(self) -> None:
        """Stop forwarding. Idempotent."""
        for unsubscribe in self._unsubscribe:
            unsubscribe()
        self._unsubscribe = []
        for task in self._tasks:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        self._tasks = []

    def show_response(self, sentence: str) -> None:
        """Show the sentence Jarvis is starting to say."""
        self._overlay.set_response(sentence)

    async def drain_ui_events(self) -> int:
        """Dispatch everything queued in the HUD now; returns how many events."""
        events = self._overlay.poll_ui_events()
        for event in events:
            await self._dispatch(event)
        return len(events)

    def _on_state(self, event: StateChanged) -> None:
        self._state = event.new
        self._overlay.set_state(OverlayState[event.new.name])

    def _on_wake(self, _event: WakeWordHeard) -> None:
        self._overlay.flash_wake_word()

    def _on_transcript(self, event: TranscriptReady) -> None:
        self._overlay.set_transcript(event.text)
        if event.text.strip():
            self._overlay.add_history(HistoryRole.USER, event.text.strip())

    def _on_reply(self, event: AssistantSaid) -> None:
        if event.text.strip():
            self._overlay.add_history(HistoryRole.JARVIS, event.text.strip())

    def _on_tool(self, event: ToolCallFinished) -> None:
        line = (
            TOOL_LINE.format(name=event.name, ms=event.duration_ms)
            if event.ok
            else TOOL_FAILED_LINE.format(name=event.name, error=event.error or "error")
        )
        self._overlay.add_history(HistoryRole.TOOL, line)

    async def _pump_levels(self) -> None:
        while True:
            source = self._levels.get(self._state)
            if source is not None:
                self._overlay.push_level(source())
            await asyncio.sleep(self._interval)

    async def _pump_ui_events(self) -> None:
        while True:
            await self.drain_ui_events()
            await asyncio.sleep(self._ui_interval)

    async def _dispatch(self, event: UiEvent) -> None:
        handlers = self._handlers
        if handlers is None:
            log.info("overlay.ui_event.unhandled", kind=type(event).__name__)
            return
        try:
            await _route(handlers, event)
        except JarvisError:
            # One failed action (a shortcut that could not be written, a
            # microphone that would not reopen) must not stop the HUD from
            # answering the next click; it is logged with its traceback.
            log.exception("overlay.ui_event.failed", kind=type(event).__name__)


async def _route(handlers: HudHandlers, event: UiEvent) -> None:
    """Call the handler for ``event``."""
    match event:
        case SubmitText(text=text):
            handlers.submit(text)
        case CloseRequested():
            handlers.close()
        case QuitRequested():
            handlers.quit()
        case ToggleMute():
            await handlers.toggle_mute()
        case SetExpanded(expanded=expanded):
            handlers.expanded(expanded)
        case SettingChanged(key=key, value=value):
            await handlers.setting_changed(key, value)
        case SettingsOpened():
            await handlers.settings_opened()
        case OpenLog():
            handlers.open_log()
        case RestartRequested():
            handlers.restart()
        case PlacementChanged():
            await handlers.placement_changed(event)
        case Minimized(automatic=automatic):
            await handlers.minimized(automatic)
        case Restored(automatic=automatic):
            await handlers.restored(automatic)


# Design notes:
# - Bus callbacks rather than a subscription task: the overlay setters take
#   nanoseconds, so running them synchronously on publish is cheaper than
#   a queue hop and still never blocks the publisher.
# - The level pump runs only one source at a time, chosen by state, so the
#   waveform shows the user while they talk and Jarvis while it talks, and
#   stays flat when idle instead of drawing room noise.
# - UI events are drained by polling, not pushed: the native UI thread
#   cannot hold the GIL or wait for the event loop without dropping frames.
#   30 ms is below what a click-to-reaction delay feels like, and a poll
#   that finds nothing costs one lock and an empty list.
# - Typed lines reach the history through the HUD handler, spoken ones
#   through TranscriptReady, and replies through AssistantSaid, so the
#   expanded view shows one conversation whichever way it was held.
# - The tray icon and dashboard subscribe to the same events; they are
#   built for both front ends by bootstrap.build_ui (core/ui_wiring.py).
