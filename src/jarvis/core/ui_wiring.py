"""The application actions behind the tray menu, and the tray/dashboard builders.

The tray (``ui/tray.py``) and dashboard (``ui/dashboard.py``) observe the
bus and never drive the pipeline. The tray's menu needs four actions that
only the composition root can perform (show the dashboard, toggle the
overlay, mute the microphone, quit); :class:`AppControls` implements them
over whatever the session actually has, so a text-mode session without an
overlay or microphone gets a logged no-op instead of an error.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from contextlib import AsyncExitStack
from typing import Protocol

from jarvis.config import Settings
from jarvis.core.events import EventBus
from jarvis.ui.dashboard import DashboardSurface, create_dashboard
from jarvis.ui.tray import TraySurface, create_tray
from jarvis.ui.view_model import MemoryInspector
from jarvis.utils.logging import get_logger
from jarvis.utils.timing import LatencyTracker

log = get_logger(__name__)


class MicrophoneSwitch(Protocol):
    """Mutes and unmutes speech input (voice mode only)."""

    async def toggle(self) -> bool:
        """Flip the mute state and return the new one (True = muted)."""


class AppControls:
    """``ui.tray.TrayCallbacks`` over the running session.

    Args:
        stop: Ends the session (``Application.request_stop``).
        dashboard: Shown from the menu; ``None`` when there is none.
        toggle_overlay: Shows/hides the overlay; ``None`` without one.
        microphone: Mutes speech input; ``None`` in text mode.
    """

    def __init__(
        self,
        *,
        stop: Callable[[], None],
        dashboard: DashboardSurface | None = None,
        toggle_overlay: Callable[[], None] | None = None,
        microphone: MicrophoneSwitch | None = None,
    ) -> None:
        self._stop = stop
        self._dashboard = dashboard
        self._toggle_overlay = toggle_overlay
        self._microphone = microphone
        self._tray: TraySurface | None = None
        self._tasks: set[asyncio.Task[None]] = set()

    def attach_tray(self, tray: TraySurface) -> None:
        """Let the controls reflect the mute state on the icon."""
        self._tray = tray

    def show_dashboard(self) -> None:
        """Bring the dashboard window up."""
        if self._dashboard is None:
            log.info("ui.dashboard.absent")
            return
        self._dashboard.show()

    def toggle_overlay(self) -> None:
        """Show or hide the status overlay."""
        if self._toggle_overlay is None:
            log.info("ui.overlay.absent")
            return
        self._toggle_overlay()

    def toggle_microphone(self) -> None:
        """Mute or unmute speech input; the device is touched off this callback."""
        if self._microphone is None:
            log.info("ui.microphone.absent")
            return
        task = asyncio.get_running_loop().create_task(self._flip_microphone(self._microphone))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    def quit(self) -> None:
        """End the session as Ctrl+C would, with full cleanup."""
        log.info("ui.quit_requested")
        self._stop()

    async def _flip_microphone(self, microphone: MicrophoneSwitch) -> None:
        muted = await microphone.toggle()
        log.info("ui.microphone.toggled", muted=muted)
        if self._tray is not None:
            self._tray.set_microphone_muted(muted)


async def build_dashboard(
    settings: Settings,
    bus: EventBus,
    stack: AsyncExitStack,
    *,
    memory: MemoryInspector | None,
) -> DashboardSurface:
    """The dashboard (or its stub), fed by a latency tracker on the bus."""
    tracker = LatencyTracker()
    stack.callback(tracker.attach(bus))
    dashboard = await create_dashboard(bus, memory, tracker, enabled=settings.dashboard_enabled)
    stack.push_async_callback(dashboard.stop)
    return dashboard


async def build_tray(
    settings: Settings, bus: EventBus, stack: AsyncExitStack, controls: AppControls
) -> TraySurface:
    """The tray icon (or its stub), with the menu routed to ``controls``."""
    tray = await create_tray(bus, controls, enabled=settings.tray_enabled)
    controls.attach_tray(tray)
    stack.push_async_callback(tray.stop)
    return tray


# Design notes:
# - Microphone toggling runs as a task because the tray calls back
#   synchronously on the loop, while muting stops or restarts the audio
#   device off the loop. Tasks are kept referenced until done.
# - The tracker is attached before the dashboard starts so the first
#   turn's latency samples are not missed.
