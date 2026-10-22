"""What each action in the interactive HUD does (ADR-0024).

:class:`HudActions` implements the overlay bridge's ``HudHandlers``: one
small method per UI event. They contain no UI code and little logic; each
forwards to the component that owns the behaviour (the combined input for
typed requests, the app controls for quit and mute, the settings controller
for edits, the user settings store for the HUD's remembered position).
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Final, Protocol

from jarvis.core.ui_wiring import AppControls, MicrophoneSwitch
from jarvis.ui.hud_protocol import HistoryRole, PlacementChanged
from jarvis.ui.hud_settings import SettingsController
from jarvis.user_settings import UserSettingsError, UserSettingsStore
from jarvis.utils.logging import get_logger

log = get_logger(__name__)

POSITION_KEY: Final = "overlay_position"
EXPANDED_BOUNDS_KEY: Final = "overlay_expanded_bounds"
RESTART_UNAVAILABLE: Final = "Quit Jarvis and start it again to apply."
LOG_UNAVAILABLE: Final = "This session logs to the terminal, not to a file."


class HudOverlay(Protocol):
    """The slice of ``ui.overlay.Overlay`` the actions use."""

    def hide(self) -> None:
        """Hide the HUD."""

    def set_transcript(self, text: str) -> None:
        """Show what the user said."""

    def add_history(self, role: HistoryRole, text: str) -> None:
        """Append one conversation line."""


class TypedRequests(Protocol):
    """Where typed lines go (``core.combined_input.CombinedInput``)."""

    def submit(self, text: str) -> None:
        """Hand one typed request to the conversation."""


class HudActions:
    """``HudHandlers`` over the running session.

    Args:
        overlay: The HUD.
        typed: Receives typed requests.
        settings: The settings page's controller.
        store: Where the HUD's position is remembered.
        microphone: Mutes speech input when the app controls are not bound.
        open_log: Opens the log file; None when the session has none.
        restart: Arranges a relaunch after exit; None when that is impossible
            (a console session cannot reattach its terminal).
    """

    def __init__(
        self,
        *,
        overlay: HudOverlay,
        typed: TypedRequests,
        settings: SettingsController,
        store: UserSettingsStore,
        microphone: MicrophoneSwitch | None = None,
        open_log: Callable[[], None] | None = None,
        restart: Callable[[], None] | None = None,
    ) -> None:
        self._overlay = overlay
        self._typed = typed
        self._settings = settings
        self._store = store
        self._microphone = microphone
        self._open_log = open_log
        self._restart = restart
        self._controls: AppControls | None = None

    def bind_controls(self, controls: AppControls) -> None:
        """Use the tray's controls for quit and mute (built after the front end)."""
        self._controls = controls

    def submit(self, text: str) -> None:
        """Show the typed line like a spoken one, then hand it to the conversation."""
        self._overlay.set_transcript(text)
        self._overlay.add_history(HistoryRole.USER, text)
        self._typed.submit(text)

    def close(self) -> None:
        """Hide the HUD; the tray's "Show overlay" brings it back."""
        log.info("hud.close")
        self._overlay.hide()

    def quit(self) -> None:
        """End the session as the tray's Quit does."""
        if self._controls is None:
            log.error("hud.quit.unbound", reason="app controls are not bound yet")
            return
        self._controls.quit()

    async def toggle_mute(self) -> None:
        """Mute or unmute through the tray's controls, so the icon follows."""
        if self._controls is not None:
            self._controls.toggle_microphone()
            return
        if self._microphone is not None:
            await self._microphone.toggle()

    def expanded(self, expanded: bool) -> None:
        """Nothing to do but note it: the history is pushed continuously."""
        log.info("hud.expanded", expanded=expanded)

    async def setting_changed(self, key: str, value: str) -> None:
        """Validate, persist and apply one edit."""
        await self._settings.change(key, value)

    async def settings_opened(self) -> None:
        """Refresh lists that may have changed (installed Ollama models)."""
        await self._settings.refresh()

    def open_log(self) -> None:
        """Open the log file in the default viewer."""
        if self._open_log is None:
            self._settings.show_notice(LOG_UNAVAILABLE)
            return
        self._open_log()

    def restart(self) -> None:
        """Quit now and start again, for settings that apply after a restart."""
        if self._restart is None:
            self._settings.show_notice(RESTART_UNAVAILABLE)
            return
        log.info("hud.restart")
        self._restart()
        self.quit()

    async def placement_changed(self, event: PlacementChanged) -> None:
        """Remember where the HUD (or the expanded panel) now is."""
        key = EXPANDED_BOUNDS_KEY if event.expanded else POSITION_KEY
        value = _position_json(event) if event.custom else None
        try:
            await asyncio.to_thread(self._store.update, key, value)
        except (UserSettingsError, OSError):
            log.exception("hud.placement.not_saved", key=key)


def _position_json(event: PlacementChanged) -> dict[str, object]:
    position: dict[str, object] = {"monitor": event.monitor, "x": event.x, "y": event.y}
    if event.expanded:
        position.update(width=event.width, height=event.height)
    return position


# Design notes:
# - Quit and mute go through AppControls when it is bound, so the tray icon,
#   its tooltip and the HUD always agree on the mute state, and Quit runs the
#   same orderly shutdown as the tray's menu.
# - Restart is "remember to relaunch, then quit": the new process can only
#   take the single-instance lock once this one has released it, which is
#   what app.background does after its session ends.
# - A position that cannot be saved is logged, not raised: the HUD is where
#   the user put it for this session either way.
