"""What each action in the interactive HUD does (ADR-0024).

:class:`HudActions` implements the overlay bridge's ``HudHandlers``: one
small method per UI event. They contain no UI code and little logic; each
forwards to the component that owns the behaviour (the combined input for
typed requests, the app controls for quit and mute, the settings controller
for edits, the user settings store for the HUD's remembered position and
whether it was left minimized). :meth:`HudActions.toggle_minimized` is
what the tray's "Minimize HUD"/"Restore HUD" and the HUD hotkey call.
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
ORB_POSITION_KEY: Final = "hud_orb_position"
MINIMIZED_KEY: Final = "hud_minimized"
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

    @property
    def minimized(self) -> bool:
        """Whether the HUD is (or is shrinking into) the orb."""

    def minimize(self) -> None:
        """Shrink the HUD into the orb."""

    def restore(self) -> None:
        """Grow the orb back into the HUD."""


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
        """Use the tray's controls for quit and mute (built after the front end).

        The tray already exists then, so its "Minimize HUD"/"Restore HUD"
        entry is set from the HUD (which may have started minimized).
        """
        self._controls = controls
        self._show_minimized(self._overlay.minimized)

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
        """Remember where the HUD (or the expanded panel, or the orb) now is."""
        key = _placement_key(event)
        value = _position_json(event) if event.custom else None
        await self._remember(key, value)

    async def minimized(self, automatic: bool) -> None:
        """The HUD shrank to the orb; remember it unless auto-minimize did it."""
        await self._minimized_changed(True, automatic)

    async def restored(self, automatic: bool) -> None:
        """The orb grew back; remember it unless auto-minimize did it."""
        await self._minimized_changed(False, automatic)

    async def toggle_minimized(self) -> None:
        """Minimize or restore the HUD (the tray entry and the HUD hotkey)."""
        minimized = not self._overlay.minimized
        if minimized:
            self._overlay.minimize()
        else:
            self._overlay.restore()
        await self._minimized_changed(minimized, False)

    async def _minimized_changed(self, minimized: bool, automatic: bool) -> None:
        log.info("hud.minimized", minimized=minimized, automatic=automatic)
        self._show_minimized(minimized)
        if not automatic:
            await self._remember(MINIMIZED_KEY, minimized)

    def _show_minimized(self, minimized: bool) -> None:
        tray = self._controls.tray if self._controls is not None else None
        if tray is not None:
            tray.set_hud_minimized(minimized)

    async def _remember(self, key: str, value: object) -> None:
        try:
            await asyncio.to_thread(self._store.update, key, value)
        except (UserSettingsError, OSError):
            log.exception("hud.not_saved", key=key)


def _placement_key(event: PlacementChanged) -> str:
    if event.orb:
        return ORB_POSITION_KEY
    return EXPANDED_BOUNDS_KEY if event.expanded else POSITION_KEY


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
# - Only the user's own minimize and restore are remembered for the next
#   start. An orb made by auto-minimize is a passing state; starting the
#   next session minimized because of it would look like a bug.
