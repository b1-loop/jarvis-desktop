"""The global hotkey that minimizes the HUD to its orb and restores it (``hud_toggle_hotkey``).

Registered through the Platform hotkey API like the voice hotkey (through
its own :class:`~jarvis.core.hotkey_router.HotkeyView`, so neither steals
the other's presses). The HUD can live without it, so every reason not to
have it is logged and skipped rather than raised: an empty setting, the
same combination as the voice hotkey (``hotkey``), or a combination
another program already owns.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Awaitable, Callable, Sequence
from typing import Final

from jarvis.core.hotkey_router import HotkeySource
from jarvis.os_integration.errors import PlatformError
from jarvis.utils.errors import JarvisError
from jarvis.utils.logging import get_logger

log = get_logger(__name__)

POLL_SECONDS: Final = 0.05
"""Matches the voice hotkey: below what a user perceives as lag."""

_ALIASES: Final = {
    "control": "ctrl",
    "super": "win",
    "cmd": "win",
    "return": "enter",
    "escape": "esc",
    "del": "delete",
}
"""Spellings ``native/os_integration`` accepts for the same key; the first is canonical."""

Action = Callable[[], Awaitable[None]]


def canonical_hotkey(spec: str) -> frozenset[str]:
    """The keys of ``spec`` regardless of order, case, spaces and aliases.

    ``"Control + Alt + H"`` and ``"alt+ctrl+h"`` give the same set, which is
    what decides whether two hotkeys collide.
    """
    tokens = (token.strip().lower() for token in spec.split("+"))
    return frozenset(_ALIASES.get(token, token) for token in tokens if token)


def same_hotkey(first: str, second: str) -> bool:
    """Whether the two specs name the same (non-empty) combination."""
    keys = canonical_hotkey(first)
    return bool(keys) and keys == canonical_hotkey(second)


class HudHotkey:
    """Calls ``action`` whenever the HUD hotkey is pressed.

    Args:
        hotkeys: Where to register and read presses (a ``HotkeyView``).
        spec: The combination, e.g. ``"ctrl+alt+h"``; empty disables it.
        action: Toggles the HUD (``HudActions.toggle_minimized``).
        reserved: Combinations it must not take (the voice hotkey).
        poll_interval: Seconds between polls of the press queue.
    """

    def __init__(
        self,
        hotkeys: HotkeySource,
        spec: str,
        action: Action,
        *,
        reserved: Sequence[str] = (),
        poll_interval: float = POLL_SECONDS,
    ) -> None:
        self._hotkeys = hotkeys
        self._spec = spec
        self._action = action
        self._reserved = tuple(reserved)
        self._poll_interval = poll_interval
        self._hotkey_id: int | None = None
        self._task: asyncio.Task[None] | None = None

    @property
    def active(self) -> bool:
        """True once registered and listening."""
        return self._task is not None

    async def start(self) -> bool:
        """Register the hotkey and start listening; False (logged) when skipped."""
        if not self._spec.strip():
            log.info("hud.hotkey.disabled")
            return False
        clash = next((other for other in self._reserved if same_hotkey(self._spec, other)), None)
        if clash is not None:
            log.warning("hud.hotkey.collides", hotkey=self._spec, other=clash)
            return False
        try:
            self._hotkey_id = await self._hotkeys.register_hotkey(self._spec)
        except PlatformError as error:
            # Invalid, or another program owns it: the tray and the HUD's own
            # controls still minimize and restore.
            log.warning("hud.hotkey.unavailable", hotkey=self._spec, error=str(error))
            return False
        self._task = asyncio.create_task(self._listen(), name="hud-hotkey")
        log.info("hud.hotkey.registered", hotkey=self._spec)
        return True

    async def aclose(self) -> None:
        """Stop listening and release the hotkey. Idempotent."""
        task, self._task = self._task, None
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        hotkey_id, self._hotkey_id = self._hotkey_id, None
        if hotkey_id is not None:
            await self._hotkeys.unregister_hotkey(hotkey_id)

    async def _listen(self) -> None:
        while True:
            for press in self._hotkeys.poll_hotkeys():
                if press.hotkey_id == self._hotkey_id:
                    await self._fire()
            await asyncio.sleep(self._poll_interval)

    async def _fire(self) -> None:
        try:
            await self._action()
        except JarvisError:
            # One failed toggle must not end the listener; it is logged.
            log.exception("hud.hotkey.action_failed", hotkey=self._spec)


# Design notes:
# - The collision check compares key sets, not strings, so "Alt+Ctrl+J"
#   is recognised as the voice hotkey "ctrl+alt+j". Windows would refuse
#   the second registration anyway (one process cannot register the same
#   combination twice), but the stub platform in tests would not, and the
#   log line names the actual reason instead of a Win32 error code.
# - Polling, not a callback, like every other native-to-Python path
#   (ADR-0004): the hotkey thread never takes the GIL.
