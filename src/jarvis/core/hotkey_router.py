"""One reader of the platform's hotkey presses, shared by every hotkey Jarvis registers.

``Platform.poll_hotkeys`` returns *and clears* every press since the last
call. That is fine for one consumer, but the voice hotkey (start a turn) and
the HUD hotkey (minimize to the orb) poll on their own schedules, and
whichever polled first would swallow the other's presses. A
:class:`HotkeyRouter` is the only caller of the platform's poll; each
consumer gets a :class:`HotkeyView` that registers through the platform and
sees only the presses of the hotkeys it registered. A view has the same
three methods as the platform slice the voice trigger already takes, so
``HotkeyTrigger`` needs no change.
"""

from __future__ import annotations

from typing import Protocol

from jarvis.os_integration.base import HotkeyEvent


class HotkeySource(Protocol):
    """The slice of ``Platform`` that registers hotkeys and reports presses."""

    async def register_hotkey(self, spec: str) -> int:
        """Register ``spec`` system-wide and return its id."""

    async def unregister_hotkey(self, hotkey_id: int) -> None:
        """Release a registered hotkey."""

    def poll_hotkeys(self) -> list[HotkeyEvent]:
        """Presses since the previous poll (and clear them)."""


class HotkeyRouter:
    """Drains the platform's presses and hands each to the view that registered it.

    Args:
        platform: The OS backend (or a ``StubPlatform`` in tests).
    """

    def __init__(self, platform: HotkeySource) -> None:
        self._platform = platform
        self._owned: set[int] = set()
        self._pending: list[HotkeyEvent] = []

    def view(self) -> HotkeyView:
        """A new consumer: its own registrations, its own presses."""
        return HotkeyView(self)

    async def _register(self, spec: str) -> int:
        hotkey_id = await self._platform.register_hotkey(spec)
        self._owned.add(hotkey_id)
        return hotkey_id

    async def _unregister(self, hotkey_id: int) -> None:
        self._owned.discard(hotkey_id)
        self._pending = [press for press in self._pending if press.hotkey_id != hotkey_id]
        await self._platform.unregister_hotkey(hotkey_id)

    def _take(self, ids: set[int]) -> list[HotkeyEvent]:
        """Presses of ``ids`` in the order they happened; the others wait for their view."""
        # Presses of ids nobody registered through a view are dropped here,
        # so the pending list cannot grow without bound.
        self._pending.extend(
            press for press in self._platform.poll_hotkeys() if press.hotkey_id in self._owned
        )
        taken = [press for press in self._pending if press.hotkey_id in ids]
        self._pending = [press for press in self._pending if press.hotkey_id not in ids]
        return taken


class HotkeyView:
    """One consumer's window onto a :class:`HotkeyRouter`; a ``HotkeySource`` itself."""

    def __init__(self, router: HotkeyRouter) -> None:
        self._router = router
        self._ids: set[int] = set()

    async def register_hotkey(self, spec: str) -> int:
        """Register through the platform; raises what it raises (taken, invalid)."""
        hotkey_id = await self._router._register(spec)
        self._ids.add(hotkey_id)
        return hotkey_id

    async def unregister_hotkey(self, hotkey_id: int) -> None:
        """Release one of this view's hotkeys; other ids are ignored."""
        if hotkey_id not in self._ids:
            return
        self._ids.discard(hotkey_id)
        await self._router._unregister(hotkey_id)

    def poll_hotkeys(self) -> list[HotkeyEvent]:
        """Presses of this view's hotkeys since its previous poll."""
        return self._router._take(self._ids)


# Design notes:
# - A demultiplexer rather than one task fanning presses out to callbacks:
#   the voice trigger deliberately polls only while it waits for a turn
#   (and drops presses made while Jarvis was busy), the HUD hotkey polls
#   all the time. A view keeps that pull model for each of them.
# - The views reach into the router's underscored methods on purpose: the
#   pair is one unit, and the router's surface stays just view().
