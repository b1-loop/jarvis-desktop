"""What starts a voice turn: the wake word, the global hotkey, or either.

Each trigger is a small object with one coroutine, ``wait()``, that returns
when the user asked for attention. :class:`AnyTrigger` races several and
cancels the losers, so the voice input does not care which sources exist.
That is what makes the degradation ladder cheap: without wake-word models
the hotkey trigger alone is passed in, and nothing else changes.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final, Protocol

from jarvis.audio.engine import AudioEvent, AudioEventKind
from jarvis.os_integration.base import HotkeyEvent
from jarvis.utils.logging import get_logger

log = get_logger(__name__)

HOTKEY_POLL_SECONDS: Final = 0.05
"""Matches ``Platform.hotkey_events``: below what a user perceives as lag."""

HOTKEY_SCORE: Final = 1.0
"""A key press is unambiguous; it reports full confidence like ``WakeWordHeard``."""


@dataclass(frozen=True, slots=True)
class TriggerHit:
    """Why a turn started.

    Attributes:
        source: ``"wake word"`` or ``"hotkey"``; also used as the state-change reason.
        score: Detector confidence, ``1.0`` for the hotkey.
    """

    source: str
    score: float


class TurnTrigger(Protocol):
    """Something that can ask Jarvis to start listening."""

    async def wait(self) -> TriggerHit:
        """Return the next time the user asks for attention."""


class WakeWordEvents(Protocol):
    """The slice of ``AudioEngine`` the wake-word trigger needs."""

    async def next_event(self, *kinds: AudioEventKind) -> AudioEvent:
        """Wait for the next event of one of ``kinds``."""


class HotkeyPlatform(Protocol):
    """The slice of ``Platform`` the hotkey trigger needs."""

    async def register_hotkey(self, spec: str) -> int:
        """Register ``spec`` system-wide and return its id."""

    async def unregister_hotkey(self, hotkey_id: int) -> None:
        """Release a registered hotkey."""

    def poll_hotkeys(self) -> list[HotkeyEvent]:
        """Presses since the previous poll."""


class WakeWordTrigger:
    """Fires on ``WAKE_WORD_DETECTED`` from the audio engine."""

    def __init__(self, engine: WakeWordEvents) -> None:
        self._engine = engine

    async def wait(self) -> TriggerHit:
        """Wait for the engine to hear the wake word."""
        event = await self._engine.next_event(AudioEventKind.WAKE_WORD_DETECTED)
        return TriggerHit("wake word", event.value)


class HotkeyTrigger:
    """Fires when the configured global hotkey is pressed.

    Args:
        platform: Registers the hotkey and reports presses.
        spec: Combination such as ``"ctrl+alt+j"`` (``Settings.hotkey``).
        poll_interval: Seconds between polls of the native press queue.
    """

    def __init__(
        self, platform: HotkeyPlatform, spec: str, *, poll_interval: float = HOTKEY_POLL_SECONDS
    ) -> None:
        self._platform = platform
        self._spec = spec
        self._poll_interval = poll_interval
        self._hotkey_id: int | None = None

    @property
    def spec(self) -> str:
        """The combination this trigger listens for."""
        return self._spec

    async def register(self) -> None:
        """Claim the hotkey system-wide.

        Raises:
            InvalidHotkeyError: If the spec cannot be parsed.
            PlatformError: If another application owns the combination.
        """
        self._hotkey_id = await self._platform.register_hotkey(self._spec)
        log.info("voice.hotkey.registered", hotkey=self._spec)

    async def aclose(self) -> None:
        """Release the hotkey. Idempotent."""
        if self._hotkey_id is None:
            return
        await self._platform.unregister_hotkey(self._hotkey_id)
        self._hotkey_id = None

    async def wait(self) -> TriggerHit:
        """Wait for a fresh press; presses made while Jarvis was busy are dropped."""
        if self._hotkey_id is None:
            raise RuntimeError("HotkeyTrigger.wait() called before register()")
        self._ours(self._platform.poll_hotkeys())  # discard presses made while busy
        while True:
            if self._ours(self._platform.poll_hotkeys()):
                return TriggerHit("hotkey", HOTKEY_SCORE)
            # Native presses are only visible by polling (ADR-0004).
            await asyncio.sleep(self._poll_interval)

    def _ours(self, presses: list[HotkeyEvent]) -> bool:
        return any(press.hotkey_id == self._hotkey_id for press in presses)


class AnyTrigger:
    """Races several triggers; the first to fire wins and the others are cancelled."""

    def __init__(self, triggers: Sequence[TurnTrigger]) -> None:
        if not triggers:
            raise ValueError("AnyTrigger needs at least one trigger")
        self._triggers = tuple(triggers)

    async def wait(self) -> TriggerHit:
        """Return the first hit; an exception from any trigger propagates."""
        if len(self._triggers) == 1:
            return await self._triggers[0].wait()
        waiters = [asyncio.create_task(trigger.wait()) for trigger in self._triggers]
        try:
            done, _ = await asyncio.wait(waiters, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for waiter in waiters:
                waiter.cancel()
            await asyncio.gather(*waiters, return_exceptions=True)
        return next(iter(done)).result()


# Design notes:
# - Polling, not callbacks, for the hotkey too: the native hotkey thread
#   never takes the GIL (ADR-0004). One poll every 50 ms costs nothing.
# - HotkeyTrigger.wait() drops presses made while Jarvis was thinking or
#   speaking. Acting on them later would start a turn the user may no
#   longer want; interrupting Jarvis is what barge-in (speaking) is for.
# - AnyTrigger cancels the losing waiters instead of keeping them alive
#   across turns: asyncio.Queue.get is cancellation-safe, so a wake-word
#   waiter that lost the race never swallows an event.
# - The ``gather(..., return_exceptions=True)`` only collects the waiters
#   that were just cancelled; the winner's own exception, if any, is
#   re-raised by ``result()``.
