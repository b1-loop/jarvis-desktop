"""Keeps the status overlay in step with the voice pipeline.

The overlay (ADR-0011) has cheap, thread-safe setters and never calls
back. This bridge is the only code that feeds it: state changes, the wake
pulse and the transcript arrive as bus events; response lines come from the
voice output as each sentence starts; and a small task samples the
microphone level while LISTENING and the playback level while SPEAKING at
about 20 Hz for the waveform.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Callable
from typing import Final, Protocol

from jarvis.core.events import EventBus, StateChanged, TranscriptReady, Unsubscribe, WakeWordHeard
from jarvis.core.state_machine import AssistantState
from jarvis.ui.overlay import OverlayState
from jarvis.utils.logging import get_logger

log = get_logger(__name__)

LEVEL_INTERVAL_SECONDS: Final = 0.05
"""20 Hz: smooth enough for a waveform, negligible work on the loop."""

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


class OverlayBridge:
    """Subscribes to the bus and samples audio levels for the overlay.

    Args:
        overlay: The panel (or its stub).
        bus: Source of state, wake and transcript events.
        input_level: Microphone RMS, sampled while LISTENING.
        output_level: Playback RMS, sampled while SPEAKING.
        interval: Seconds between level samples.
    """

    def __init__(
        self,
        overlay: OverlayView,
        bus: EventBus,
        *,
        input_level: LevelSource,
        output_level: LevelSource,
        interval: float = LEVEL_INTERVAL_SECONDS,
    ) -> None:
        self._overlay = overlay
        self._bus = bus
        self._levels: dict[AssistantState, LevelSource] = {
            AssistantState.LISTENING: input_level,
            AssistantState.SPEAKING: output_level,
        }
        self._interval = interval
        self._state = AssistantState.IDLE
        self._unsubscribe: list[Unsubscribe] = []
        self._pump: asyncio.Task[None] | None = None

    def start(self) -> None:
        """Begin forwarding events and sampling levels."""
        self._unsubscribe = [
            self._bus.subscribe_callback(StateChanged, self._on_state),
            self._bus.subscribe_callback(WakeWordHeard, self._on_wake),
            self._bus.subscribe_callback(TranscriptReady, self._on_transcript),
        ]
        self._pump = asyncio.create_task(self._pump_levels(), name="overlay-levels")

    async def aclose(self) -> None:
        """Stop forwarding. Idempotent."""
        for unsubscribe in self._unsubscribe:
            unsubscribe()
        self._unsubscribe = []
        if self._pump is not None:
            self._pump.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._pump
            self._pump = None

    def show_response(self, sentence: str) -> None:
        """Show the sentence Jarvis is starting to say."""
        self._overlay.set_response(sentence)

    def _on_state(self, event: StateChanged) -> None:
        self._state = event.new
        self._overlay.set_state(OverlayState[event.new.name])

    def _on_wake(self, _event: WakeWordHeard) -> None:
        self._overlay.flash_wake_word()

    def _on_transcript(self, event: TranscriptReady) -> None:
        self._overlay.set_transcript(event.text)

    async def _pump_levels(self) -> None:
        while True:
            source = self._levels.get(self._state)
            if source is not None:
                self._overlay.push_level(source())
            await asyncio.sleep(self._interval)


# Design notes:
# - Bus callbacks rather than a subscription task: the overlay setters take
#   nanoseconds, so running them synchronously on publish is cheaper than
#   a queue hop and still never blocks the publisher.
# - The level pump runs only one source at a time, chosen by state, so the
#   waveform shows the user while they talk and Jarvis while it talks, and
#   stays flat when idle instead of drawing room noise.
# - The tray icon and dashboard subscribe to the same events; they are
#   built for both front ends by bootstrap.build_ui (core/ui_wiring.py).
