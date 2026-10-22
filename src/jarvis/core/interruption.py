"""Barge-in: stop speaking the moment the user starts talking.

The stop itself happens natively: when a barge-in target is set, the audio
engine's worker thread calls ``Playback::stop()`` on the VAD's
``SpeechStarted`` without waiting for Python (ADR-0003). This module
decides *when* that target is armed and what happens in Python afterwards:

1. :meth:`BargeInController.playback_started` (called by the voice output
   when the first audio of an answer reaches the device) disarms the
   engine and re-arms it only after an echo guard window, so the burst of
   Jarvis's own voice leaking from the speakers into the microphone cannot
   stop it. Speech must also persist for the segmenter's ``minSpeechMs``
   before the engine reports it at all.
2. :meth:`BargeInController.playback_interrupted` (the voice output saw
   playback stopped under it) starts retaining the user's audio at once,
   cancels the agent turn and moves the state machine SPEAKING ->
   LISTENING, so the next request is the words the user is saying now.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Final, Protocol

from jarvis.audio.engine import NativePlayback
from jarvis.core.state_machine import AssistantState, StateMachine
from jarvis.utils.errors import JarvisError
from jarvis.utils.logging import get_logger

log = get_logger(__name__)

DEFAULT_ECHO_GUARD_SECONDS: Final = 0.4
"""Mirrors ``Settings.barge_in_echo_guard_ms``."""


class BargeInNotReadyError(JarvisError):
    """Playback was interrupted before :meth:`BargeInController.attach` ran."""


class BargeInTarget(Protocol):
    """The engine side: who to stop natively on speech onset."""

    def enable_barge_in(self, playback: NativePlayback | None) -> None:
        """Arm with ``playback``, or disarm with ``None``."""


class TurnCanceller(Protocol):
    """The orchestrator side: abandon the answer being produced."""

    def cancel_current_turn(self) -> bool:
        """Cancel the running turn; True if one was running."""


class BargeInCapture(Protocol):
    """The input side: keep the words the user is saying right now."""

    def begin_barge_in_capture(self) -> None:
        """Start retaining audio and make the next request skip the wake word."""


class BargeInController:
    """Arms native barge-in while Jarvis speaks and turns an interruption into a new turn.

    Implements the voice output's ``PlaybackObserver`` Protocol. The
    orchestrator and the voice input exist only after this object (they
    depend on the sink it observes), so they are connected with
    :meth:`attach` once built.

    Args:
        engine: Where the native barge-in target is set.
        playback: The playback the engine should stop.
        state: The assistant state machine, owned by the orchestrator.
        echo_guard_seconds: Speech onset during this window after playback
            starts is ignored.
    """

    def __init__(
        self,
        *,
        engine: BargeInTarget,
        playback: NativePlayback,
        state: StateMachine,
        echo_guard_seconds: float = DEFAULT_ECHO_GUARD_SECONDS,
    ) -> None:
        self._engine = engine
        self._playback = playback
        self._state = state
        self._echo_guard = echo_guard_seconds
        self._turns: TurnCanceller | None = None
        self._capture: BargeInCapture | None = None
        self._timer: asyncio.TimerHandle | None = None
        self._armed = False
        self._holds = 0
        self._interruptions = 0

    @property
    def armed(self) -> bool:
        """True while speech onset would stop playback natively."""
        return self._armed

    @property
    def interruptions(self) -> int:
        """Barge-ins handled so far, for the dashboard and tests."""
        return self._interruptions

    def attach(self, *, turns: TurnCanceller, capture: BargeInCapture) -> None:
        """Connect the orchestrator and the voice input once they exist."""
        self._turns = turns
        self._capture = capture

    @contextmanager
    def suspended(self) -> Iterator[None]:
        """Keep barge-in disarmed, e.g. while asking a yes/no question.

        The user is expected to answer a confirmation prompt, possibly
        before it finishes; that answer must not cancel the turn asking it.
        """
        self._holds += 1
        self._disarm()
        try:
            yield
        finally:
            self._holds -= 1

    def playback_started(self) -> None:
        """First audio of an answer is playing: arm after the echo guard."""
        self._disarm()
        if self._holds:
            return
        loop = asyncio.get_running_loop()
        self._timer = loop.call_later(self._echo_guard, self._arm)

    def playback_finished(self) -> None:
        """The answer was played in full: nothing left to interrupt."""
        self._disarm()

    def playback_interrupted(self) -> None:
        """Playback stopped under the speaker: treat it as the user talking over Jarvis."""
        self._disarm()
        if self._state.state is not AssistantState.SPEAKING:
            log.info("barge_in.ignored", state=self._state.state.value)
            return
        if self._turns is None or self._capture is None:
            raise BargeInNotReadyError("barge-in fired before the controller was attached")
        self._capture.begin_barge_in_capture()
        cancelled = self._turns.cancel_current_turn()
        self._state.transition(AssistantState.LISTENING, "barge-in")
        self._interruptions += 1
        log.info("barge_in.handled", turn_cancelled=cancelled, count=self._interruptions)

    def _arm(self) -> None:
        self._timer = None
        self._engine.enable_barge_in(self._playback)
        self._armed = True
        log.debug("barge_in.armed", guard_s=self._echo_guard)

    def _disarm(self) -> None:
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
        if self._armed:
            self._engine.enable_barge_in(None)
            self._armed = False


# Design notes:
# - Native first, Python second: the audio stops inside one device period of
#   the VAD decision, independent of event-loop load. Python only cleans up
#   (cancel the turn, change state, capture), which may take a poll interval
#   longer without the user hearing any of it.
# - The echo guard is implemented by arming late rather than by filtering
#   events, because the native stop does not consult Python. Disarming
#   between answers also keeps the user's own "hey Jarvis" or a cough while
#   Jarvis is idle from incrementing the stop counter.
# - No acoustic echo cancellation: speakers at high volume can still trigger
#   barge-in after the guard window. The README recommends headphones or a
#   lower volume; AEC (WebRTC APM) is a recorded follow-up in ADR-0019.
# - Capture starts before the turn is cancelled and the state changes: the
#   engine seeds the capture with its pre-roll buffer, which only reaches
#   back 300 ms, so every millisecond of delay here risks clipping the
#   user's first word.
# - An interruption outside SPEAKING (the "didn't catch that" apology, a
#   confirmation prompt) is logged and otherwise ignored; those paths hold
#   ``suspended()`` so it should not happen, and acting on it would request
#   an invalid transition.
