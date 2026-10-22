"""Barge-in: stop speaking when the user interrupts, and only then.

What counts as an interruption depends on the session's
:class:`~jarvis.core.barge_in_mode.BargeInMode`:

* ``VOICE`` - a speech onset stops playback. With the echo check disabled
  the engine's worker thread calls ``Playback::stop()`` on the VAD's
  ``SpeechStarted`` itself, without waiting for Python (ADR-0003). With it
  enabled the onset is checked against the recent microphone and playback
  levels first (:class:`~jarvis.core.echo_detector.EchoDetector`), so
  Jarvis's own voice through the speakers is ignored.
* ``WAKE_WORD`` - only the wake word (or the hotkey, or typing). Plain
  speech and echo never stop Jarvis; this is the mode for speakers.
* ``OFF`` - only the hotkey or typing.

In every mode the first ``echo_guard`` after an answer becomes audible is
ignored for speech, and a wake word while speaking must clear a higher
score and must not follow a sentence in which Jarvis said its own name
(the self-wake guard). Typed barge-in is handled by ``CombinedInput``.

:meth:`BargeInController.playback_interrupted` (the voice output saw
playback stopped under it) then turns the interruption into a new turn:
it starts capturing (the words already in progress after a speech onset,
or a fresh request after the earcon for the wake word and the hotkey),
cancels the agent turn and moves the state machine SPEAKING -> LISTENING.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Coroutine, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Final, Protocol

from jarvis.audio.engine import AudioEvent, AudioEventKind, NativePlayback
from jarvis.core.barge_in_mode import BargeInMode
from jarvis.core.echo_detector import EchoDetector, LevelSample
from jarvis.core.state_machine import AssistantState, StateMachine
from jarvis.core.voice_triggers import HOTKEY_SCORE, TriggerHit
from jarvis.utils.errors import JarvisError
from jarvis.utils.logging import get_logger

log = get_logger(__name__)

DEFAULT_ECHO_GUARD_SECONDS: Final = 0.4
"""Mirrors ``Settings.barge_in_echo_guard_ms``."""

DEFAULT_WAKE_THRESHOLD: Final = 0.8
"""Mirrors ``Settings.wake_word_threshold``."""

DEFAULT_SPEAKING_WAKE_MARGIN: Final = 0.1
"""Mirrors ``Settings.speaking_wake_margin``."""

DEFAULT_SELF_WAKE_GUARD_SECONDS: Final = 2.5
"""Mirrors ``Settings.self_wake_guard_ms``."""

MAX_WAKE_SCORE: Final = 0.99
"""The raised threshold is capped below 1 so the wake word can always interrupt."""

LEVEL_POLL_SECONDS: Final = 0.01
"""How often the echo detector samples the microphone and speaker levels."""

HOTKEY_POLL_SECONDS: Final = 0.05
"""Matches ``voice_triggers.HOTKEY_POLL_SECONDS``."""

WAKE_SOURCE: Final = "wake word"
HOTKEY_SOURCE: Final = "hotkey"


class BargeInNotReadyError(JarvisError):
    """Playback was interrupted before :meth:`BargeInController.attach` ran."""


class BargeInTarget(Protocol):
    """The engine side: who to stop natively on speech onset."""

    def enable_barge_in(self, playback: NativePlayback | None) -> None:
        """Arm with ``playback``, or disarm with ``None``."""


class SpeakingEvents(Protocol):
    """The engine's event stream, consumed while Jarvis speaks (nobody else reads it then)."""

    async def next_event(self, *kinds: AudioEventKind) -> AudioEvent:
        """Wait for the next event of one of ``kinds``."""


class HotkeyPresses(Protocol):
    """The global hotkey (``HotkeyTrigger``)."""

    def pressed(self) -> bool:
        """True when it was pressed since the previous call."""


LevelSource = Callable[[], tuple[float, float]]
"""Current (microphone RMS, speaker RMS)."""


class TurnCanceller(Protocol):
    """The orchestrator side: abandon the answer being produced."""

    def cancel_current_turn(self) -> bool:
        """Cancel the running turn; True if one was running."""


class BargeInCapture(Protocol):
    """The input side: keep the words the user is saying right now."""

    def begin_barge_in_capture(self) -> None:
        """Start retaining audio and make the next request skip the wake word."""

    def begin_triggered_capture(self, hit: TriggerHit) -> None:
        """The wake word or hotkey interrupted: acknowledge it and listen afresh."""


@dataclass(frozen=True, slots=True)
class InterruptPolicy:
    """What may interrupt an answer, from ``Settings`` (see the module docs).

    Attributes:
        mode: The effective barge-in mode for this session.
        wake_threshold: The wake word's normal score threshold.
        speaking_wake_margin: Extra score a wake word needs while Jarvis speaks.
        self_wake_guard_seconds: Wake words are ignored this long after a
            sentence containing ``assistant_name`` starts playing.
        assistant_name: The name Jarvis may say in its own answers.
    """

    mode: BargeInMode = BargeInMode.VOICE
    wake_threshold: float = DEFAULT_WAKE_THRESHOLD
    speaking_wake_margin: float = DEFAULT_SPEAKING_WAKE_MARGIN
    self_wake_guard_seconds: float = DEFAULT_SELF_WAKE_GUARD_SECONDS
    assistant_name: str = "Jarvis"

    @property
    def speaking_wake_threshold(self) -> float:
        """The score a wake word needs while Jarvis is speaking."""
        return min(self.wake_threshold + self.speaking_wake_margin, MAX_WAKE_SCORE)


@dataclass(frozen=True, slots=True)
class SpeakingInputs:
    """What the controller watches while Jarvis speaks; each part is optional.

    Attributes:
        events: Wake-word and speech-onset events from the engine.
        hotkey: The global hotkey, which interrupts in every mode.
        levels: Microphone and speaker levels for the echo check.
        echo: Decides whether an onset is only echo (voice mode).
    """

    events: SpeakingEvents | None = None
    hotkey: HotkeyPresses | None = None
    levels: LevelSource | None = None
    echo: EchoDetector | None = None


class BargeInController:
    """Watches for interruptions while Jarvis speaks and turns one into a new turn.

    Implements the voice output's ``PlaybackObserver`` Protocol. The
    orchestrator and the voice input exist only after this object (they
    depend on the sink it observes), so they are connected with
    :meth:`attach` once built.

    Args:
        engine: Where the native barge-in target is set.
        playback: The playback to stop.
        state: The assistant state machine, owned by the orchestrator.
        echo_guard_seconds: Speech onset during this window after playback
            starts is ignored.
        policy: Mode and wake-word rules; the default is plain voice barge-in.
        inputs: Event, hotkey and level sources watched while speaking.
        clock: Monotonic seconds, for the guards and the level history.
    """

    def __init__(
        self,
        *,
        engine: BargeInTarget,
        playback: NativePlayback,
        state: StateMachine,
        echo_guard_seconds: float = DEFAULT_ECHO_GUARD_SECONDS,
        policy: InterruptPolicy | None = None,
        inputs: SpeakingInputs | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._engine = engine
        self._playback = playback
        self._state = state
        self._echo_guard = echo_guard_seconds
        self._policy = policy or InterruptPolicy()
        self._inputs = inputs or SpeakingInputs()
        self._clock = clock
        self._turns: TurnCanceller | None = None
        self._capture: BargeInCapture | None = None
        self._timer: asyncio.TimerHandle | None = None
        self._watchers: list[asyncio.Task[None]] = []
        self._armed = False
        self._speech_open = False
        self._holds = 0
        self._interruptions = 0
        self._pending: TriggerHit | None = None
        self._self_wake_until = float("-inf")

    @property
    def armed(self) -> bool:
        """True while speech onset would stop playback natively."""
        return self._armed

    @property
    def mode(self) -> BargeInMode:
        """The barge-in mode this controller enforces."""
        return self._policy.mode

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
        """First audio of an answer is playing: watch for interruptions, speech after the guard."""
        self._disarm()
        if self._holds:
            return
        self._start_watchers()
        loop = asyncio.get_running_loop()
        self._timer = loop.call_later(self._echo_guard, self._arm)

    def sentence_started(self, text: str) -> None:
        """A sentence starts playing; one that names Jarvis opens the self-wake guard."""
        if self._policy.assistant_name.casefold() in text.casefold():
            self._self_wake_until = self._clock() + self._policy.self_wake_guard_seconds

    def playback_finished(self) -> None:
        """The answer was played in full: nothing left to interrupt."""
        self._disarm()

    def playback_interrupted(self) -> None:
        """Playback stopped under the speaker: the user interrupted, start their turn."""
        self._disarm()
        hit, self._pending = self._pending, None
        if self._state.state is not AssistantState.SPEAKING:
            log.info("barge_in.ignored", state=self._state.state.value)
            return
        if self._turns is None or self._capture is None:
            raise BargeInNotReadyError("barge-in fired before the controller was attached")
        if hit is None:
            self._capture.begin_barge_in_capture()
        else:
            self._capture.begin_triggered_capture(hit)
        cancelled = self._turns.cancel_current_turn()
        reason = "barge-in" if hit is None else f"barge-in ({hit.source})"
        self._state.transition(AssistantState.LISTENING, reason)
        self._interruptions += 1
        log.info("barge_in.handled", turn_cancelled=cancelled, count=self._interruptions)

    @property
    def _native_stop(self) -> bool:
        """Voice mode without a usable echo check: the engine stops playback itself."""
        echo, events = self._inputs.echo, self._inputs.events
        python_decides = echo is not None and echo.enabled and events is not None
        return self._policy.mode is BargeInMode.VOICE and not python_decides

    def _arm(self) -> None:
        self._timer = None
        if self._policy.mode is not BargeInMode.VOICE:
            return
        if self._native_stop:
            self._engine.enable_barge_in(self._playback)
            self._armed = True
        else:
            self._speech_open = True
        log.debug("barge_in.armed", guard_s=self._echo_guard, native=self._native_stop)

    def _disarm(self) -> None:
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
        if self._armed:
            self._engine.enable_barge_in(None)
            self._armed = False
        self._speech_open = False
        for watcher in self._watchers:
            watcher.cancel()
        self._watchers.clear()

    def _start_watchers(self) -> None:
        inputs = self._inputs
        if inputs.events is not None and self._policy.mode is not BargeInMode.OFF:
            self._spawn(self._watch_events(inputs.events), "barge-in-events")
        if inputs.hotkey is not None:
            self._spawn(self._watch_hotkey(inputs.hotkey), "barge-in-hotkey")
        echo, levels = inputs.echo, inputs.levels
        python_voice = self._policy.mode is BargeInMode.VOICE and not self._native_stop
        if echo is not None and levels is not None and python_voice:
            self._spawn(self._sample_levels(levels, echo), "barge-in-levels")

    def _spawn(self, work: Coroutine[object, object, None], name: str) -> None:
        task = asyncio.create_task(work, name=name)
        task.add_done_callback(_report_crash)
        self._watchers.append(task)

    async def _watch_events(self, events: SpeakingEvents) -> None:
        kinds = [AudioEventKind.WAKE_WORD_DETECTED]
        if self._policy.mode is BargeInMode.VOICE:
            kinds.append(AudioEventKind.SPEECH_STARTED)
        while not self._interrupts(await events.next_event(*kinds)):
            pass

    async def _watch_hotkey(self, hotkey: HotkeyPresses) -> None:
        hotkey.pressed()  # the press that started this turn, if any, is not an interruption
        # Presses are only visible by polling (ADR-0004); ASYNC110 wants an Event.
        while not hotkey.pressed():  # noqa: ASYNC110
            await asyncio.sleep(HOTKEY_POLL_SECONDS)
        self._interrupt(TriggerHit(HOTKEY_SOURCE, HOTKEY_SCORE))

    async def _sample_levels(self, levels: LevelSource, echo: EchoDetector) -> None:
        echo.reset()
        while True:
            heard, played = levels()
            echo.record(LevelSample(self._clock(), heard, played))
            await asyncio.sleep(LEVEL_POLL_SECONDS)

    def _interrupts(self, event: AudioEvent) -> bool:
        """Act on one event while speaking; True once it stopped playback."""
        if event.kind is AudioEventKind.WAKE_WORD_DETECTED:
            accepted = self._wake_accepted(event.value)
            if accepted:
                self._interrupt(TriggerHit(WAKE_SOURCE, event.value))
            return accepted
        accepted = self._speech_accepted()
        if accepted:
            self._interrupt(None)
        return accepted

    def _wake_accepted(self, score: float) -> bool:
        if self._clock() < self._self_wake_until:
            log.info("barge_in.self_wake_ignored", score=round(score, 3))
            return False
        if score < self._policy.speaking_wake_threshold:
            log.info(
                "barge_in.wake_below_speaking_threshold",
                score=round(score, 3),
                needed=round(self._policy.speaking_wake_threshold, 3),
            )
            return False
        return True

    def _speech_accepted(self) -> bool:
        if not self._speech_open:
            return False  # inside the echo guard, or speech does not interrupt in this mode
        echo = self._inputs.echo
        if echo is not None and echo.is_echo(self._clock()):
            log.info("barge_in.echo_ignored")
            return False
        return True

    def _interrupt(self, hit: TriggerHit | None) -> None:
        """Stop playback; the voice output notices and calls :meth:`playback_interrupted`."""
        self._pending = hit
        log.info("barge_in.interrupting", source=hit.source if hit else "speech")
        self._playback.stop()


def _report_crash(task: asyncio.Task[None]) -> None:
    if not task.cancelled() and (error := task.exception()) is not None:
        log.error("barge_in.watcher_failed", watcher=task.get_name(), error=repr(error))


# Design notes:
# - Native first where it is safe: with voice mode and no echo check the
#   audio stops inside one device period of the VAD decision, independent
#   of event-loop load. Every other decision (wake word, hotkey, echo check)
#   needs Python, which adds at most a poll interval (10-50 ms) to the stop.
# - The wake word while speaking is the speaker-friendly interruption:
#   openWakeWord is trained on one phrase, so Jarvis's own sentences rarely
#   score high, and two guards cover what remains. The raised threshold
#   applies to every wake word while speaking; the self-wake guard covers
#   the one case that genuinely looks like the phrase, Jarvis saying its
#   name, for a few seconds after such a sentence starts.
# - The watchers consume the engine's event queue only while Jarvis speaks,
#   when nothing else reads it; the voice input discards stale events
#   before it waits for the next trigger anyway.
# - No acoustic echo cancellation: the echo check is an energy heuristic.
#   AEC (WebRTC APM) remains the recorded follow-up in ADR-0019.
# - Capture starts before the turn is cancelled and the state changes: the
#   engine seeds the capture with its pre-roll buffer, which only reaches
#   back 300 ms, so every millisecond of delay here risks clipping the
#   user's first word. The wake word and the hotkey instead get the
#   earcon and a fresh capture, exactly like starting a turn from idle.
# - An interruption outside SPEAKING (the "didn't catch that" apology, a
#   confirmation prompt) is logged and otherwise ignored; those paths hold
#   ``suspended()`` so it should not happen, and acting on it would request
#   an invalid transition.
