"""Barge-in modes, the echo check and the self-wake guard (ADR-0019 update)."""

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from jarvis.audio.engine import AudioEvent, AudioEventKind
from jarvis.config import Settings
from jarvis.core.barge_in_mode import BargeInMode, looks_like_headphones, resolve_barge_in_mode
from jarvis.core.echo_detector import EchoDetector, LevelSample
from jarvis.core.events import StateChanged
from jarvis.core.interruption import BargeInController, InterruptPolicy, SpeakingInputs
from jarvis.core.state_machine import AssistantState, StateMachine
from jarvis.core.voice_triggers import TriggerHit
from jarvis.llm.base import TextDelta
from jarvis.os_integration.base import OutputEndpoint
from tests.agent.llm_fakes import Block, ScriptedLLM, text_round
from tests.audio.fakes import FakeNativePlayback, native_event
from tests.core.test_interruption import FakeTurns, RecordingTarget, speaking_machine
from tests.core.test_voice_integration import WAIT, VoiceSession
from tests.core.voice_fakes import eventually

GUARD = 0.01

# --- fakes -------------------------------------------------------------------


class FakeEvents:
    """The engine's event queue as the controller reads it while speaking."""

    def __init__(self) -> None:
        self.queue: asyncio.Queue[AudioEvent] = asyncio.Queue()

    def emit(self, kind: AudioEventKind, value: float = 0.0) -> None:
        self.queue.put_nowait(AudioEvent(kind, value, 0))

    async def next_event(self, *kinds: AudioEventKind) -> AudioEvent:
        while True:
            event = await self.queue.get()
            if event.kind in kinds:
                return event


@dataclass
class FakeHotkey:
    presses: list[bool] = field(default_factory=list)

    def pressed(self) -> bool:
        return self.presses.pop(0) if self.presses else False


@dataclass
class TriggerCapture:
    barge_ins: int = 0
    triggered: list[TriggerHit] = field(default_factory=list)

    def begin_barge_in_capture(self) -> None:
        self.barge_ins += 1

    def begin_triggered_capture(self, hit: TriggerHit) -> None:
        self.triggered.append(hit)


@dataclass
class Clock:
    now: float = 100.0

    def __call__(self) -> float:
        return self.now


@dataclass
class Rig:
    mode: BargeInMode
    events: FakeEvents = field(default_factory=FakeEvents)
    hotkey: FakeHotkey = field(default_factory=FakeHotkey)
    playback: FakeNativePlayback = field(default_factory=FakeNativePlayback)
    target: RecordingTarget = field(default_factory=RecordingTarget)
    state: StateMachine = field(default_factory=speaking_machine)
    turns: FakeTurns = field(default_factory=FakeTurns)
    capture: TriggerCapture = field(default_factory=TriggerCapture)
    clock: Callable[[], float] = field(default_factory=Clock)
    levels: Callable[[], tuple[float, float]] | None = None
    echo: EchoDetector | None = None

    def controller(self) -> BargeInController:
        controller = BargeInController(
            engine=self.target,
            playback=self.playback,
            state=self.state,
            echo_guard_seconds=GUARD,
            policy=InterruptPolicy(mode=self.mode, wake_threshold=0.8, speaking_wake_margin=0.1),
            inputs=SpeakingInputs(
                events=self.events, hotkey=self.hotkey, levels=self.levels, echo=self.echo
            ),
            clock=self.clock,
        )
        controller.attach(turns=self.turns, capture=self.capture)
        return controller


async def settle() -> None:
    await asyncio.sleep(GUARD * 4)


# --- modes -------------------------------------------------------------------


@pytest.mark.parametrize("mode", [BargeInMode.WAKE_WORD, BargeInMode.OFF])
async def test_speech_and_echo_never_interrupt_outside_voice_mode(mode: BargeInMode) -> None:
    rig = Rig(mode)
    barge_in = rig.controller()
    barge_in.playback_started()
    await settle()

    rig.events.emit(AudioEventKind.SPEECH_STARTED, 0.9)
    await settle()

    assert rig.playback.stops == 0
    assert rig.target.targets == []  # the native target is never armed
    assert not barge_in.armed
    barge_in.playback_finished()


async def test_the_wake_word_interrupts_in_wake_word_mode() -> None:
    rig = Rig(BargeInMode.WAKE_WORD)
    barge_in = rig.controller()
    barge_in.playback_started()

    rig.events.emit(AudioEventKind.WAKE_WORD_DETECTED, 0.95)
    await settle()
    assert rig.playback.stops == 1
    barge_in.playback_interrupted()  # what the voice output does once it saw the stop

    assert rig.capture.triggered == [TriggerHit("wake word", 0.95)]
    assert rig.capture.barge_ins == 0
    assert rig.turns.cancels == 1
    assert rig.state.state is AssistantState.LISTENING


async def test_off_mode_ignores_the_wake_word_but_not_the_hotkey() -> None:
    rig = Rig(BargeInMode.OFF)
    barge_in = rig.controller()
    rig.hotkey.presses = [True, False, False, True]  # the first is the stale turn-starting press
    barge_in.playback_started()
    rig.events.emit(AudioEventKind.WAKE_WORD_DETECTED, 0.99)

    await eventually(lambda: rig.playback.stops == 1)
    barge_in.playback_interrupted()

    assert rig.capture.triggered == [TriggerHit("hotkey", 1.0)]
    assert rig.state.state is AssistantState.LISTENING


async def test_a_wake_word_below_the_speaking_threshold_is_ignored() -> None:
    rig = Rig(BargeInMode.WAKE_WORD)
    barge_in = rig.controller()
    barge_in.playback_started()

    rig.events.emit(AudioEventKind.WAKE_WORD_DETECTED, 0.85)  # above 0.8, below 0.8 + 0.1
    await settle()

    assert rig.playback.stops == 0
    barge_in.playback_finished()


async def test_jarvis_saying_its_own_name_cannot_wake_it() -> None:
    clock = Clock()
    rig = Rig(BargeInMode.WAKE_WORD, clock=clock)
    barge_in = rig.controller()
    barge_in.playback_started()

    barge_in.sentence_started("Certainly, Jarvis will do that.")
    rig.events.emit(AudioEventKind.WAKE_WORD_DETECTED, 0.99)
    await settle()
    assert rig.playback.stops == 0

    clock.now += 3.0  # past the 2.5 s self-wake guard
    rig.events.emit(AudioEventKind.WAKE_WORD_DETECTED, 0.99)
    await settle()
    assert rig.playback.stops == 1


async def test_a_sentence_without_the_name_opens_no_guard() -> None:
    rig = Rig(BargeInMode.WAKE_WORD)
    barge_in = rig.controller()
    barge_in.playback_started()

    barge_in.sentence_started("It is sunny.")
    rig.events.emit(AudioEventKind.WAKE_WORD_DETECTED, 0.95)
    await settle()

    assert rig.playback.stops == 1


async def test_watchers_stop_when_the_answer_ends() -> None:
    rig = Rig(BargeInMode.WAKE_WORD)
    barge_in = rig.controller()
    barge_in.playback_started()
    barge_in.playback_finished()

    rig.events.emit(AudioEventKind.WAKE_WORD_DETECTED, 0.99)
    await settle()

    assert rig.playback.stops == 0


@dataclass
class Levels:
    heard: float
    played: float

    def __call__(self) -> tuple[float, float]:
        return self.heard, self.played


async def test_voice_mode_ignores_an_onset_that_is_only_echo_and_takes_a_real_one() -> None:
    levels = Levels(heard=0.03, played=0.1)  # speakers: the mic hears 30 % of the playback
    rig = Rig(
        BargeInMode.VOICE,
        clock=asyncio.get_running_loop().time,
        levels=levels,
        echo=EchoDetector(ratio=2.0, window_seconds=0.05),
    )
    barge_in = rig.controller()
    barge_in.playback_started()
    await asyncio.sleep(0.15)  # calibration during and after the guard

    rig.events.emit(AudioEventKind.SPEECH_STARTED, 0.9)
    await settle()
    assert rig.playback.stops == 0
    assert rig.target.targets == []  # Python decides; the engine target stays unarmed

    levels.heard = 0.2  # the user talks over Jarvis
    await asyncio.sleep(0.1)
    rig.events.emit(AudioEventKind.SPEECH_STARTED, 0.9)
    await settle()
    assert rig.playback.stops == 1
    barge_in.playback_interrupted()
    assert rig.capture.barge_ins == 1
    assert rig.state.state is AssistantState.LISTENING


async def test_voice_mode_onset_inside_the_guard_is_ignored() -> None:
    rig = Rig(BargeInMode.VOICE, levels=Levels(0.2, 0.1), echo=EchoDetector())
    barge_in = rig.controller()
    barge_in.playback_started()

    rig.events.emit(AudioEventKind.SPEECH_STARTED, 0.9)
    await asyncio.sleep(GUARD / 4)

    assert rig.playback.stops == 0
    barge_in.playback_finished()


async def test_voice_mode_without_the_echo_check_arms_the_engine() -> None:
    rig = Rig(BargeInMode.VOICE, echo=EchoDetector(ratio=0.0))
    barge_in = rig.controller()
    barge_in.playback_started()
    await settle()

    assert barge_in.armed
    assert rig.target.current is rig.playback
    barge_in.playback_finished()


# --- echo detector with synthetic level traces ---------------------------------


def trace(detector: EchoDetector, start: float, end: float, heard: float, played: float) -> None:
    steps = round((end - start) / 0.01)
    for step in range(steps):
        detector.record(LevelSample(start + step * 0.01, heard, played))


def test_echo_alone_is_recognised_as_echo() -> None:
    detector = EchoDetector(ratio=2.0, window_seconds=0.3)
    trace(detector, 0.0, 1.0, heard=0.04, played=0.1)

    assert detector.is_echo(1.0)


def test_the_user_talking_over_the_echo_is_not_echo() -> None:
    detector = EchoDetector(ratio=2.0, window_seconds=0.3)
    trace(detector, 0.0, 0.7, heard=0.04, played=0.1)
    trace(detector, 0.7, 1.0, heard=0.15, played=0.1)

    assert detector.coupling(0.7) == pytest.approx(0.4)
    assert not detector.is_echo(1.0)


def test_on_headphones_quiet_speech_still_counts() -> None:
    detector = EchoDetector(ratio=2.0, window_seconds=0.3)
    trace(detector, 0.0, 0.7, heard=0.002, played=0.2)  # noise floor only
    trace(detector, 0.7, 1.0, heard=0.03, played=0.2)

    assert not detector.is_echo(1.0)


def test_without_audible_playback_nothing_is_echo() -> None:
    detector = EchoDetector()
    trace(detector, 0.0, 1.0, heard=0.04, played=0.0)

    assert not detector.is_echo(1.0)


def test_without_calibration_the_onset_counts() -> None:
    detector = EchoDetector(window_seconds=0.3)
    trace(detector, 0.0, 0.3, heard=0.04, played=0.1)

    assert detector.coupling(0.0) is None
    assert not detector.is_echo(0.3)


def test_a_disabled_detector_never_reports_echo() -> None:
    detector = EchoDetector(ratio=0.0)
    trace(detector, 0.0, 1.0, heard=0.04, played=0.1)

    assert not detector.enabled
    assert not detector.is_echo(1.0)


def test_reset_forgets_the_calibration() -> None:
    detector = EchoDetector()
    trace(detector, 0.0, 1.0, heard=0.04, played=0.1)
    detector.reset()

    assert detector.coupling(2.0) is None


# --- auto mode from the output device ----------------------------------------


SPEAKERS = OutputEndpoint("Speakers (Realtek(R) Audio)", "speakers")
HEADPHONES = OutputEndpoint("Headphones (WH-1000XM4)", "headphones")
HEADSET = OutputEndpoint("Headset Earphone (Jabra)", "headset")
UNKNOWN = OutputEndpoint("Hörlurar (USB Audio)", "unknown")


@pytest.mark.parametrize(
    ("endpoint", "microphone", "expected"),
    [
        (SPEAKERS, None, BargeInMode.WAKE_WORD),
        (HEADPHONES, None, BargeInMode.VOICE),
        (HEADSET, None, BargeInMode.VOICE),
        (UNKNOWN, None, BargeInMode.VOICE),  # unknown form factor: the name decides
        (OutputEndpoint("Realtek HD", "unknown"), None, BargeInMode.WAKE_WORD),
        (None, "Headset Microphone (Jabra)", BargeInMode.VOICE),  # native unavailable
        (None, "Microphone Array (Realtek)", BargeInMode.WAKE_WORD),
        (None, None, BargeInMode.WAKE_WORD),  # nothing known: the safe choice
        (SPEAKERS, "Headset Microphone", BargeInMode.WAKE_WORD),  # declared speakers win
    ],
)
def test_auto_mode_picks_voice_only_for_headphones(
    endpoint: OutputEndpoint | None, microphone: str | None, expected: BargeInMode
) -> None:
    mode = resolve_barge_in_mode("auto", endpoint=lambda: endpoint, microphone=lambda: microphone)

    assert mode is expected


def test_an_explicit_mode_skips_the_device_probe() -> None:
    def probe() -> OutputEndpoint:
        raise AssertionError("must not be called")

    assert resolve_barge_in_mode("off", endpoint=probe, microphone=probe) is BargeInMode.OFF  # type: ignore[arg-type]


def test_headphone_hints_include_swedish() -> None:
    assert looks_like_headphones(None, "Hörlurar (Bluetooth)")


# --- through the whole voice front end -----------------------------------------


@pytest.fixture
def settings(make_settings: Callable[..., Settings], tmp_path: Path) -> Settings:
    return make_settings(
        llm_provider="ollama",
        data_dir=tmp_path,
        barge_in_echo_guard_ms=10,
        barge_in_mode="wake_word",
    )


async def test_on_speakers_echo_is_ignored_and_the_wake_word_interrupts(
    settings: Settings,
) -> None:
    session = VoiceSession()
    llm = ScriptedLLM(
        rounds=[[TextDelta("Here is a very long story. "), Block()], text_round("Sunny and mild.")]
    )
    application = await session.build(settings, llm)
    running = asyncio.create_task(application.run())

    await session.speak(application, "Tell me a story.")
    session.playback.hold = True
    await eventually(lambda: "Jarvis> Here is a very long story.\n" in session.printed)
    await asyncio.sleep(0.05)  # past the echo guard
    stops = session.playback.stops
    session.native.script(native_event("SPEECH_STARTED"))  # Jarvis heard through the speakers
    await asyncio.sleep(0.05)
    assert application.orchestrator.state is AssistantState.SPEAKING
    assert session.playback.stops == stops
    assert all(target is None for target in session.native.barge_in_targets)

    session.stt.texts.append("What's the weather?")
    calls = len(session.native.listening_calls)
    session.native.script(native_event("WAKE_WORD_DETECTED", 0.95))
    await eventually(lambda: application.orchestrator.state is AssistantState.LISTENING)
    session.playback.release()  # the earcon may already be queued behind the hold
    # LISTENING is published before the earcon and before the capture clears
    # stale events; speech scripted in between would be discarded with them.
    await eventually(lambda: True in session.native.listening_calls[calls:])
    session.native.say_utterance()
    await eventually(lambda: "Jarvis> Sunny and mild.\n" in session.printed)
    await eventually(lambda: application.orchestrator.state is AssistantState.IDLE)
    await session.speak(application, "Goodbye.")
    await asyncio.wait_for(running, WAIT)

    assert "you> What's the weather?\n" in session.printed
    reasons = [e.reason for e in session.events if isinstance(e, StateChanged)]
    assert "barge-in (wake word)" in reasons
