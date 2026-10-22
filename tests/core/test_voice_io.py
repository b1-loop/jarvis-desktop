import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

import numpy as np
import pytest

from jarvis.audio.engine import AudioEngine, Samples, Speaker
from jarvis.core.events import Event, EventBus, LatencySample, TranscriptReady, WakeWordHeard
from jarvis.core.interruption import BargeInController
from jarvis.core.orchestrator import TerminalOutput
from jarvis.core.state_machine import AssistantState, StateMachine
from jarvis.core.voice_io import (
    DIDNT_CATCH,
    PLEASE_ANSWER,
    Earcon,
    SystemVoice,
    VoiceConfirmation,
    VoiceInput,
    VoiceOutput,
    make_earcon,
    parse_spoken_yes_no,
)
from jarvis.core.voice_latency import (
    FIRST_SENTENCE_TO_AUDIO,
    SPEECH_END_TO_TRANSCRIPT,
    WAKE_TO_LISTENING,
    TurnTimeline,
)
from jarvis.core.voice_triggers import AnyTrigger, HotkeyTrigger, TriggerHit, WakeWordTrigger
from jarvis.os_integration.stub import StubPlatform
from tests.audio.fakes import native_event
from tests.core.voice_fakes import (
    BargeInEngine,
    DevicePlayback,
    ScriptedSTT,
    ScriptedTTS,
    eventually,
    stt_error,
)

POLL = 0.001
WAIT = 2.0


@dataclass
class Rig:
    """One voice front end over fakes, with everything observable."""

    native: BargeInEngine = field(default_factory=BargeInEngine)
    playback: DevicePlayback = field(default_factory=DevicePlayback)
    bus: EventBus = field(default_factory=EventBus)
    stt: ScriptedSTT = field(default_factory=ScriptedSTT)
    announced: list[str] = field(default_factory=list)
    printed: list[str] = field(default_factory=list)
    events: list[Event] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.engine = AudioEngine(self.native, poll_interval=POLL, max_utterance_seconds=1.0)
        self.speaker = Speaker(self.playback, poll_interval=POLL)
        self.state = StateMachine(self.bus)
        self.timeline = TurnTimeline(self.bus)
        self.bus.subscribe_callback(Event, self.events.append)

    async def announce(self, text: str) -> None:
        self.announced.append(text)

    def voice_input(self, trigger: object, *, listen_timeout: float = 0.2) -> VoiceInput:
        return VoiceInput(
            engine=self.engine,
            stt=self.stt,
            trigger=trigger,  # type: ignore[arg-type]
            state=self.state,
            bus=self.bus,
            timeline=self.timeline,
            earcon=Earcon(self.speaker, make_earcon(self.speaker.sample_rate)),
            announce=self.announce,
            listen_timeout=listen_timeout,
            echo=self.printed.append,
        )

    def latency(self, stage: str) -> list[float]:
        return [e.ms for e in self.events if isinstance(e, LatencySample) and e.stage == stage]


@pytest.fixture
async def rig() -> AsyncIterator[Rig]:
    rig = Rig()
    async with rig.engine:
        yield rig


# --- earcon ----------------------------------------------------------------


def test_earcon_is_a_short_click_free_tone() -> None:
    tone = make_earcon(24_000)

    assert tone.dtype == np.float32
    assert tone.size == 1920  # 80 ms
    assert abs(float(tone[0])) < 1e-6
    assert abs(float(tone[-1])) < 1e-6
    assert 0.1 < float(np.abs(tone).max()) <= 0.2


# --- voice input -----------------------------------------------------------


async def test_wake_word_turn_plays_earcon_listens_and_transcribes(rig: Rig) -> None:
    rig.stt.texts = ["What time is it?"]
    rig.native.script(native_event("WAKE_WORD_DETECTED", 0.9))
    rig.native.say_utterance()
    source = rig.voice_input(WakeWordTrigger(rig.engine))

    text = await asyncio.wait_for(source.read_line("you> "), WAIT)

    assert text == "What time is it?"
    assert rig.state.state is AssistantState.LISTENING
    assert len(rig.playback.played) == 1  # the earcon
    assert rig.printed == ["you> What time is it?\n"]
    assert any(isinstance(e, WakeWordHeard) and e.score == pytest.approx(0.9) for e in rig.events)
    assert [e.text for e in rig.events if isinstance(e, TranscriptReady)] == ["What time is it?"]
    assert len(rig.latency(WAKE_TO_LISTENING)) == 1
    assert len(rig.latency(SPEECH_END_TO_TRANSCRIPT)) == 1


async def test_hotkey_starts_a_turn(rig: Rig) -> None:
    platform = StubPlatform()
    hotkey = HotkeyTrigger(platform, "ctrl+alt+j", poll_interval=POLL)
    await hotkey.register()
    rig.stt.texts = ["Open the browser."]
    source = rig.voice_input(hotkey)

    async def press_then_speak() -> None:
        await asyncio.sleep(0.02)
        platform.press(1)
        rig.native.say_utterance()

    pressing = asyncio.create_task(press_then_speak())
    text = await asyncio.wait_for(source.read_line("you> "), WAIT)
    await pressing

    assert text == "Open the browser."
    assert [e.reason for e in rig.events if hasattr(e, "reason")] == ["hotkey"]
    await hotkey.aclose()
    assert platform.hotkeys == {}


async def test_speech_that_is_not_understood_gets_an_apology(rig: Rig) -> None:
    rig.stt.texts = ["...", "Lights off."]
    rig.native.script(native_event("WAKE_WORD_DETECTED", 0.8))
    rig.native.say_utterance()
    source = rig.voice_input(WakeWordTrigger(rig.engine))

    first = asyncio.create_task(source.read_line("you> "))
    await eventually(lambda: rig.announced)
    assert rig.announced == [DIDNT_CATCH]
    assert rig.state.state is AssistantState.IDLE
    await asyncio.sleep(0.02)  # the input is waiting for the next trigger again
    rig.native.script(native_event("WAKE_WORD_DETECTED", 0.8))
    rig.native.say_utterance()
    assert await asyncio.wait_for(first, WAIT) == "Lights off."


async def test_silence_after_the_trigger_returns_to_idle_without_apology(rig: Rig) -> None:
    hits = [TriggerHit("hotkey", 1.0), TriggerHit("hotkey", 1.0)]
    gate = asyncio.Event()

    class TwoPresses:
        async def wait(self) -> TriggerHit:
            if len(hits) == 1:
                await gate.wait()
                rig.native.say_utterance()
            return hits.pop(0)

    rig.stt.texts = ["Hello there."]
    source = rig.voice_input(TwoPresses(), listen_timeout=0.03)
    task = asyncio.create_task(source.read_line("you> "))
    await eventually(lambda: len(hits) == 1 and rig.state.state is AssistantState.IDLE)
    assert rig.announced == []
    assert rig.stt.calls == []  # nothing was heard, so nothing was transcribed
    gate.set()

    assert await asyncio.wait_for(task, WAIT) == "Hello there."


async def test_a_failing_transcription_counts_as_not_understood(rig: Rig) -> None:
    rig.stt.texts = [stt_error(), "Second try."]
    rig.native.script(native_event("WAKE_WORD_DETECTED", 0.8))
    rig.native.say_utterance()
    source = rig.voice_input(WakeWordTrigger(rig.engine))

    task = asyncio.create_task(source.read_line("you> "))
    await eventually(lambda: rig.announced)
    await asyncio.sleep(0.02)
    rig.native.script(native_event("WAKE_WORD_DETECTED", 0.8))
    rig.native.say_utterance()
    assert await asyncio.wait_for(task, WAIT) == "Second try."
    assert rig.announced == [DIDNT_CATCH]


async def test_barge_in_capture_skips_the_trigger(rig: Rig) -> None:
    rig.state.transition(AssistantState.LISTENING, "test")
    rig.stt.texts = ["Stop, tell me the weather."]
    source = rig.voice_input(WakeWordTrigger(rig.engine))
    rig.native.script(native_event("SPEECH_ENDED"))  # stale, from before the barge-in
    await asyncio.sleep(0.02)

    source.begin_barge_in_capture()
    rig.native.captured = np.ones(800, dtype=np.float32)
    rig.native.script(native_event("SPEECH_ENDED"))
    text = await asyncio.wait_for(source.read_line("you> "), WAIT)

    assert text == "Stop, tell me the weather."
    assert rig.native.listening_calls[0] is True
    assert rig.playback.played == []  # no earcon: the user is already talking
    assert rig.stt.calls == [800]


async def test_muted_input_ignores_triggers(rig: Rig) -> None:
    calls: list[int] = []

    class Presses:
        async def wait(self) -> TriggerHit:
            calls.append(1)
            if len(calls) == 2:
                source.muted = False
                rig.native.say_utterance()
            return TriggerHit("hotkey", 1.0)

    rig.stt.texts = ["Unmuted now."]
    source = rig.voice_input(Presses())
    source.muted = True

    assert await asyncio.wait_for(source.read_line("you> "), WAIT) == "Unmuted now."
    assert len(calls) == 2


async def test_listen_for_answer_needs_no_trigger(rig: Rig) -> None:
    rig.stt.texts = ["Yes please."]
    rig.native.say_utterance()
    source = rig.voice_input(WakeWordTrigger(rig.engine))

    assert await asyncio.wait_for(source.listen_for_answer(), WAIT) == "Yes please."
    assert rig.state.state is AssistantState.IDLE


# --- triggers --------------------------------------------------------------


async def test_any_trigger_returns_the_first_and_cancels_the_rest() -> None:
    cancelled: list[str] = []

    class Slow:
        async def wait(self) -> TriggerHit:
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                cancelled.append("slow")
                raise
            return TriggerHit("never", 0.0)

    class Fast:
        async def wait(self) -> TriggerHit:
            await asyncio.sleep(0.01)
            return TriggerHit("fast", 1.0)

    hit = await asyncio.wait_for(AnyTrigger([Slow(), Fast()]).wait(), WAIT)

    assert hit.source == "fast"
    assert cancelled == ["slow"]


def test_any_trigger_needs_a_trigger() -> None:
    with pytest.raises(ValueError, match="at least one"):
        AnyTrigger([])


async def test_hotkey_trigger_drops_presses_made_while_busy() -> None:
    platform = StubPlatform()
    hotkey = HotkeyTrigger(platform, "ctrl+alt+j", poll_interval=POLL)
    with pytest.raises(RuntimeError, match="register"):
        await hotkey.wait()
    await hotkey.register()
    platform.press(1)  # pressed while Jarvis was speaking

    waiting = asyncio.create_task(hotkey.wait())
    await asyncio.sleep(0.02)
    assert not waiting.done()
    platform.press(1)

    assert (await asyncio.wait_for(waiting, WAIT)).source == "hotkey"


# --- voice output ----------------------------------------------------------


@dataclass
class RecordingObserver:
    calls: list[str] = field(default_factory=list)

    def playback_started(self) -> None:
        self.calls.append("started")

    def playback_finished(self) -> None:
        self.calls.append("finished")

    def playback_interrupted(self) -> None:
        self.calls.append("interrupted")


def voice_output(
    tts: ScriptedTTS, playback: DevicePlayback, bus: EventBus, printed: list[str]
) -> tuple[VoiceOutput, RecordingObserver]:
    observer = RecordingObserver()
    output = VoiceOutput(
        tts=tts,
        speaker=Speaker(playback, poll_interval=POLL),
        bus=bus,
        terminal=TerminalOutput("Jarvis", write=printed.append),
        observer=observer,
    )
    return output, observer


async def test_sentences_are_spoken_in_order_and_printed() -> None:
    bus, printed, playback = EventBus(), [], DevicePlayback()
    samples: list[LatencySample] = []
    bus.subscribe_callback(LatencySample, samples.append)
    output, observer = voice_output(ScriptedTTS(chunks=2), playback, bus, printed)

    await output.say("Good evening.")
    await output.say("It is nine o'clock.")
    assert await asyncio.wait_for(output.flush(), WAIT)
    await output.aclose()

    assert printed == ["Jarvis> Good evening.\n", "Jarvis> It is nine o'clock.\n"]
    assert len(playback.played) == 4
    assert observer.calls == ["started", "finished"]
    assert [s.stage for s in samples] == [FIRST_SENTENCE_TO_AUDIO]


async def test_synthesis_of_the_next_sentence_overlaps_playback_of_the_current() -> None:
    log: list[str] = []
    playback = DevicePlayback(hold=True)
    output, _ = voice_output(ScriptedTTS(chunks=2, log=log), playback, EventBus(), [])

    await output.say("First sentence.")
    await output.say("Second sentence.")
    flushed = asyncio.create_task(output.flush())
    await eventually(lambda: "synth-done:Second sentence." in log)

    # Sentence one is still on the device (held) while sentence two is
    # already fully synthesised: the two stages overlap.
    assert playback.is_playing()
    assert len(playback.played) == 2
    playback.release()
    await eventually(lambda: len(playback.played) == 4)
    playback.release()
    assert await asyncio.wait_for(flushed, WAIT)
    await output.aclose()


async def test_tts_failure_prints_the_text_and_continues() -> None:
    printed: list[str] = []
    playback = DevicePlayback()
    tts = ScriptedTTS(failing=frozenset({"Broken one.", "Broken two."}))
    output, _ = voice_output(tts, playback, EventBus(), printed)

    for sentence in ("Broken one.", "Broken two.", "Fine one."):
        await output.say(sentence)
    assert await asyncio.wait_for(output.flush(), WAIT)
    await output.aclose()

    assert "Jarvis> Broken one.\n" in printed
    assert "Jarvis> Broken two.\n" in printed
    notices = [line for line in printed if line.startswith("[Speech output failed")]
    assert len(notices) == 1  # told once, not per sentence
    assert len(playback.played) == 2  # the healthy sentence still played


async def test_interruption_drops_the_rest_and_reports_it() -> None:
    playback = DevicePlayback(hold=True)
    tts = ScriptedTTS(chunks=3, delay=0.005)
    output, observer = voice_output(tts, playback, EventBus(), [])

    await output.say("A long first sentence.")
    await output.say("A second sentence.")
    await output.say("A third sentence.")
    flushed = asyncio.create_task(output.flush())
    await eventually(lambda: playback.played)
    playback.stop()  # what native barge-in does

    assert not await asyncio.wait_for(flushed, WAIT)
    assert observer.calls == ["started", "interrupted"]
    assert output.interruptions == 1
    played_before = len(playback.played)
    await asyncio.sleep(0.05)
    assert len(playback.played) == played_before  # nothing queued survived

    playback.release()
    await output.say("A fresh answer.")
    assert await asyncio.wait_for(output.flush(), WAIT)
    await output.aclose()


async def test_a_crashed_worker_surfaces_instead_of_hanging() -> None:
    class Exploding(RecordingObserver):
        def playback_started(self) -> None:
            raise RuntimeError("observer bug")

    output = VoiceOutput(
        tts=ScriptedTTS(),
        speaker=Speaker(DevicePlayback(), poll_interval=POLL),
        bus=EventBus(),
        terminal=TerminalOutput("Jarvis", write=lambda _text: None),
        observer=Exploding(),
    )
    await output.say("Hello.")
    with pytest.raises(RuntimeError, match="observer bug"):
        await asyncio.wait_for(output.flush(), WAIT)
    with pytest.raises(RuntimeError, match="crashed"):
        await output.say("Again.")
    await output.aclose()


async def test_notices_are_printed_not_spoken() -> None:
    printed: list[str] = []
    playback = DevicePlayback()
    output, _ = voice_output(ScriptedTTS(), playback, EventBus(), printed)

    await output.notice("Loading")

    assert printed == ["[Loading]\n"]
    assert playback.played == []


# --- system voice and confirmation ----------------------------------------


@dataclass
class FakeOutput:
    said: list[str] = field(default_factory=list)
    flushes: int = 0

    async def say(self, sentence: str) -> None:
        self.said.append(sentence)

    async def notice(self, text: str) -> None:
        self.said.append(f"[{text}]")

    async def flush(self) -> bool:
        self.flushes += 1
        return True


async def test_system_voice_speaks_in_full_with_barge_in_held() -> None:
    state = StateMachine(EventBus())
    playback = DevicePlayback()
    barge_in = BargeInController(
        engine=AudioEngine(BargeInEngine()), playback=playback, state=state
    )
    output = FakeOutput()
    held: list[bool] = []

    class Spy(FakeOutput):
        async def say(self, sentence: str) -> None:
            barge_in.playback_started()
            held.append(barge_in._timer is None)

    await SystemVoice(Spy(), barge_in.suspended)("Sorry?")
    await SystemVoice(output, barge_in.suspended)("Sorry, I didn't catch that.")

    assert held == [True]
    assert output.said == ["Sorry, I didn't catch that."]
    assert output.flushes == 1


@pytest.mark.parametrize(
    ("answer", "verdict"),
    [
        ("Yes.", True),
        ("yes, please do", True),
        ("No thanks", False),
        ("Nope.", False),
        ("maybe later", None),
        ("", None),
    ],
)
def test_spoken_yes_no(answer: str, verdict: bool | None) -> None:
    assert parse_spoken_yes_no(answer) is verdict


@dataclass
class ScriptedAnswers:
    answers: list[str]

    async def listen_for_answer(self) -> str:
        return self.answers.pop(0) if self.answers else ""


async def test_confirmation_is_asked_aloud_and_answered_by_voice() -> None:
    spoken: list[str] = []

    async def speak(text: str) -> None:
        spoken.append(text)

    ask = VoiceConfirmation(listener=ScriptedAnswers(["hmm", "Yes, go ahead."]), speak=speak)

    assert await ask("Delete the file report.txt?")
    assert spoken == ["Delete the file report.txt?", PLEASE_ANSWER]


async def test_confirmation_defaults_to_no_when_nothing_clear_is_heard() -> None:
    async def speak(_text: str) -> None:
        return None

    ask = VoiceConfirmation(listener=ScriptedAnswers([]), speak=speak, attempts=2)

    assert not await ask("Send the email?")


async def test_earcon_waits_for_the_tone() -> None:
    playback = DevicePlayback()
    tone: Samples = make_earcon(playback.rate)

    await Earcon(Speaker(playback, poll_interval=POLL), tone)()

    assert len(playback.played) == 1
    assert playback.played[0].size == tone.size
