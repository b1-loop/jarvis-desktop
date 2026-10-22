"""End to end in voice mode: the real orchestrator, agent, memory and voice front end.

Only the edges are fake: the native capture engine (scripted VAD and wake
word events, behaving like the C++ engine on barge-in), the playback
device, STT, TTS and the model.
"""

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from jarvis.agent.registry import ToolRegistry
from jarvis.audio.models import AudioModelPaths
from jarvis.config import Settings
from jarvis.core.bootstrap import Application, Overrides, build_application
from jarvis.core.events import Event, LatencySample, StateChanged
from jarvis.core.state_machine import AssistantState
from jarvis.core.voice_bootstrap import VoiceFactories, VoiceFrontEndBuilder
from jarvis.core.voice_latency import VOICE_STAGES
from jarvis.llm.base import TextDelta
from jarvis.memory.vector_store import InMemoryVectorStore
from jarvis.os_integration.base import OutputEndpoint
from jarvis.os_integration.stub import StubPlatform
from jarvis.ui.overlay import Overlay, StubOverlay
from tests.agent.llm_fakes import Block, ScriptedLLM, text_round
from tests.audio.fakes import native_event
from tests.core.voice_fakes import (
    BargeInEngine,
    DevicePlayback,
    ScriptedSTT,
    ScriptedTTS,
    eventually,
)

pytestmark = pytest.mark.integration

WAIT = 5.0


@dataclass
class VoiceSession:
    """An assembled voice application plus the fakes that drive it."""

    native: BargeInEngine = field(default_factory=BargeInEngine)
    playback: DevicePlayback = field(default_factory=DevicePlayback)
    stt: ScriptedSTT = field(default_factory=ScriptedSTT)
    tts: ScriptedTTS = field(default_factory=ScriptedTTS)
    printed: list[str] = field(default_factory=list)
    events: list[Event] = field(default_factory=list)

    def factories(self) -> VoiceFactories:
        def engine(
            _paths: AudioModelPaths,
            *,
            wake_word: bool,
            min_speech_ms: int | None,
            wake_word_threshold: float | None = None,
            silence_end_ms: int | None = None,
        ) -> BargeInEngine:
            assert wake_word
            assert min_speech_ms == 250
            assert silence_end_ms == 700
            return self.native

        return VoiceFactories(
            native_error=lambda: OSError,
            engine=engine,
            playback=lambda _rate: self.playback,
            stt=lambda _settings: self.stt,
            tts=lambda _settings: self.tts,
            overlay=lambda _settings: Overlay(StubOverlay()),
            output_endpoint=lambda: OutputEndpoint("Headphones (USB)", "headphones"),
            microphone_name=lambda: None,
            poll_interval=0.001,
        )

    async def build(self, settings: Settings, llm: ScriptedLLM) -> Application:
        builder = VoiceFrontEndBuilder(factories=self.factories(), write=self.printed.append)
        overrides = Overrides(
            llm=llm,
            platform=StubPlatform(),
            vectors=InMemoryVectorStore(),
            registry=ToolRegistry(),
            front_end=builder,
        )
        application = await build_application(settings, overrides)
        application.agent.context.bus.subscribe_callback(Event, self.events.append)
        return application

    def states(self) -> list[AssistantState]:
        return [e.new for e in self.events if isinstance(e, StateChanged)]

    async def speak(self, application: Application, text: str) -> None:
        """Say the wake word, wait for the earcon and listening, then say ``text``."""
        await asyncio.sleep(0.03)  # the input is waiting for a trigger again
        self.stt.texts.append(text)
        self.native.script(native_event("WAKE_WORD_DETECTED", 0.95))
        await eventually(lambda: application.orchestrator.state is AssistantState.LISTENING)
        self.native.say_utterance()


@pytest.fixture
def settings(make_settings: Callable[..., Settings], tmp_path: Path) -> Settings:
    return make_settings(llm_provider="ollama", data_dir=tmp_path, barge_in_echo_guard_ms=10)


async def test_wake_word_turn_is_answered_aloud(settings: Settings) -> None:
    session = VoiceSession()
    llm = ScriptedLLM(rounds=[text_round("It is ", "half past nine, sir.")])
    application = await session.build(settings, llm)
    running = asyncio.create_task(application.run())

    await session.speak(application, "What time is it?")
    await eventually(lambda: "Jarvis> It is half past nine, sir.\n" in session.printed)
    await eventually(lambda: application.orchestrator.state is AssistantState.IDLE)
    await session.speak(application, "Goodbye.")
    await asyncio.wait_for(running, WAIT)

    assert session.states()[:4] == [
        AssistantState.LISTENING,
        AssistantState.THINKING,
        AssistantState.SPEAKING,
        AssistantState.IDLE,
    ]
    assert "you> What time is it?\n" in session.printed
    assert session.tts.log[:2] == [
        "synth:It is half past nine, sir.",
        "synth-done:It is half past nine, sir.",
    ]
    assert "synth:Goodbye." in session.tts.log  # the farewell is spoken too
    stages = {e.stage for e in session.events if isinstance(e, LatencySample)}
    assert set(VOICE_STAGES) <= stages
    assert session.native.stopped == 1  # the microphone was closed on shutdown


async def test_barge_in_cancels_the_turn_and_hears_the_new_request(settings: Settings) -> None:
    session = VoiceSession()
    llm = ScriptedLLM(
        rounds=[
            [TextDelta("Here is a very long story. "), TextDelta("It goes on and on. "), Block()],
            text_round("Sunny and mild."),
        ]
    )
    native_barge_in = settings.model_copy(update={"barge_in_echo_ratio": 0.0})
    application = await session.build(native_barge_in, llm)
    running = asyncio.create_task(application.run())

    await session.speak(application, "Tell me a story.")
    session.playback.hold = True  # after the earcon: the story stays on the device
    targets = session.native.barge_in_targets
    await eventually(lambda: bool(targets) and targets[-1] is not None)  # armed after the guard
    # The user talks over Jarvis: the engine stops playback natively.
    session.stt.texts.append("Stop. What's the weather?")
    session.playback.hold = False
    session.native.script(native_event("SPEECH_STARTED"))
    await eventually(lambda: application.orchestrator.state is not AssistantState.SPEAKING)
    session.native.say_utterance()
    session.native.batches.popleft()  # speech already started; only its end is still to come
    await eventually(lambda: "Jarvis> Sunny and mild.\n" in session.printed)
    await eventually(lambda: application.orchestrator.state is AssistantState.IDLE)
    await session.speak(application, "Goodbye.")
    await asyncio.wait_for(running, WAIT)

    assert session.playback.stops >= 1
    assert llm.closed == 2  # the blocked stream was closed by the cancellation
    assert "you> Stop. What's the weather?\n" in session.printed
    barge = [e for e in session.events if isinstance(e, StateChanged) and e.reason == "barge-in"]
    assert len(barge) == 1
    assert barge[0].old is AssistantState.SPEAKING
    assert "Jarvis> It goes on and on.\n" not in session.printed  # never reached the speaker
