import asyncio
from collections.abc import Callable
from contextlib import AsyncExitStack
from dataclasses import replace
from pathlib import Path

import pytest

from jarvis.audio.engine import AudioModelError
from jarvis.audio.models import AudioModelPaths
from jarvis.config import RuntimeMode, Settings
from jarvis.core import voice_bootstrap
from jarvis.core.bootstrap import FrontEndRequest, Overrides
from jarvis.core.combined_input import CombinedInput
from jarvis.core.events import EventBus
from jarvis.core.state_machine import StateMachine
from jarvis.core.voice_bootstrap import (
    VoiceFactories,
    VoiceFrontEndBuilder,
    VoiceUnavailableError,
    build_stt,
    build_tts,
    native_audio_error,
    run_voice_mode,
)
from jarvis.core.voice_io import VoiceInput
from jarvis.memory.vector_store import InMemoryVectorStore
from jarvis.os_integration.errors import PlatformError
from jarvis.os_integration.stub import StubPlatform
from jarvis.ui.overlay import Overlay, StubOverlay
from jarvis.utils.errors import ConfigError, NativeUnavailableError
from tests.agent.llm_fakes import ScriptedLLM
from tests.core.fakes import RecordingOutput, ScriptedInput
from tests.core.voice_fakes import BargeInEngine, DevicePlayback, ScriptedSTT, ScriptedTTS

MakeSettings = Callable[..., Settings]


class FailingPlatform(StubPlatform):
    async def register_hotkey(self, spec: str) -> int:
        raise PlatformError(f"{spec} is taken by another application")


class WarmSTT(ScriptedSTT):
    warmed: int = 0

    async def warm_up(self) -> None:
        self.warmed += 1


def factories(
    *,
    engine: Callable[..., BargeInEngine] | None = None,
    playback: Callable[[int], DevicePlayback] | None = None,
    stt: ScriptedSTT | None = None,
) -> VoiceFactories:
    native = BargeInEngine()

    def default_engine(_paths: AudioModelPaths, **_options: object) -> BargeInEngine:
        return native

    return VoiceFactories(
        native_error=lambda: OSError,
        engine=engine or default_engine,
        playback=playback or (lambda _rate: DevicePlayback()),
        stt=lambda _settings: stt or ScriptedSTT(),
        tts=lambda _settings: ScriptedTTS(),
        overlay=lambda _settings: Overlay(StubOverlay()),
        autostart=lambda _settings: None,
        poll_interval=0.001,
    )


def request(
    settings: Settings, stack: AsyncExitStack, platform: StubPlatform | None = None
) -> FrontEndRequest:
    bus = EventBus()
    return FrontEndRequest(
        settings, bus, StateMachine(bus), platform or StubPlatform(), ScriptedLLM(), stack
    )


async def test_full_front_end_has_wake_word_hotkey_and_warm_stt(
    make_settings: MakeSettings,
) -> None:
    printed: list[str] = []
    stt = WarmSTT()
    builder = VoiceFrontEndBuilder(factories=factories(stt=stt), write=printed.append)
    async with AsyncExitStack() as stack:
        front = await builder(request(make_settings(), stack))

        assert isinstance(front.source, CombinedInput)
        assert isinstance(front.source.voice, VoiceInput)
        assert "Say 'Hey Jarvis' or press ctrl+alt+j" in front.greeting
        assert stt.warmed == 1
        assert printed == [voice_bootstrap.STT_LOADING]
        assert front.toggle_overlay is not None
        assert front.microphone is not None


async def test_wake_word_model_failure_falls_back_to_the_hotkey(
    make_settings: MakeSettings,
) -> None:
    native = BargeInEngine()
    asked: list[bool] = []

    def engine(
        _paths: AudioModelPaths,
        *,
        wake_word: bool,
        min_speech_ms: int | None,
        wake_word_threshold: float | None = None,
    ) -> BargeInEngine:
        asked.append(wake_word)
        if wake_word:
            raise AudioModelError("hey_jarvis_v0.1.onnx is corrupt")
        return native

    builder = VoiceFrontEndBuilder(factories=factories(engine=engine), write=lambda _t: None)
    async with AsyncExitStack() as stack:
        front = await builder(request(make_settings(), stack))

    assert asked == [True, False]
    assert front.greeting.startswith(
        "The wake word is unavailable (hey_jarvis_v0.1.onnx is corrupt)"
    )
    assert "Press ctrl+alt+j" in front.greeting


async def test_no_wakeword_flag_is_hotkey_only_without_a_warning(
    make_settings: MakeSettings,
) -> None:
    asked: list[bool] = []

    def engine(
        _paths: AudioModelPaths,
        *,
        wake_word: bool,
        min_speech_ms: int | None,
        wake_word_threshold: float | None = None,
    ) -> BargeInEngine:
        asked.append(wake_word)
        return BargeInEngine()

    builder = VoiceFrontEndBuilder(wake_word=False, factories=factories(engine=engine))
    async with AsyncExitStack() as stack:
        front = await builder(request(make_settings(), stack))

    assert asked == [False]
    assert front.greeting.startswith("Press ctrl+alt+j")


async def test_missing_vad_model_makes_voice_unavailable(make_settings: MakeSettings) -> None:
    def engine(_paths: AudioModelPaths, **_options: object) -> BargeInEngine:
        raise ConfigError("audio models missing (silero_vad.onnx)")

    builder = VoiceFrontEndBuilder(factories=factories(engine=engine))
    async with AsyncExitStack() as stack:
        with pytest.raises(VoiceUnavailableError, match="silero_vad"):
            await builder(request(make_settings(), stack))


async def test_missing_microphone_makes_voice_unavailable(make_settings: MakeSettings) -> None:
    class NoDevice(BargeInEngine):
        def start(self) -> None:
            raise OSError("no capture device")

    builder = VoiceFrontEndBuilder(factories=factories(engine=lambda _p, **_o: NoDevice()))
    async with AsyncExitStack() as stack:
        with pytest.raises(VoiceUnavailableError, match="microphone"):
            await builder(request(make_settings(), stack))


async def test_missing_speakers_make_voice_unavailable(make_settings: MakeSettings) -> None:
    def playback(_rate: int) -> DevicePlayback:
        raise OSError("no output device")

    builder = VoiceFrontEndBuilder(factories=factories(playback=playback))
    async with AsyncExitStack() as stack:
        with pytest.raises(VoiceUnavailableError, match="output device"):
            await builder(request(make_settings(), stack))


async def test_hotkey_conflict_keeps_the_wake_word(make_settings: MakeSettings) -> None:
    builder = VoiceFrontEndBuilder(factories=factories())
    async with AsyncExitStack() as stack:
        front = await builder(request(make_settings(), stack, FailingPlatform()))

    assert "Hey Jarvis" in front.greeting


async def test_no_wake_word_and_no_hotkey_is_unavailable(make_settings: MakeSettings) -> None:
    builder = VoiceFrontEndBuilder(wake_word=False, factories=factories())
    async with AsyncExitStack() as stack:
        with pytest.raises(VoiceUnavailableError, match="no wake word and no hotkey"):
            await builder(request(make_settings(), stack, FailingPlatform()))


async def test_microphone_switch_closes_the_device_and_ignores_triggers(
    make_settings: MakeSettings,
) -> None:
    native = BargeInEngine()
    builder = VoiceFrontEndBuilder(factories=factories(engine=lambda _p, **_o: native))
    async with AsyncExitStack() as stack:
        front = await builder(request(make_settings(), stack))
        assert front.microphone is not None and isinstance(front.source, CombinedInput)
        voice = front.source.voice
        assert isinstance(voice, VoiceInput)

        assert await front.microphone.toggle()
        assert voice.muted
        assert not native.running()
        assert not await front.microphone.toggle()
        assert native.running()


async def test_the_hud_gets_a_settings_page_and_follows_mute(make_settings: MakeSettings) -> None:
    native = StubOverlay()
    voice = replace(factories(), overlay=lambda _settings: Overlay(native))
    builder = VoiceFrontEndBuilder(factories=voice)
    async with AsyncExitStack() as stack:
        front = await builder(request(make_settings(), stack))
        items = native.settings["items"]
        assert isinstance(items, list)
        rows = {item["key"]: item for item in items}
        # ScriptedTTS cannot switch voices live, so the voice is restart-only.
        assert rows["tts_voice"]["restart_required"] is True
        assert rows["overlay_theme"]["restart_required"] is False
        assert front.bind_controls is not None

        assert front.microphone is not None
        await front.microphone.toggle()
        assert native.muted


async def test_the_wake_word_setting_turns_the_wake_word_off(make_settings: MakeSettings) -> None:
    builder = VoiceFrontEndBuilder(factories=factories())
    async with AsyncExitStack() as stack:
        front = await builder(request(make_settings(wake_word_enabled=False), stack))
        assert "Press" in front.greeting
        assert "Say" not in front.greeting


async def test_overlay_toggle_flips_visibility(make_settings: MakeSettings) -> None:
    overlay = Overlay(StubOverlay())
    voice = replace(factories(), overlay=lambda _settings: overlay)
    builder = VoiceFrontEndBuilder(factories=voice)
    async with AsyncExitStack() as stack:
        front = await builder(request(make_settings(), stack))
        assert front.toggle_overlay is not None
        was_visible = overlay.visible

        front.toggle_overlay()
        assert overlay.visible is not was_visible
        front.toggle_overlay()
        assert overlay.visible is was_visible


def test_native_audio_error_reports_a_missing_build(monkeypatch: pytest.MonkeyPatch) -> None:
    def not_built(name: str) -> object:
        raise NativeUnavailableError(name, "not built")

    monkeypatch.setattr(voice_bootstrap._native, "load", not_built)

    with pytest.raises(VoiceUnavailableError, match="jarvis_audio"):
        native_audio_error()


def test_speech_provider_problems_become_voice_unavailable(
    monkeypatch: pytest.MonkeyPatch, make_settings: MakeSettings
) -> None:
    def missing(_settings: Settings) -> object:
        raise ConfigError("piper voice not found")

    monkeypatch.setattr(voice_bootstrap.providers, "build_stt", missing)
    monkeypatch.setattr(voice_bootstrap.providers, "build_tts", missing)
    settings = make_settings(mode=RuntimeMode.LOCAL)

    with pytest.raises(VoiceUnavailableError, match="speech recognition"):
        build_stt(settings)
    with pytest.raises(VoiceUnavailableError, match="speech output"):
        build_tts(settings)


async def test_run_voice_mode_falls_back_to_text_with_the_reason(
    make_settings: MakeSettings, tmp_path: Path
) -> None:
    def broken() -> type[Exception]:
        raise VoiceUnavailableError("jarvis_audio is not built")

    printed: list[str] = []
    output = RecordingOutput()
    overrides = Overrides(
        source=ScriptedInput(["exit"]),
        sink=output,
        llm=ScriptedLLM(),
        platform=StubPlatform(),
        vectors=InMemoryVectorStore(),
    )
    settings = make_settings(llm_provider="ollama", data_dir=tmp_path)

    await asyncio.wait_for(
        run_voice_mode(
            settings,
            factories=replace(factories(), native_error=broken),
            overrides=overrides,
            write=printed.append,
        ),
        10,
    )

    assert "Voice mode is unavailable: jarvis_audio is not built" in printed[0]
    assert output.said == ["Goodbye."]


def test_build_overlay_starts_the_boot_sequence(make_settings: MakeSettings) -> None:
    overlay = voice_bootstrap.build_overlay(
        make_settings(overlay_enabled=False, overlay_theme="amber")
    )

    assert isinstance(overlay._native, StubOverlay)
    assert overlay._native.boot_started


def test_build_overlay_honours_a_disabled_boot_animation(make_settings: MakeSettings) -> None:
    overlay = voice_bootstrap.build_overlay(
        make_settings(overlay_enabled=False, overlay_boot_animation=False)
    )

    assert isinstance(overlay._native, StubOverlay)
    assert not overlay._native.boot_started


def test_build_overlay_passes_the_boot_readout(
    make_settings: MakeSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(voice_bootstrap, "default_microphone_name", lambda: "Test Mic")
    overlay = voice_bootstrap.build_overlay(
        make_settings(overlay_enabled=False, ollama_model="qwen2.5:7b", llm_provider="ollama")
    )

    assert isinstance(overlay._native, StubOverlay)
    assert overlay._native.boot_info == [
        ("MODEL", "qwen2.5:7b"),
        ("MODE", "CLOUD"),
        ("MIC", "Test Mic"),
    ]


def test_boot_readout_names_the_model_of_each_mode(
    make_settings: MakeSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(voice_bootstrap, "default_microphone_name", lambda: None)

    claude = voice_bootstrap.boot_readout(
        make_settings(llm_provider="anthropic", llm_model="claude-test")
    )
    local = voice_bootstrap.boot_readout(make_settings(mode=RuntimeMode.LOCAL))

    assert claude["MODEL"] == "claude-test"
    assert claude["MIC"] == "system default"
    assert local["MODE"] == "LOCAL"
    assert local["MODEL"].endswith(".gguf")
