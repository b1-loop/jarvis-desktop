"""Composition of the voice front end: microphone, STT, TTS, speaker, barge-in.

``core/bootstrap.py`` builds the shared core (bus, memory, LLM, tools,
platform, agent, orchestrator) and asks a front-end builder for input and
output. :class:`VoiceFrontEndBuilder` is that builder for voice mode. It is
the one place that knows the concrete audio classes, and it implements the
degradation ladder from CLAUDE.md:

* ``jarvis_audio`` not built, VAD model missing, no audio device, or LOCAL
  speech providers not available yet -> :class:`VoiceUnavailableError`, and
  :func:`run_voice_mode` falls back to text mode with a clear message.
* wake-word models missing or broken -> warn once, hotkey-only.
* TTS failing at runtime -> printed answers (``VoiceOutput``).
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Final, Protocol, runtime_checkable

from jarvis import _native
from jarvis.audio.engine import (
    AudioEngine,
    AudioModelError,
    NativeEngine,
    NativePlayback,
    Speaker,
    create_native_engine,
    create_native_playback,
)
from jarvis.audio.models import AudioModelPaths
from jarvis.config import Settings
from jarvis.core import providers
from jarvis.core.bootstrap import (
    FrontEnd,
    FrontEndRequest,
    Overrides,
    build_application,
    run_text_mode,
)
from jarvis.core.interruption import BargeInController
from jarvis.core.orchestrator import Orchestrator, TerminalOutput, Writer
from jarvis.core.overlay_bridge import OverlayBridge
from jarvis.core.voice_io import (
    Earcon,
    SystemVoice,
    VoiceConfirmation,
    VoiceInput,
    VoiceOutput,
    make_earcon,
)
from jarvis.core.voice_latency import FirstTokenProbe, TurnTimeline
from jarvis.core.voice_triggers import AnyTrigger, HotkeyTrigger, TurnTrigger, WakeWordTrigger
from jarvis.os_integration.errors import PlatformError
from jarvis.stt.base import STT
from jarvis.tts.base import TTS
from jarvis.ui.overlay import Overlay, OverlayConfig, OverlayTheme, create_overlay
from jarvis.utils.errors import ConfigError, JarvisError, NativeUnavailableError
from jarvis.utils.logging import get_logger

log = get_logger(__name__)

MS_PER_SECOND: Final = 1000.0
GREETING: Final = (
    "Say '{wake}' or press {hotkey}, then speak. Talk over me to interrupt; Ctrl+C quits."
)
HOTKEY_GREETING: Final = "Press {hotkey}, then speak. Talk over me to interrupt; Ctrl+C quits."
WAKE_WORD_UNAVAILABLE: Final = "The wake word is unavailable ({reason}); the hotkey works. "
TEXT_FALLBACK: Final = (
    "Voice mode is unavailable: {reason}\nStarting text mode instead; type your requests.\n"
)
STT_LOADING: Final = "[Loading the speech recognition model...]\n"


class VoiceUnavailableError(JarvisError):
    """Voice mode cannot run on this machine right now; text mode still can."""


class EngineFactory(Protocol):
    """Builds the native capture engine (``audio.engine.create_native_engine``)."""

    def __call__(
        self, paths: AudioModelPaths, *, wake_word: bool, min_speech_ms: int | None
    ) -> NativeEngine:
        """Load the models and return an unstarted engine."""


@runtime_checkable
class _WarmsUp(Protocol):
    async def warm_up(self) -> None: ...


def native_audio_error() -> type[Exception]:
    """The ``jarvis_audio.NativeError`` class (device and model failures).

    Raises:
        VoiceUnavailableError: If the extension module is not built.
    """
    try:
        module = _native.load("jarvis_audio")
    except NativeUnavailableError as error:
        raise VoiceUnavailableError(str(error)) from error
    error_type: type[Exception] = module.NativeError
    return error_type


def build_stt(settings: Settings) -> STT:
    """The mode's STT (faster-whisper or whisper.cpp), via ``core.providers``.

    Raises:
        VoiceUnavailableError: If the provider's package, build or model is missing.
    """
    try:
        return providers.build_stt(settings)
    except ConfigError as error:
        raise VoiceUnavailableError(f"speech recognition unavailable: {error}") from error


def build_tts(settings: Settings) -> TTS:
    """The mode's TTS (edge-tts or Piper), via ``core.providers``.

    Raises:
        VoiceUnavailableError: If the provider's package, build or voice is missing.
    """
    try:
        return providers.build_tts(settings)
    except ConfigError as error:
        raise VoiceUnavailableError(f"speech output unavailable: {error}") from error


def build_overlay(settings: Settings) -> Overlay:
    """The native overlay when enabled and available, otherwise the stub.

    Starts the boot sequence right away; ``play_boot`` is idempotent, so a
    launcher that also asks for it at startup changes nothing.
    """
    config = OverlayConfig(
        theme=OverlayTheme(settings.overlay_theme),
        boot_animation=settings.overlay_boot_animation,
    )
    overlay = create_overlay(config, enabled=settings.overlay_enabled)
    overlay.play_boot()
    return overlay


def _stdout_write(text: str) -> None:
    sys.stdout.write(text)
    sys.stdout.flush()


@dataclass(frozen=True, slots=True)
class VoiceFactories:
    """Every concrete thing the voice front end creates; tests swap in fakes."""

    native_error: Callable[[], type[Exception]] = native_audio_error
    engine: EngineFactory = create_native_engine
    playback: Callable[[int], NativePlayback] = create_native_playback
    stt: Callable[[Settings], STT] = build_stt
    tts: Callable[[Settings], TTS] = build_tts
    overlay: Callable[[Settings], Overlay] = build_overlay
    poll_interval: float = 0.01


@dataclass(frozen=True, slots=True)
class _Audio:
    engine: AudioEngine
    wake_word_problem: str | None
    wake_word_active: bool


@dataclass(frozen=True, slots=True)
class _OverlayParts:
    overlay: Overlay
    bridge: OverlayBridge

    def toggle(self) -> None:
        if self.overlay.visible:
            self.overlay.hide()
        else:
            self.overlay.show()


class VoiceMicrophone:
    """``MicrophoneSwitch`` for the tray: closes the device and ignores triggers while muted."""

    def __init__(self, engine: AudioEngine, voice_input: VoiceInput) -> None:
        self._engine = engine
        self._input = voice_input

    async def toggle(self) -> bool:
        """Flip mute; returns True when now muted."""
        muted = not self._input.muted
        self._input.muted = muted
        if muted:
            await self._engine.stop()
        else:
            await self._engine.start()
        return muted


class VoiceFrontEndBuilder:
    """``FrontEndBuilder`` for voice mode.

    Args:
        wake_word: Listen for the wake word (``--no-wakeword`` turns it off).
        factories: Concrete constructors; the defaults build the real thing.
        write: Terminal output for notices and the transcript.
    """

    def __init__(
        self,
        *,
        wake_word: bool = True,
        factories: VoiceFactories | None = None,
        write: Writer | None = None,
    ) -> None:
        self._wake_word = wake_word
        self._factories = factories or VoiceFactories()
        self._write = write or _stdout_write

    async def __call__(self, request: FrontEndRequest) -> FrontEnd:
        """Open the audio devices and assemble voice input and output.

        Raises:
            VoiceUnavailableError: If voice cannot run (see the module docs).
        """
        native_error = self._factories.native_error()
        audio = await self._start_audio(request, native_error)
        tts = self._factories.tts(request.settings)
        try:
            speaker = Speaker(self._factories.playback(tts.sample_rate))
        except native_error as error:
            raise VoiceUnavailableError(f"no audio output device: {error}") from error
        stt = await self._load_stt(request.settings)
        trigger = await self._trigger(request, audio)
        overlay = self._overlay(request, audio.engine, speaker)
        return self._assemble(
            request=request,
            audio=audio,
            speaker=speaker,
            stt=stt,
            tts=tts,
            trigger=trigger,
            overlay=overlay,
        )

    async def _start_audio(self, request: FrontEndRequest, native_error: type[Exception]) -> _Audio:
        settings = request.settings
        native, problem = self._create_engine(settings)
        engine = AudioEngine(
            native,
            poll_interval=self._factories.poll_interval,
            max_utterance_seconds=settings.max_utterance_seconds,
        )
        try:
            await request.stack.enter_async_context(engine)
        except native_error as error:
            raise VoiceUnavailableError(f"no usable microphone: {error}") from error
        active = self._wake_word and problem is None
        return _Audio(engine, problem, active)

    def _create_engine(self, settings: Settings) -> tuple[NativeEngine, str | None]:
        """The engine with the wake word if possible, else VAD-only plus the reason."""
        paths = AudioModelPaths.in_directory(settings.models_dir)
        problem: str | None = None
        if self._wake_word:
            try:
                return self._engine(paths, settings, wake_word=True), None
            except (ConfigError, AudioModelError) as error:
                log.warning("voice.wake_word.unavailable", error=str(error))
                problem = str(error)
        try:
            return self._engine(paths, settings, wake_word=False), problem
        except (ConfigError, AudioModelError) as error:
            raise VoiceUnavailableError(f"speech detection unavailable: {error}") from error

    def _engine(
        self, paths: AudioModelPaths, settings: Settings, *, wake_word: bool
    ) -> NativeEngine:
        return self._factories.engine(
            paths, wake_word=wake_word, min_speech_ms=settings.vad_min_speech_ms
        )

    async def _load_stt(self, settings: Settings) -> STT:
        stt = self._factories.stt(settings)
        if isinstance(stt, _WarmsUp):
            self._write(STT_LOADING)
            await stt.warm_up()
        return stt

    async def _trigger(self, request: FrontEndRequest, audio: _Audio) -> TurnTrigger:
        triggers: list[TurnTrigger] = []
        if audio.wake_word_active:
            triggers.append(WakeWordTrigger(audio.engine))
        hotkey = HotkeyTrigger(request.platform, request.settings.hotkey)
        try:
            await hotkey.register()
        except PlatformError as error:
            log.warning("voice.hotkey.unavailable", hotkey=hotkey.spec, error=str(error))
            if not triggers:
                raise VoiceUnavailableError(f"no wake word and no hotkey: {error}") from error
        else:
            request.stack.push_async_callback(hotkey.aclose)
            triggers.append(hotkey)
        return AnyTrigger(triggers)

    def _overlay(
        self, request: FrontEndRequest, engine: AudioEngine, speaker: Speaker
    ) -> _OverlayParts:
        overlay = self._factories.overlay(request.settings)
        request.stack.callback(overlay.close)
        bridge = OverlayBridge(
            overlay,
            request.bus,
            input_level=lambda: engine.input_level,
            output_level=lambda: speaker.output_level,
        )
        bridge.start()
        request.stack.push_async_callback(bridge.aclose)
        return _OverlayParts(overlay, bridge)

    def _assemble(
        self,
        *,
        request: FrontEndRequest,
        audio: _Audio,
        speaker: Speaker,
        stt: STT,
        tts: TTS,
        trigger: TurnTrigger,
        overlay: _OverlayParts,
    ) -> FrontEnd:
        settings = request.settings
        terminal = TerminalOutput(settings.assistant_name, write=self._write)
        output = VoiceOutput(
            tts=tts,
            speaker=speaker,
            bus=request.bus,
            terminal=terminal,
            on_sentence=overlay.bridge.show_response,
        )
        request.stack.push_async_callback(output.aclose)
        barge_in = BargeInController(
            engine=audio.engine,
            playback=speaker.native,
            state=request.state,
            echo_guard_seconds=settings.barge_in_echo_guard_ms / MS_PER_SECOND,
        )
        output.set_observer(barge_in)
        system_voice = SystemVoice(output, barge_in.suspended)
        timeline = TurnTimeline(request.bus)
        voice_input = self._input(
            request=request,
            audio=audio,
            speaker=speaker,
            stt=stt,
            trigger=trigger,
            timeline=timeline,
            system_voice=system_voice,
        )

        def attach(orchestrator: Orchestrator) -> None:
            barge_in.attach(turns=orchestrator, capture=voice_input)

        return FrontEnd(
            source=voice_input,
            sink=output,
            ask=VoiceConfirmation(listener=voice_input, speak=system_voice),
            llm=FirstTokenProbe(request.llm, timeline),
            greeting=self._greeting(settings, audio),
            attach=attach,
            toggle_overlay=overlay.toggle,
            microphone=VoiceMicrophone(audio.engine, voice_input),
        )

    def _input(
        self,
        *,
        request: FrontEndRequest,
        audio: _Audio,
        speaker: Speaker,
        stt: STT,
        trigger: TurnTrigger,
        timeline: TurnTimeline,
        system_voice: SystemVoice,
    ) -> VoiceInput:
        settings = request.settings
        return VoiceInput(
            engine=audio.engine,
            stt=stt,
            trigger=trigger,
            state=request.state,
            bus=request.bus,
            timeline=timeline,
            earcon=Earcon(speaker, make_earcon(speaker.sample_rate)),
            announce=system_voice,
            listen_timeout=settings.listen_timeout_seconds,
            min_chars=settings.min_transcript_chars,
            echo=self._write,
        )

    def _greeting(self, settings: Settings, audio: _Audio) -> str:
        hotkey = settings.hotkey
        if audio.wake_word_active:
            return GREETING.format(wake=settings.wake_word.title(), hotkey=hotkey)
        greeting = HOTKEY_GREETING.format(hotkey=hotkey)
        if audio.wake_word_problem is None:
            return greeting
        return WAKE_WORD_UNAVAILABLE.format(reason=audio.wake_word_problem) + greeting


async def run_voice_mode(
    settings: Settings,
    *,
    wake_word: bool = True,
    factories: VoiceFactories | None = None,
    overrides: Overrides | None = None,
    write: Writer | None = None,
) -> None:
    """Build the assistant with the voice front end and converse until Ctrl+C.

    Falls back to text mode, after saying why, when voice cannot run here.
    """
    write = write or _stdout_write
    builder = VoiceFrontEndBuilder(wake_word=wake_word, factories=factories, write=write)
    base = overrides or Overrides()
    try:
        application = await build_application(settings, replace(base, front_end=builder))
    except VoiceUnavailableError as error:
        log.warning("voice.unavailable", reason=str(error))
        write(TEXT_FALLBACK.format(reason=error))
        await run_text_mode(settings, base)
        return
    await application.run()


# Design notes:
# - A separate module from bootstrap.py so text mode never imports the
#   audio stack, and so the voice wiring can be read in one place.
# - Fallback to text mode rebuilds the core (memory, tools) from scratch:
#   the half-built voice application already released everything through
#   its exit stack, and assembling twice costs a few milliseconds, which is
#   simpler than a partially reusable application.
# - Cleanup order (reverse of registration): voice output workers stop,
#   then the overlay, the hotkey, and finally the audio engine closes the
#   microphone; the event bus (bootstrap) closes after all of them.
# - Speech providers come from core/providers.py, so --mode local speaks
#   through whisper.cpp and Piper; a missing prerequisite there (a
#   ConfigError naming the fix) becomes VoiceUnavailableError and text mode.
# - The tray and dashboard are built by bootstrap.build_ui for both front
#   ends; this module only contributes the overlay toggle and the
#   microphone switch their menu needs.
# - Muting stops the audio engine (the device is really closed, which
#   Windows shows in its microphone indicator) and makes the input ignore
#   hotkey presses until unmuted.
