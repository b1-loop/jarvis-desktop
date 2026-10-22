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

It also decides the session's barge-in mode (``core/barge_in_mode.py``):
with ``JARVIS_BARGE_IN_MODE=auto``, talking over Jarvis interrupts it on
headphones, and only the wake word (or hotkey, or typing) does on speakers.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Final, Protocol, runtime_checkable

import httpx

from jarvis import _native
from jarvis.app.autostart import StartupShortcut
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
from jarvis.config import HudPosition, RuntimeMode, Settings
from jarvis.core import providers
from jarvis.core.barge_in_mode import BargeInMode, resolve_barge_in_mode
from jarvis.core.bootstrap import (
    FrontEnd,
    FrontEndRequest,
    Overrides,
    build_application,
    run_text_mode,
)
from jarvis.core.combined_input import CombinedAnswers, CombinedInput
from jarvis.core.echo_detector import EchoDetector
from jarvis.core.hud_wiring import HudActions
from jarvis.core.interruption import BargeInController, InterruptPolicy, SpeakingInputs
from jarvis.core.orchestrator import Orchestrator, TerminalOutput, Writer
from jarvis.core.overlay_bridge import OverlayBridge
from jarvis.core.reply_output import SwitchableOutput, TextReplies
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
from jarvis.os_integration.audio_endpoint import default_output_endpoint
from jarvis.os_integration.base import OutputEndpoint
from jarvis.os_integration.errors import PlatformError
from jarvis.stt.base import STT
from jarvis.tts.base import TTS
from jarvis.ui.hud_settings import AppearanceSwitch, Applier, Autostart, SettingsController
from jarvis.ui.model_catalog import OllamaModels
from jarvis.ui.overlay import (
    Overlay,
    OverlayConfig,
    OverlayIntensity,
    OverlaySize,
    OverlayTheme,
    SavedPosition,
    create_overlay,
    default_microphone_name,
)
from jarvis.user_settings import UserSettingsStore, user_settings_path
from jarvis.utils.errors import ConfigError, JarvisError, NativeUnavailableError
from jarvis.utils.logging import get_logger

log = get_logger(__name__)

MS_PER_SECOND: Final = 1000.0
GREETING: Final = "Say '{wake}' or press {hotkey}, then speak. {interrupt}; Ctrl+C quits."
HOTKEY_GREETING: Final = "Press {hotkey}, then speak. {interrupt}; Ctrl+C quits."
INTERRUPT_BY_VOICE: Final = "Talk over me to interrupt"
INTERRUPT_BY_WAKE_WORD: Final = "Say '{wake}' to interrupt me"
INTERRUPT_BY_HOTKEY: Final = "Press {hotkey} to interrupt me"
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
        self,
        paths: AudioModelPaths,
        *,
        wake_word: bool,
        min_speech_ms: int | None,
        wake_word_threshold: float | None = None,
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


def boot_readout(settings: Settings) -> dict[str, str]:
    """The real values the overlay's boot sequence shows in its systems check.

    The microphone is the default capture device, which is the one the audio
    engine records from; "system default" when its name cannot be read.
    """
    if settings.mode is RuntimeMode.LOCAL:
        model = settings.local_llm_model.name
    elif settings.llm_provider == "anthropic":
        model = settings.llm_model
    else:
        model = settings.ollama_model
    return {
        "MODEL": model,
        "MODE": settings.mode.value.upper(),
        "MIC": default_microphone_name() or "system default",
    }


def build_overlay(settings: Settings) -> Overlay:
    """The native overlay when enabled and available, otherwise the stub.

    Starts the boot sequence right away with the systems-check readout;
    ``play_boot`` is idempotent, so a launcher that also asks for it at
    startup changes nothing.
    """
    overlay = create_overlay(overlay_config(settings), enabled=settings.overlay_enabled)
    overlay.play_boot(boot_readout(settings))
    return overlay


def overlay_config(settings: Settings) -> OverlayConfig:
    """The overlay's configuration from the settings, remembered positions included."""
    return OverlayConfig(
        theme=OverlayTheme(settings.overlay_theme),
        boot_animation=settings.overlay_boot_animation,
        intensity=OverlayIntensity(settings.overlay_intensity),
        size=OverlaySize(settings.overlay_size),
        interactive=settings.overlay_interactive,
        position=_saved(settings.overlay_position),
        expanded_bounds=_saved(settings.overlay_expanded_bounds),
    )


def _saved(position: HudPosition | None) -> SavedPosition | None:
    if position is None:
        return None
    return SavedPosition(
        monitor=position.monitor,
        x=position.x,
        y=position.y,
        width=position.width,
        height=position.height,
    )


def startup_shortcut(settings: Settings) -> Autostart | None:
    """The Startup-folder shortcut on Windows; elsewhere there is none to offer."""
    if sys.platform != "win32":
        return None
    return StartupShortcut(settings.data_dir)


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
    autostart: Callable[[Settings], Autostart | None] = startup_shortcut
    output_endpoint: Callable[[], OutputEndpoint | None] = default_output_endpoint
    microphone_name: Callable[[], str | None] = default_microphone_name
    poll_interval: float = 0.01


@dataclass(frozen=True, slots=True)
class SessionHooks:
    """What only the launcher knows how to do, offered to the HUD's settings page.

    Attributes:
        open_log: Opens the log file; None when the session logs to a terminal.
        restart: Arranges a relaunch after exit; None for a console session.
    """

    open_log: Callable[[], None] | None = None
    restart: Callable[[], None] | None = None


@dataclass(frozen=True, slots=True)
class _Audio:
    engine: AudioEngine
    wake_word_problem: str | None
    wake_word_active: bool


@dataclass(frozen=True, slots=True)
class _Triggers:
    turn: TurnTrigger
    hotkey: HotkeyTrigger | None


@dataclass(frozen=True, slots=True)
class _OverlayParts:
    overlay: Overlay
    bridge: OverlayBridge

    def toggle(self) -> None:
        if self.overlay.visible:
            self.overlay.hide()
        else:
            self.overlay.show()


@dataclass(frozen=True, slots=True)
class _Inputs:
    voice: VoiceInput
    combined: CombinedInput
    barge_in: BargeInController


@dataclass(frozen=True, slots=True)
class _HudParts:
    """What the HUD's live settings act on."""

    audio: _Audio
    tts: TTS
    replies: SwitchableOutput
    inputs: _Inputs


@runtime_checkable
class _VoiceSwitch(Protocol):
    def set_voice(self, voice: str) -> None: ...


def live_appliers(overlay: Overlay, settings: Settings, parts: _HudParts) -> dict[str, Applier]:
    """The settings this session can change live, each with its applier.

    Settings without one here are offered as "applies after restart":
    the voice only with edge-tts (not Piper), the wake-word threshold only
    while the wake word runs.
    """
    appearance = AppearanceSwitch(overlay, settings)
    appliers: dict[str, Applier] = {
        "overlay_size": appearance.set_size,
        "overlay_theme": appearance.set_theme,
        "overlay_intensity": appearance.set_intensity,
        "speak_replies": parts.replies.set_speak,
    }
    if parts.audio.wake_word_active:
        appliers["wake_word_threshold"] = parts.audio.engine.set_wake_word_threshold
    if isinstance(parts.tts, _VoiceSwitch):
        appliers["tts_voice"] = parts.tts.set_voice
    return appliers


class VoiceMicrophone:
    """``MicrophoneSwitch`` for the tray and the HUD: closes the device and ignores triggers.

    Args:
        engine: The audio engine to stop and start.
        voice_input: Ignores triggers while muted.
        on_change: Told the new state (the HUD's microphone control).
    """

    def __init__(
        self,
        engine: AudioEngine,
        voice_input: VoiceInput,
        *,
        on_change: Callable[[bool], None] | None = None,
    ) -> None:
        self._engine = engine
        self._input = voice_input
        self._on_change = on_change

    async def toggle(self) -> bool:
        """Flip mute; returns True when now muted."""
        muted = not self._input.muted
        self._input.muted = muted
        if muted:
            await self._engine.stop()
        else:
            await self._engine.start()
        if self._on_change is not None:
            self._on_change(muted)
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
        hooks: SessionHooks | None = None,
    ) -> None:
        self._wake_word = wake_word
        self._factories = factories or VoiceFactories()
        self._write = write or _stdout_write
        self._hooks = hooks or SessionHooks()
        self._mode: BargeInMode | None = None

    async def __call__(self, request: FrontEndRequest) -> FrontEnd:
        """Open the audio devices and assemble voice input and output.

        Raises:
            VoiceUnavailableError: If voice cannot run (see the module docs).
        """
        # --no-wakeword and the HUD's "Wake word" switch both turn it off.
        self._wake_word = self._wake_word and request.settings.wake_word_enabled
        native_error = self._factories.native_error()
        audio = await self._start_audio(request, native_error)
        tts = self._factories.tts(request.settings)
        try:
            speaker = Speaker(self._factories.playback(tts.sample_rate))
        except native_error as error:
            raise VoiceUnavailableError(f"no audio output device: {error}") from error
        stt = await self._load_stt(request.settings)
        triggers = await self._trigger(request, audio)
        overlay = self._overlay(request, audio.engine, speaker)
        return self._assemble(
            request=request,
            audio=audio,
            speaker=speaker,
            stt=stt,
            tts=tts,
            triggers=triggers,
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
            paths,
            wake_word=wake_word,
            min_speech_ms=settings.vad_min_speech_ms,
            wake_word_threshold=settings.wake_word_threshold,
        )

    async def _load_stt(self, settings: Settings) -> STT:
        stt = self._factories.stt(settings)
        if isinstance(stt, _WarmsUp):
            self._write(STT_LOADING)
            await stt.warm_up()
        return stt

    async def _trigger(self, request: FrontEndRequest, audio: _Audio) -> _Triggers:
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
            return _Triggers(AnyTrigger(triggers), None)
        request.stack.push_async_callback(hotkey.aclose)
        triggers.append(hotkey)
        return _Triggers(AnyTrigger(triggers), hotkey)

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
        triggers: _Triggers,
        overlay: _OverlayParts,
    ) -> FrontEnd:
        output, replies = self._outputs(request, speaker, tts, overlay)
        barge_in = self._barge_in(request, audio, speaker, triggers)
        output.set_observer(barge_in)
        # System phrases follow "speak replies" too: a silent typed session
        # must not suddenly ask a confirmation question out loud.
        system_voice = SystemVoice(replies, barge_in.suspended)
        timeline = TurnTimeline(request.bus)
        voice_input = self._input(
            request=request,
            audio=audio,
            speaker=speaker,
            stt=stt,
            trigger=triggers.turn,
            timeline=timeline,
            system_voice=system_voice,
        )
        inputs = _Inputs(
            voice=voice_input,
            combined=CombinedInput(voice_input, silence=speaker.stop, echo=self._write),
            barge_in=barge_in,
        )
        request.stack.push_async_callback(inputs.combined.aclose)
        parts = _HudParts(audio=audio, tts=tts, replies=replies, inputs=inputs)
        return self._front_end(request, overlay, parts, system_voice, timeline)

    def _barge_in(
        self, request: FrontEndRequest, audio: _Audio, speaker: Speaker, triggers: _Triggers
    ) -> BargeInController:
        """The controller for this session's barge-in mode (decided once, logged)."""
        settings = request.settings
        policy = InterruptPolicy(
            mode=self._barge_in_mode(settings),
            wake_threshold=settings.wake_word_threshold,
            speaking_wake_margin=settings.speaking_wake_margin,
            self_wake_guard_seconds=settings.self_wake_guard_ms / MS_PER_SECOND,
            assistant_name=settings.assistant_name,
        )
        echo = EchoDetector(
            ratio=settings.barge_in_echo_ratio,
            window_seconds=settings.barge_in_echo_window_ms / MS_PER_SECOND,
        )
        engine = audio.engine
        inputs = SpeakingInputs(
            events=engine,
            hotkey=triggers.hotkey,
            levels=lambda: (engine.input_level, speaker.output_level),
            echo=echo,
        )
        return BargeInController(
            engine=engine,
            playback=speaker.native,
            state=request.state,
            echo_guard_seconds=settings.barge_in_echo_guard_ms / MS_PER_SECOND,
            policy=policy,
            inputs=inputs,
        )

    def _barge_in_mode(self, settings: Settings) -> BargeInMode:
        if self._mode is None:
            self._mode = resolve_barge_in_mode(
                settings.barge_in_mode,
                endpoint=self._factories.output_endpoint,
                microphone=self._factories.microphone_name,
            )
        return self._mode

    def _outputs(
        self, request: FrontEndRequest, speaker: Speaker, tts: TTS, overlay: _OverlayParts
    ) -> tuple[VoiceOutput, SwitchableOutput]:
        """The speaking output and the switch in front of it ("speak replies")."""
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
        silent = TextReplies(terminal, overlay.bridge.show_response)
        return output, SwitchableOutput(output, silent, speak=settings.speak_replies)

    def _front_end(
        self,
        request: FrontEndRequest,
        overlay: _OverlayParts,
        parts: _HudParts,
        system_voice: SystemVoice,
        timeline: TurnTimeline,
    ) -> FrontEnd:
        inputs = parts.inputs
        microphone = VoiceMicrophone(
            parts.audio.engine, inputs.voice, on_change=overlay.overlay.set_muted
        )
        hud = self._hud(request, overlay, parts, microphone)

        def attach(orchestrator: Orchestrator) -> None:
            inputs.barge_in.attach(turns=orchestrator, capture=inputs.voice)
            inputs.combined.attach(orchestrator)

        answers = CombinedAnswers(inputs.voice, inputs.combined)
        return FrontEnd(
            source=inputs.combined,
            sink=parts.replies,
            ask=VoiceConfirmation(listener=answers, speak=system_voice),
            llm=FirstTokenProbe(request.llm, timeline),
            greeting=self._greeting(request.settings, parts.audio),
            attach=attach,
            toggle_overlay=overlay.toggle,
            microphone=microphone,
            bind_controls=hud.bind_controls,
        )

    def _hud(
        self,
        request: FrontEndRequest,
        overlay: _OverlayParts,
        parts: _HudParts,
        microphone: VoiceMicrophone,
    ) -> HudActions:
        """The settings page and the HUD's actions, connected to the bridge."""
        settings = request.settings
        store = UserSettingsStore(user_settings_path(settings.data_dir))
        http = httpx.AsyncClient()
        request.stack.push_async_callback(http.aclose)
        controller = SettingsController(
            settings=settings,
            store=store,
            view=overlay.overlay,
            appliers=live_appliers(overlay.overlay, settings, parts),
            models=OllamaModels(http, settings.ollama_base_url),
            autostart=self._factories.autostart(settings),
        )
        controller.push()
        actions = HudActions(
            overlay=overlay.overlay,
            typed=parts.inputs.combined,
            settings=controller,
            store=store,
            microphone=microphone,
            open_log=self._hooks.open_log,
            restart=self._hooks.restart,
        )
        overlay.bridge.attach_handlers(actions)
        return actions

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
        hotkey, wake = settings.hotkey, settings.wake_word.title()
        interrupt = self._interrupt_hint(audio).format(wake=wake, hotkey=hotkey)
        if audio.wake_word_active:
            return GREETING.format(wake=wake, hotkey=hotkey, interrupt=interrupt)
        greeting = HOTKEY_GREETING.format(hotkey=hotkey, interrupt=interrupt)
        if audio.wake_word_problem is None:
            return greeting
        return WAKE_WORD_UNAVAILABLE.format(reason=audio.wake_word_problem) + greeting

    def _interrupt_hint(self, audio: _Audio) -> str:
        """How to interrupt Jarvis in this session, for the greeting."""
        if self._mode is BargeInMode.VOICE:
            return INTERRUPT_BY_VOICE
        if self._mode is BargeInMode.WAKE_WORD and audio.wake_word_active:
            return INTERRUPT_BY_WAKE_WORD
        return INTERRUPT_BY_HOTKEY


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
# - The barge-in mode is resolved once per builder (and logged): the output
#   device is read through jarvis_os before anything is spoken, so the
#   greeting can already say how to interrupt in this session.
# - Muting stops the audio engine (the device is really closed, which
#   Windows shows in its microphone indicator) and makes the input ignore
#   hotkey presses until unmuted.
