"""The windowless desktop session: tray icon only, voice mode, errors made visible.

Started by ``jarvis --background`` (the shortcuts, autostart) or whenever
there is no console (``pythonw``). The sequence is:

1. Take the single-instance lock; if another Jarvis holds it, ask it to
   show its dashboard (or notify "already running") and exit.
2. Listen for such requests from later launches.
3. Start Ollama if it is the LLM and is not answering.
4. Build the assistant in voice mode with the dashboard hidden, the tray
   as the only visible surface and terminal text routed to the log.
5. Notify "Jarvis is listening" and converse until Quit.

When voice cannot start (no native build, no models, no microphone, a
missing key) there is no terminal to fall back to, so instead of exiting
silently the session notifies the reason and the log path, keeps a tray
icon whose tooltip carries the error, and exits when the user picks Quit.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Coroutine
from contextlib import AsyncExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Protocol

from jarvis.app.ollama_launcher import OllamaStartError, OllamaStatus, create_ollama_launcher
from jarvis.app.single_instance import (
    HANDOFF_FILENAME,
    LOCK_FILENAME,
    InstanceLock,
    InstanceServer,
    ask_running_instance_to_show,
)
from jarvis.app.windowless import LogWriter, log_file_for, log_path_for
from jarvis.config import Settings
from jarvis.core.bootstrap import Application, FrontEndBuilder, Overrides, build_application
from jarvis.core.bootstrap import build_platform as build_platform_backend
from jarvis.core.events import EventBus
from jarvis.core.orchestrator import Writer
from jarvis.core.providers import effective_llm_provider
from jarvis.core.voice_bootstrap import VoiceFactories, VoiceFrontEndBuilder, build_overlay
from jarvis.os_integration.base import Platform
from jarvis.os_integration.errors import PlatformError
from jarvis.ui.overlay import Overlay
from jarvis.ui.tray import StubTray, TrayCallbacks, TraySurface, create_tray
from jarvis.utils.logging import close_file_logging, configure_logging, get_logger

log = get_logger(__name__)

NOTIFICATION_TITLE: Final = "Jarvis"
LISTENING_MESSAGE: Final = "Jarvis is listening; say '{wake}' or press {hotkey}."
OLLAMA_PROBLEM: Final = " {reason}; Jarvis cannot answer until Ollama runs."
FAILURE_MESSAGE: Final = "Jarvis could not start listening: {reason}. Details: {log}"
ALREADY_RUNNING_MESSAGE: Final = "Jarvis is already running; its icon is in the notification area."
NOT_LISTENING_NOTICE: Final = "Not listening: {reason}"
OLLAMA_NOTICE: Final = "LLM offline: {reason}"
EXIT_OK: Final = 0
EXIT_FAILED: Final = 2
"""Same code as ``jarvis.__main__.EXIT_STARTUP_FAILED`` for a console launch."""


class InstanceGuard(Protocol):
    """The single-instance lock (:class:`~jarvis.app.single_instance.InstanceLock`)."""

    def acquire(self) -> bool:
        """True when this process now owns the instance."""

    def release(self) -> None:
        """Give the instance up."""


class HandoverServer(Protocol):
    """Answers "show" requests from later launches (``InstanceServer``)."""

    async def start(self) -> None:
        """Start listening and publish how to reach this instance."""

    async def aclose(self) -> None:
        """Stop listening."""


class OllamaStarter(Protocol):
    """Makes sure Ollama answers (``OllamaLauncher``)."""

    async def ensure_running(self) -> OllamaStatus:
        """Return once Ollama answers; raise ``OllamaStartError`` otherwise."""


ErrorTrayFactory = Callable[[EventBus, TrayCallbacks, Settings], Awaitable[TraySurface | None]]
"""Shows a tray icon for the error state; ``None`` when no tray can be shown."""


def boot_overlay(settings: Settings) -> Overlay:
    """The overlay, playing its boot animation when this build has one.

    ``Overlay.play_boot`` and ``overlay_boot_animation`` arrive with the
    overlay-animation work; looking them up by name lets this module work
    with or without it.
    """
    overlay = build_overlay(settings)
    play_boot = getattr(overlay, "play_boot", None)
    if callable(play_boot) and getattr(settings, "overlay_boot_animation", True):
        play_boot()
    return overlay


def voice_front_end(wake_word: bool, write: Writer) -> FrontEndBuilder:
    """The voice front end with terminal output sent to ``write`` (the log)."""
    factories = VoiceFactories(overlay=boot_overlay)
    return VoiceFrontEndBuilder(wake_word=wake_word, factories=factories, write=write)


def ollama_starter(settings: Settings) -> OllamaStarter:
    """The real launcher for ``settings.ollama_base_url``."""
    return create_ollama_launcher(settings.ollama_base_url, settings.ollama_start_timeout_seconds)


async def error_tray(
    bus: EventBus, callbacks: TrayCallbacks, settings: Settings
) -> TraySurface | None:
    """A real tray icon for the error state, or ``None`` when only a stub is possible."""
    tray = await create_tray(bus, callbacks, enabled=settings.tray_enabled)
    if isinstance(tray, StubTray):
        await tray.stop()
        return None
    return tray


@dataclass(frozen=True, slots=True)
class BackgroundDeps:
    """Every effect of the windowless session; tests replace them with fakes."""

    lock: Callable[[Path], InstanceGuard] = InstanceLock
    ping: Callable[[Path], Awaitable[bool]] = ask_running_instance_to_show
    server: Callable[[Callable[[], None], Path], HandoverServer] = InstanceServer
    platform: Callable[[AsyncExitStack], Platform] = build_platform_backend
    application: Callable[[Settings, Overrides], Awaitable[Application]] = build_application
    front_end: Callable[[bool, Writer], FrontEndBuilder] = voice_front_end
    ollama: Callable[[Settings], OllamaStarter] = ollama_starter
    tray: ErrorTrayFactory = error_tray


def background_settings(settings: Settings) -> Settings:
    """``settings`` for a windowless start: the dashboard waits for the tray."""
    return settings.model_copy(update={"dashboard_start_visible": False})


def hotkey_label(hotkey: str) -> str:
    """``ctrl+alt+j`` as Windows writes it: ``Ctrl+Alt+J``."""
    return "+".join(part.strip().capitalize() for part in hotkey.split("+"))


def listening_message(settings: Settings, ollama_problem: str | None = None) -> str:
    """The one start-up notification, with the Ollama problem appended if any."""
    message = LISTENING_MESSAGE.format(
        wake=settings.wake_word.title(), hotkey=hotkey_label(settings.hotkey)
    )
    if ollama_problem is None:
        return message
    return message + OLLAMA_PROBLEM.format(reason=ollama_problem)


def uses_ollama(settings: Settings) -> bool:
    """Whether this session's LLM is the Ollama server."""
    return effective_llm_provider(settings) == "ollama"


async def run_background(
    settings: Settings, *, wake_word: bool = True, deps: BackgroundDeps | None = None
) -> int:
    """Run the windowless session, or hand over to the one already running.

    Returns:
        The process exit code: 0 after Quit or a hand-over, ``EXIT_FAILED``
        when the assistant could not start and the user quit the error tray.
    """
    deps = deps or BackgroundDeps()
    lock = deps.lock(settings.data_dir / LOCK_FILENAME)
    if not lock.acquire():
        await _hand_over(settings, deps)
        return EXIT_OK
    try:
        return await BackgroundSession(settings, wake_word=wake_word, deps=deps).run()
    finally:
        lock.release()


async def _hand_over(settings: Settings, deps: BackgroundDeps) -> None:
    """Ask the running instance to come forward; notify when it cannot answer."""
    if await deps.ping(settings.data_dir / HANDOFF_FILENAME):
        log.info("desktop.handed_over")
        return
    async with AsyncExitStack() as stack:
        await notify(deps.platform(stack), ALREADY_RUNNING_MESSAGE)


async def notify(platform: Platform, message: str) -> None:
    """Show a notification; a failure is logged with its traceback, never raised.

    A notification is the messenger for problems, so it must not become the
    reason the session ends; the tray tooltip and the log still carry them.
    """
    log.info("desktop.notify", message=message)
    try:
        await platform.notify(NOTIFICATION_TITLE, message)
    except PlatformError:
        log.exception("desktop.notify.failed")


class ErrorTrayControls:
    """``TrayCallbacks`` for the error tray: the menu explains, Quit quits.

    Args:
        quit_requested: Set when the user picks Quit.
        explain: Repeats the error (a notification) for "Show dashboard".
    """

    def __init__(self, quit_requested: asyncio.Event, explain: Callable[[], None]) -> None:
        self._quit = quit_requested
        self._explain = explain

    def show_dashboard(self) -> None:
        """There is no dashboard without an assistant; say why again."""
        self._explain()

    def toggle_overlay(self) -> None:
        """Nothing to toggle; logged so a click is visible in the log."""
        log.info("desktop.error_tray.overlay_unavailable")

    def toggle_microphone(self) -> None:
        """Nothing to mute; logged."""
        log.info("desktop.error_tray.microphone_unavailable")

    def quit(self) -> None:
        """End the error state and the process."""
        self._quit.set()


class BackgroundSession:
    """One windowless run, from the hand-over listener to Quit.

    Args:
        settings: Settings as loaded; the dashboard is hidden on top.
        wake_word: Listen for the wake word (``--no-wakeword`` turns it off).
        deps: The effects (see :class:`BackgroundDeps`).
    """

    def __init__(self, settings: Settings, *, wake_word: bool, deps: BackgroundDeps) -> None:
        self._settings = background_settings(settings)
        self._wake_word = wake_word
        self._deps = deps
        self._show: Callable[[], None] = self._show_while_starting
        self._tasks: set[asyncio.Task[None]] = set()
        self._platform: Platform | None = None

    async def run(self) -> int:
        """Start, converse until Quit, and release everything."""
        async with AsyncExitStack() as stack:
            self._platform = self._deps.platform(stack)
            handoff = self._settings.data_dir / HANDOFF_FILENAME
            server = self._deps.server(self.show_requested, handoff)
            await server.start()
            stack.push_async_callback(server.aclose)
            ollama_problem = await self._ensure_ollama()
            try:
                application = await self._deps.application(self._settings, self._overrides())
            except Exception as error:
                log.exception("desktop.start_failed")
                return await self._fail(error)
            return await self._converse(application, ollama_problem)

    def show_requested(self) -> None:
        """A later launch asked this instance to come forward."""
        self._show()

    def _overrides(self) -> Overrides:
        writer = LogWriter()
        front_end = self._deps.front_end(self._wake_word, writer)
        return Overrides(platform=self._platform, front_end=front_end)

    async def _ensure_ollama(self) -> str | None:
        """Start Ollama when it is the LLM; the reason it is down, or ``None``."""
        if not (self._settings.ollama_autostart and uses_ollama(self._settings)):
            return None
        try:
            status = await self._deps.ollama(self._settings).ensure_running()
        except OllamaStartError as error:
            log.warning("desktop.ollama.unavailable", reason=str(error))
            return str(error)
        log.info("desktop.ollama.ready", status=status.value)
        return None

    async def _converse(self, application: Application, ollama_problem: str | None) -> int:
        controls = application.controls
        if controls is not None:
            self._show = controls.show_dashboard
            if ollama_problem is not None and controls.tray is not None:
                controls.tray.set_notice(OLLAMA_NOTICE.format(reason=ollama_problem))
        await notify(self._require_platform(), listening_message(self._settings, ollama_problem))
        try:
            await application.run()
        except Exception as error:
            log.exception("desktop.crashed")
            return await self._fail(error)
        log.info("desktop.stopped")
        return EXIT_OK

    async def _fail(self, error: Exception) -> int:
        """Notify the error, keep an error tray until Quit, return the failure code."""
        reason = (str(error) or type(error).__name__).rstrip(". ")
        message = FAILURE_MESSAGE.format(reason=reason, log=log_path_for(self._settings))
        platform = self._require_platform()
        await notify(platform, message)
        quit_requested = asyncio.Event()
        controls = ErrorTrayControls(
            quit_requested, explain=lambda: self._spawn(notify(platform, message))
        )
        bus = EventBus()
        try:
            await self._hold_error_tray(bus, controls, quit_requested, reason)
        finally:
            await bus.close()
        return EXIT_FAILED

    async def _hold_error_tray(
        self, bus: EventBus, controls: ErrorTrayControls, quit_requested: asyncio.Event, reason: str
    ) -> None:
        tray = await self._deps.tray(bus, controls, self._settings)
        if tray is None:
            log.error("desktop.error_tray.unavailable", reason=reason)
            return
        try:
            tray.set_notice(NOT_LISTENING_NOTICE.format(reason=reason))
            self._show = controls.show_dashboard
            await quit_requested.wait()
        finally:
            await tray.stop()

    def _show_while_starting(self) -> None:
        log.info("desktop.show_requested.while_starting")

    def _spawn(self, work: Coroutine[Any, Any, None]) -> None:
        task = asyncio.get_running_loop().create_task(work)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    def _require_platform(self) -> Platform:
        if self._platform is None:
            raise RuntimeError("BackgroundSession.run() builds the platform first")
        return self._platform


def run_windowless(settings: Settings, *, wake_word: bool = True) -> int:
    """Entry point for a launch without a console: file logging, then the session.

    Returns:
        The process exit code.
    """
    log_file = log_file_for(settings)
    configure_logging(settings.log_level, log_file=log_file)
    log.info("desktop.start", log=str(log_file.path), wake_word=wake_word)
    try:
        return asyncio.run(run_background(settings, wake_word=wake_word))
    except Exception:
        log.exception("desktop.fatal")
        return EXIT_FAILED
    finally:
        close_file_logging()


# Design notes:
# - Voice failure does not fall back to text mode here, unlike the console
#   launch: text mode reads stdin, which a windowless process does not
#   have. The error tray is the windowless equivalent of "say why and
#   carry on": the reason is in a notification, the tooltip and the log.
# - Exceptions while building or running are caught broadly on purpose,
#   and only here: this is the top of a process nobody can see, so every
#   failure is logged with its traceback and shown, then the process waits
#   for Quit instead of vanishing. The console launch keeps its narrower
#   handling in __main__.
# - The platform is built once, outside the assistant (Overrides.platform),
#   so notifications still work after the assistant's own resources were
#   released by a failed start.
# - "Show" requests from a later launch go through self._show, which is
#   re-pointed as the session moves on: a log line while starting, the
#   dashboard while running, the error notification in the error tray.
# - The Ollama check happens before assembly so the first spoken question
#   does not race the server's start-up; a failure is not fatal (Ollama
#   may be started by hand later) and is reported in the start notification
#   and the tray tooltip.
# - boot_overlay looks play_boot and overlay_boot_animation up by name
#   because the boot animation lands in a separate change; with neither
#   present the overlay simply appears as before. Assumption: when the
#   method exists but the setting does not, the animation plays.
# - Native balloon notifications use NIIF_RESPECT_QUIET_TIME but may chime;
#   one notification per start is intended, so no quiet flag is forced.
