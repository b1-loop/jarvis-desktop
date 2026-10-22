"""Start the local Ollama server when the windowless app needs it and it is down.

With the default provider (Ollama, ADR-0020) an assistant that starts at
login is useless until ``ollama serve`` runs, and Ollama's own tray app is
not always set to start with Windows. Before building the assistant the
windowless session therefore probes ``/api/tags``; if nothing answers it
starts Ollama hidden and detached (the official tray app when installed,
otherwise ``ollama serve``) and waits until it answers.

Every effect sits behind a Protocol (probe, process launcher, executable
lookup, sleep and clock) so the decision logic is tested without spawning
anything or opening sockets.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Final, Protocol

import httpx

from jarvis.utils.errors import JarvisError
from jarvis.utils.logging import get_logger

log = get_logger(__name__)

PROBE_TIMEOUT_S: Final = 2.0
"""A local server that is up answers /api/tags in milliseconds."""
POLL_INTERVAL_S: Final = 0.5
TAGS_PATH: Final = "/api/tags"
OLLAMA_APP: Final = Path("Programs") / "Ollama" / "ollama app.exe"
"""The official installer's tray app, relative to ``%LOCALAPPDATA%``."""
OLLAMA_EXECUTABLE: Final = "ollama"
SERVE_ARGUMENT: Final = "serve"

if sys.platform == "win32":
    _HIDDEN_DETACHED: Final = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
    _BREAKAWAY: Final = subprocess.CREATE_BREAKAWAY_FROM_JOB
else:
    _HIDDEN_DETACHED = 0
    _BREAKAWAY = 0


class OllamaStartError(JarvisError):
    """Ollama is not answering and could not be started; the message says why."""


class OllamaStatus(StrEnum):
    """What :meth:`OllamaLauncher.ensure_running` found or did."""

    ALREADY_RUNNING = "already running"
    STARTED = "started"


class OllamaProbe(Protocol):
    """Asks the server whether it is up."""

    async def is_up(self, wait_seconds: float) -> bool:
        """True when ``/api/tags`` answered successfully within ``wait_seconds``."""


class ProcessLauncher(Protocol):
    """Starts a program without a window and without waiting for it."""

    def spawn(self, argv: Sequence[str]) -> None:
        """Start ``argv`` hidden and detached.

        Raises:
            OSError: If the program cannot be started.
        """


ExecutableLookup = Callable[[str], str | None]
"""``shutil.which``: resolve a program name on PATH."""


def find_ollama_command(
    environ: Mapping[str, str], which: ExecutableLookup = shutil.which
) -> tuple[str, ...] | None:
    """The command that starts Ollama here, or ``None`` when it is not installed.

    The tray app is preferred: it is what the installer sets up, it keeps
    the server alive across crashes, and the user can see and quit it.
    ``ollama serve`` on PATH covers manual and winget installs.
    """
    local_app_data = environ.get("LOCALAPPDATA")
    if local_app_data:
        app = Path(local_app_data) / OLLAMA_APP
        if app.is_file():
            return (str(app),)
    executable = which(OLLAMA_EXECUTABLE)
    if executable is not None:
        return (executable, SERVE_ARGUMENT)
    return None


@dataclass(frozen=True, slots=True)
class LaunchTiming:
    """How long to wait at each step.

    Attributes:
        probe_timeout: Per-request timeout of the "is it up?" check.
        start_timeout: How long a freshly started server may take to answer.
        poll_interval: Pause between checks while waiting.
    """

    probe_timeout: float = PROBE_TIMEOUT_S
    start_timeout: float = 20.0
    poll_interval: float = POLL_INTERVAL_S


class OllamaLauncher:
    """Makes sure an Ollama server answers, starting one if needed.

    Args:
        probe: Checks the server.
        launcher: Starts the process.
        command: Finds the command to start (see :func:`find_ollama_command`).
        timing: Timeouts and poll interval.
        sleep: ``asyncio.sleep``; injected so tests do not wait.
        clock: Monotonic seconds; injected so tests control the deadline.
    """

    def __init__(
        self,
        probe: OllamaProbe,
        launcher: ProcessLauncher,
        command: Callable[[], tuple[str, ...] | None],
        *,
        timing: LaunchTiming | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._probe = probe
        self._launcher = launcher
        self._command = command
        self._timing = timing or LaunchTiming()
        self._sleep = sleep
        self._clock = clock

    async def ensure_running(self) -> OllamaStatus:
        """Return once Ollama answers.

        Raises:
            OllamaStartError: If it is not installed, cannot be started, or
                does not answer within ``timing.start_timeout``.
        """
        if await self._probe.is_up(self._timing.probe_timeout):
            return OllamaStatus.ALREADY_RUNNING
        command = self._command()
        if command is None:
            raise OllamaStartError("Ollama is not running and is not installed")
        log.info("ollama.starting", command=list(command))
        try:
            await asyncio.to_thread(self._launcher.spawn, command)
        except OSError as error:
            raise OllamaStartError(f"Ollama could not be started: {error}") from error
        await self._wait_until_up()
        log.info("ollama.started")
        return OllamaStatus.STARTED

    async def _wait_until_up(self) -> None:
        deadline = self._clock() + self._timing.start_timeout
        while self._clock() < deadline:
            await self._sleep(self._timing.poll_interval)
            if await self._probe.is_up(self._timing.probe_timeout):
                return
        raise OllamaStartError(
            f"Ollama was started but did not answer within {self._timing.start_timeout:g} s"
        )


class HttpOllamaProbe:
    """:class:`OllamaProbe` over ``GET {base_url}/api/tags``.

    Args:
        base_url: The server, ``JARVIS_OLLAMA_BASE_URL``.
    """

    def __init__(self, base_url: str) -> None:
        self._url = base_url.rstrip("/") + TAGS_PATH

    async def is_up(self, wait_seconds: float) -> bool:
        """True on a 2xx answer; connection errors and timeouts mean "not up"."""
        try:
            async with httpx.AsyncClient(timeout=wait_seconds) as client:
                response = await client.get(self._url)
        except httpx.HTTPError as error:
            log.debug("ollama.probe.down", url=self._url, error=str(error))
            return False
        return response.is_success


class DetachedProcessLauncher:
    """:class:`ProcessLauncher` with ``subprocess.Popen``: no window, not a child to wait on."""

    def spawn(self, argv: Sequence[str]) -> None:
        """Start ``argv`` hidden, outside Jarvis's job object when Windows allows it.

        Raises:
            OSError: If the program cannot be started at all.
        """
        try:
            self._popen(argv, _HIDDEN_DETACHED | _BREAKAWAY)
        except PermissionError:
            # The job Jarvis runs in (uv's) forbids breakaway; Ollama then
            # stops when Jarvis does, which is still better than not at all.
            log.info("ollama.spawn.no_breakaway")
            self._popen(argv, _HIDDEN_DETACHED)

    @staticmethod
    def _popen(argv: Sequence[str], flags: int) -> None:
        subprocess.Popen(  # noqa: S603 - fixed program found by find_ollama_command
            list(argv),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=flags,
            close_fds=True,
        )


def create_ollama_launcher(base_url: str, start_timeout: float) -> OllamaLauncher:
    """The real launcher for ``base_url``, finding Ollama in this user's environment."""
    return OllamaLauncher(
        HttpOllamaProbe(base_url),
        DetachedProcessLauncher(),
        lambda: find_ollama_command(os.environ),
        timing=LaunchTiming(start_timeout=start_timeout),
    )


# Design notes:
# - Only localhost is worth starting: the caller checks that the provider
#   is Ollama, and a base URL pointing at another machine simply fails the
#   probe and the wait, which the notification then reports.
# - The Popen object is dropped on purpose: Ollama outlives the call and is
#   not Jarvis's child to reap. CREATE_NO_WINDOW keeps `ollama serve` from
#   flashing a console; CREATE_BREAKAWAY_FROM_JOB lets it survive Jarvis
#   when the parent job (uv's) permits it, and the retry without it keeps
#   Ollama usable for this session when the job does not.
# - Probe failures are "not up", not errors: that is the question being
#   asked. Only the final outcome (not installed, cannot start, no answer
#   in time) is an OllamaStartError, which the windowless app shows in a
#   notification and the tray tooltip.
