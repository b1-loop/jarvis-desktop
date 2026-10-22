"""Restarting the windowless app from the HUD, single-instance aware.

Some settings (the LLM provider, the Ollama model, the speech language)
only take effect when the assistant is rebuilt. The HUD's Restart button
calls :meth:`Relaunch.request` and quits; once the session has shut down
and released the single-instance lock, ``app.background`` calls
:meth:`Relaunch.run_if_requested`, which starts a new detached
``python -m jarvis --background``. Starting it any earlier would hand over
to the dying instance (``single_instance``) and exit at once.
"""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Final

from jarvis.utils.logging import get_logger

log = get_logger(__name__)

BACKGROUND_FLAG: Final = "--background"
NO_WAKE_WORD_FLAG: Final = "--no-wakeword"

Spawner = Callable[[Sequence[str], Path], None]
"""Starts ``command`` in ``cwd`` without waiting for it."""


def relaunch_command(*, wake_word: bool, executable: str = sys.executable) -> list[str]:
    """The command line of the new windowless instance.

    ``sys.executable`` is the interpreter running now (the venv's
    ``pythonw.exe`` under ``uvw``), so the new instance uses the same
    environment and packages without going through uv again.
    """
    command = [executable, "-m", "jarvis", BACKGROUND_FLAG]
    if not wake_word:
        command.append(NO_WAKE_WORD_FLAG)
    return command


def spawn_detached(command: Sequence[str], cwd: Path) -> None:
    """Start ``command`` detached: no console, its own process group, no handles shared."""
    if sys.platform == "win32":
        flags = (
            subprocess.DETACHED_PROCESS
            | subprocess.CREATE_NEW_PROCESS_GROUP
            | subprocess.CREATE_NO_WINDOW
        )
        subprocess.Popen(  # noqa: S603 - our own interpreter and module, no user input
            list(command),
            cwd=cwd,
            creationflags=flags,
            close_fds=True,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    else:
        subprocess.Popen(  # noqa: S603 - our own interpreter and module, no user input
            list(command), cwd=cwd, start_new_session=True, close_fds=True
        )


class Relaunch:
    """Remembers a restart request and performs it after shutdown.

    Args:
        wake_word: Whether this session listens for the wake word; the new
            one is started the same way.
        spawn: Starts the new process (a fake in tests: no process is ever
            started by the test suite).
        cwd: Working directory of the new process (``.env`` and ``data/``
            resolve against it).
    """

    def __init__(
        self, *, wake_word: bool, spawn: Spawner = spawn_detached, cwd: Path | None = None
    ) -> None:
        self._wake_word = wake_word
        self._spawn = spawn
        self._cwd = cwd or Path.cwd()
        self._requested = False

    @property
    def requested(self) -> bool:
        """Whether a restart was asked for."""
        return self._requested

    def request(self) -> None:
        """Ask for a restart once this session has ended."""
        self._requested = True
        log.info("desktop.restart.requested")

    def run_if_requested(self) -> bool:
        """Start the new instance if asked to; call after releasing the lock.

        Returns:
            True if a new instance was started.
        """
        if not self._requested:
            return False
        command = relaunch_command(wake_word=self._wake_word)
        log.info("desktop.restart.spawning", command=command)
        self._spawn(command, self._cwd)
        return True


# Design notes:
# - No retry loop and no waiting for the new instance: if it fails to start
#   it logs (and notifies) like any windowless start, and the user can use
#   the desktop shortcut. A restart that waits would keep this process,
#   and its microphone, alive for longer than necessary.
# - Console sessions do not offer Restart: a terminal cannot be reattached
#   to a new process, so the HUD says to quit and start again instead.
