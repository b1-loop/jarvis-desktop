"""Detecting a launch without a console, and giving its output somewhere to go.

``pythonw.exe`` (and ``uvw.exe``, which starts it) run Jarvis with no
console: ``sys.stdout`` and ``sys.stderr`` are ``None``, so a single
``print`` or a structlog line to stderr raises and the process dies
before anyone sees why. A shortcut that goes through ``uvw run`` may
instead start the ``jarvis`` console script with a hidden console, which
has valid but invisible streams; that launch passes ``--background``.
Either way the session must log to a file and never read the terminal.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

from jarvis.config import Settings
from jarvis.utils.logging import LogFile, get_logger

log = get_logger(__name__)

LOG_SUBDIR: Final = "logs"
LOG_FILENAME: Final = "jarvis.log"


def is_windowless(*, background: bool, stdout: object | None) -> bool:
    """Whether this launch has no usable console.

    Args:
        background: ``--background`` was passed (shortcuts and autostart).
        stdout: ``sys.stdout``; ``None`` under ``pythonw``.
    """
    return background or stdout is None


def log_file_for(settings: Settings) -> LogFile:
    """The rotating log file of the windowless app, under ``data_dir/logs``."""
    return LogFile(
        path=(settings.data_dir / LOG_SUBDIR / LOG_FILENAME).resolve(),
        max_bytes=settings.log_file_max_bytes,
        backups=settings.log_file_backups,
    )


def log_path_for(settings: Settings) -> Path:
    """Absolute path of the active log file, for notifications and the tooltip."""
    return log_file_for(settings).path


class LogWriter:
    """A ``Writer`` (terminal text sink) that turns lines into log events.

    The voice front end and the terminal output write prompts, notices and
    the ``you> ...`` / ``Jarvis> ...`` transcript as streamed chunks. With
    no console, each completed line becomes one ``terminal.line`` event so
    the log shows what the terminal would have shown.

    Args:
        event: Event name of the emitted log lines.
    """

    def __init__(self, event: str = "terminal.line") -> None:
        self._event = event
        self._pending = ""

    def __call__(self, text: str) -> None:
        """Buffer ``text`` and log every line it completes."""
        self._pending += text
        *lines, self._pending = self._pending.split("\n")
        for line in lines:
            self._emit(line)

    def flush(self) -> None:
        """Log a trailing partial line, if any."""
        line, self._pending = self._pending, ""
        self._emit(line)

    def _emit(self, line: str) -> None:
        if line.strip():
            log.info(self._event, text=line.rstrip())


# Design notes:
# - Detection is "flag or no stdout", not a check for a console window via
#   GetConsoleWindow: the hidden console uv creates is a real console, so
#   only the launcher knows the user cannot see it, and the flag says so.
# - The log path is resolved to an absolute path because notifications and
#   the tray tooltip show it to the user, and data_dir is relative to the
#   working directory the shortcut sets (the repository root).
