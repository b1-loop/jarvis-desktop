"""Structured logging built on structlog.

Jarvis logs key/value events rather than prose so that the dashboard, the
latency benchmark and a human reading the terminal all consume the same
stream. Secrets are redacted by key name before any renderer sees them,
which is cheaper and safer than remembering to scrub at every call site.
"""

from __future__ import annotations

import io
import logging
import logging.handlers
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, TextIO, cast

import structlog
from structlog.typing import EventDict, FilteringBoundLogger, WrappedLogger

_SECRET_KEY_MARKERS: Final[tuple[str, ...]] = ("key", "token", "secret", "password")
_REDACTED: Final = "***"
_FILE_LOGGER_NAME: Final = "jarvis.file"
DEFAULT_LOG_FILE_MAX_BYTES: Final = 1_000_000
DEFAULT_LOG_FILE_BACKUPS: Final = 3


@dataclass(frozen=True, slots=True)
class LogFile:
    """Where a windowless process writes its log, and how much of it to keep.

    Attributes:
        path: The active log file; rotated copies get ``.1``, ``.2``... suffixes.
        max_bytes: Size at which the file is rotated.
        backups: How many rotated copies to keep; older ones are deleted.
    """

    path: Path
    max_bytes: int = DEFAULT_LOG_FILE_MAX_BYTES
    backups: int = DEFAULT_LOG_FILE_BACKUPS


def redact_secrets(_logger: WrappedLogger, _method: str, event_dict: EventDict) -> EventDict:
    """Replace the value of any key that looks like a credential.

    Matching is by suffix on the lower-cased key so ``api_key``,
    ``ANTHROPIC_API_KEY`` and ``refresh_token`` are all caught without a
    maintained allow-list, while counters such as ``input_tokens`` (which
    only contain a marker) stay readable.
    """
    for key in event_dict:
        lowered = key.lower()
        if lowered.endswith(_SECRET_KEY_MARKERS):
            event_dict[key] = _REDACTED
    return event_dict


def configure_logging(
    level: str = "INFO",
    *,
    json_output: bool = False,
    stream: TextIO | None = None,
    log_file: LogFile | None = None,
) -> None:
    """Configure structlog for the whole process.

    Args:
        level: Standard logging level name, case-insensitive.
        json_output: Emit one JSON object per line instead of the coloured
            console format. Used when logs are piped to a file or the
            dashboard.
        stream: Where log lines go; ``sys.stderr`` by default. Characters
            its encoding cannot represent are escaped, never fatal.
        log_file: Write to this size-capped rotating file instead of a
            stream. Used by the windowless desktop app, which has no
            stderr (``pythonw``/``uvw``). File lines are JSON, never coloured.

    Raises:
        KeyError: If ``level`` is not a known logging level name.
        OSError: If ``log_file``'s directory cannot be created or opened.
    """
    min_level = logging.getLevelNamesMapping()[level.upper()]
    renderer: structlog.typing.Processor = (
        structlog.processors.JSONRenderer()
        if json_output or log_file is not None
        else structlog.dev.ConsoleRenderer()
    )
    factory: Callable[..., Any] = (
        structlog.PrintLoggerFactory(file=tolerant(stream or sys.stderr))
        if log_file is None
        else _file_logger_factory(log_file)
    )
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            redact_secrets,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(min_level),
        logger_factory=factory,
        cache_logger_on_first_use=False,
    )


def _file_logger_factory(log_file: LogFile) -> Callable[..., logging.Logger]:
    """A structlog logger factory writing rendered lines to a rotating file.

    structlog renders the whole event to one string; the standard library's
    ``RotatingFileHandler`` only stores it and rotates by size. The stdlib
    logger is private (no propagation) so third-party logging is unaffected,
    and handlers from a previous configuration are closed, not leaked.
    """
    log_file.path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.handlers.RotatingFileHandler(
        log_file.path,
        maxBytes=log_file.max_bytes,
        backupCount=log_file.backups,
        encoding="utf-8",
    )
    handler.setFormatter(logging.Formatter("%(message)s"))
    close_file_logging()
    file_logger = logging.getLogger(_FILE_LOGGER_NAME)
    file_logger.addHandler(handler)
    file_logger.setLevel(logging.DEBUG)
    file_logger.propagate = False
    return lambda *_args: file_logger


def close_file_logging() -> None:
    """Flush and close the rotating log file, if one is configured. Idempotent."""
    file_logger = logging.getLogger(_FILE_LOGGER_NAME)
    for handler in list(file_logger.handlers):
        file_logger.removeHandler(handler)
        handler.close()


def tolerant(stream: TextIO) -> TextIO:
    """Make ``stream`` escape unencodable characters instead of raising.

    Redirected stderr on Windows uses the ANSI code page (cp1252 here), so a
    log line carrying a localised OS error or any character outside that
    page raised ``UnicodeEncodeError`` from inside a logging call.
    """
    if isinstance(stream, io.TextIOWrapper):
        stream.reconfigure(errors="backslashreplace")
    return stream


def get_logger(name: str) -> FilteringBoundLogger:
    """Return a lazy logger bound to ``name``.

    The name is attached as the ``logger`` key so events from different
    modules can be filtered in the dashboard. Modules call this at import
    time, long before ``configure_logging`` runs, so the returned object
    must not resolve the configuration yet: the proxy looks it up again on
    every call, which is what lets ``JARVIS_LOG_LEVEL`` reach loggers that
    already exist.
    """
    return cast(FilteringBoundLogger, _LazyLogger(name))


class _LazyLogger:
    """Stands in for a bound logger and resolves the current configuration per call.

    ``logger=name`` cannot go into ``structlog.get_logger``'s initial values
    (it collides with ``wrap_logger``'s own ``logger`` parameter), and
    ``.bind`` on structlog's proxy resolves the configuration at once, so
    this small proxy defers the ``bind`` to each logging call instead.
    """

    __slots__ = ("_name",)

    def __init__(self, name: str) -> None:
        self._name = name

    def __getattr__(self, attribute: str) -> Any:
        return getattr(structlog.get_logger().bind(logger=self._name), attribute)


# Design notes:
# - structlog over stdlib logging: the pipeline emits events with timings and
#   identifiers that downstream consumers (dashboard, benchmark) parse; prose
#   log lines would need to be parsed back into fields.
# - cache_logger_on_first_use is off so tests can reconfigure freely. The cost
#   is negligible for an application that logs tens of events per second.
# - get_logger returns a proxy that binds on every call instead of binding
#   once: ``bind`` resolves structlog's own lazy proxy, so a module-level
#   ``log = get_logger(__name__)`` kept the default configuration
#   (everything at DEBUG, to stdout) for the life of the process and
#   JARVIS_LOG_LEVEL=WARNING still printed info lines. One extra ``bind``
#   per event is noise next to rendering it.
# - The output stream escapes what its encoding cannot represent
#   (``backslashreplace``): a log line must never crash the caller, and the
#   escaped text still shows which character it was. Streams that are not
#   ``TextIOWrapper`` (pytest's capture, StringIO) accept any text already.
# - File logging goes through a private stdlib logger only for its
#   RotatingFileHandler: structlog still renders (and redacts) every line,
#   and the stdlib's rotation is proven where a hand-rolled one would need
#   tests for every edge. structlog calls level methods (info, error...)
#   that stdlib loggers have; ``exception`` arrives as ``error`` with the
#   traceback already rendered by format_exc_info, so it is not doubled.
# - Redaction is a processor, not a convention, because a convention is only
#   as good as the most tired contributor.
