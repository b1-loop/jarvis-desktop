"""Structured logging built on structlog.

Jarvis logs key/value events rather than prose so that the dashboard, the
latency benchmark and a human reading the terminal all consume the same
stream. Secrets are redacted by key name before any renderer sees them,
which is cheaper and safer than remembering to scrub at every call site.
"""

from __future__ import annotations

import logging
import sys
from typing import Final, cast

import structlog
from structlog.typing import EventDict, FilteringBoundLogger, WrappedLogger

_SECRET_KEY_MARKERS: Final[tuple[str, ...]] = ("key", "token", "secret", "password")
_REDACTED: Final = "***"


def redact_secrets(_logger: WrappedLogger, _method: str, event_dict: EventDict) -> EventDict:
    """Replace the value of any key that looks like a credential.

    Matching is by substring on the lower-cased key so ``api_key``,
    ``ANTHROPIC_API_KEY`` and ``refresh_token`` are all caught without a
    maintained allow-list.
    """
    for key in event_dict:
        lowered = key.lower()
        if any(marker in lowered for marker in _SECRET_KEY_MARKERS):
            event_dict[key] = _REDACTED
    return event_dict


def configure_logging(level: str = "INFO", *, json_output: bool = False) -> None:
    """Configure structlog for the whole process.

    Args:
        level: Standard logging level name, case-insensitive.
        json_output: Emit one JSON object per line instead of the coloured
            console format. Used when logs are piped to a file or the
            dashboard.

    Raises:
        KeyError: If ``level`` is not a known logging level name.
    """
    min_level = logging.getLevelNamesMapping()[level.upper()]
    renderer: structlog.typing.Processor = (
        structlog.processors.JSONRenderer() if json_output else structlog.dev.ConsoleRenderer()
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
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=False,
    )


def get_logger(name: str) -> FilteringBoundLogger:
    """Return a logger bound to ``name``.

    The name is attached as the ``logger`` key so events from different
    modules can be filtered in the dashboard.
    """
    return cast(FilteringBoundLogger, structlog.get_logger().bind(logger=name))


# Design notes:
# - structlog over stdlib logging: the pipeline emits events with timings and
#   identifiers that downstream consumers (dashboard, benchmark) parse; prose
#   log lines would need to be parsed back into fields.
# - cache_logger_on_first_use is off so tests can reconfigure freely. The cost
#   is negligible for an application that logs tens of events per second.
# - Redaction is a processor, not a convention, because a convention is only
#   as good as the most tired contributor.
