"""What happens when an account needs signing in and nobody is at a terminal.

Jarvis runs in two very different places. In a terminal (``--text-only``,
``scripts/connect_accounts.py``) an OAuth flow can open the browser or print
a device code and the user sees it. In the windowless desktop app and in
voice mode there is no console: a browser popping up from a hidden process
at a random moment, or a device code written to the log, is a sign-in
nobody can complete. This module holds the pieces that make the second case
explicit instead:

* :func:`consent_required` builds the :class:`ConsentRequiredError` a
  non-interactive token source raises, with a spoken-friendly remedy that
  names the exact connect command.
* :class:`NotifyOnConsent` wraps a token source and reports the first such
  error once (the composition root turns that into a tray notification).
* :class:`ConnectionLog` keeps the last sign-in problem per service in the
  token store, so ``connect_accounts.py status`` can show it later.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Final

from jarvis.integrations.base import ConsentRequiredError
from jarvis.integrations.tokens import TokenSource, TokenStore
from jarvis.utils.logging import get_logger

log = get_logger(__name__)

CONNECT_SCRIPT: Final = "uv run python scripts/connect_accounts.py"
"""The command that signs an account in from a terminal (see the README)."""

SERVICE_LABELS: Final[dict[str, str]] = {
    "google": "Google",
    "microsoft": "Microsoft",
    "discord": "Discord",
}

ERROR_KEY_SUFFIX: Final = "-error"

ConsentNotifier = Callable[[ConsentRequiredError], Awaitable[None]]
"""Shows a consent problem to the user (a tray notification in the desktop app)."""


def connect_command(service: str) -> str:
    """``"google"`` -> ``"uv run python scripts/connect_accounts.py google"``."""
    return f"{CONNECT_SCRIPT} {service}"


def consent_required(service: str, reason: str, *, expired: bool = False) -> ConsentRequiredError:
    """The error a non-interactive source raises instead of starting a flow.

    Args:
        service: Store key and connect sub-command (``google``, ``microsoft``).
        reason: Technical cause for the log ("no stored token").
        expired: True when a sign-in existed but can no longer be refreshed,
            which changes the wording from "not connected yet" to "expired".
    """
    label = SERVICE_LABELS.get(service, service)
    state = "sign-in has expired" if expired else "isn't connected yet"
    subject = f"The {label}" if expired else label
    remedy = f"{subject} {state}; run `{connect_command(service)}` in a terminal."
    return ConsentRequiredError(service, reason=reason, remedy=remedy)


class ConnectionLog:
    """The last sign-in problem per service, kept next to its tokens.

    Stored under ``<service>-error`` in the :class:`TokenStore` so it lives
    in the same git-ignored directory and ``disconnect`` removes it with the
    tokens. Only the message is kept, never a token.
    """

    def __init__(self, store: TokenStore) -> None:
        self._store = store

    def record(self, service: str, message: str) -> None:
        """Remember ``message`` as the latest problem for ``service``."""
        entry = {"error": message, "at": datetime.now(UTC).isoformat(timespec="seconds")}
        self._store.save(service + ERROR_KEY_SUFFIX, entry)

    def clear(self, service: str) -> None:
        """Forget the problem after a successful sign-in."""
        self._store.clear(service + ERROR_KEY_SUFFIX)

    def last(self, service: str) -> tuple[str, str] | None:
        """``(message, ISO timestamp)`` of the latest problem, or ``None``."""
        entry = self._store.load(service + ERROR_KEY_SUFFIX)
        if entry is None:
            return None
        return str(entry.get("error", "")), str(entry.get("at", ""))


class NotifyOnConsent:
    """A token source that reports the first :class:`ConsentRequiredError` once.

    Every later request still raises (the tool tells the user each time it
    is asked), but the notification appears once per process: a tray
    balloon per tool call would be noise.

    Args:
        inner: The real (non-interactive) token source.
        notify: Called with the first consent error.
    """

    def __init__(self, inner: TokenSource, notify: ConsentNotifier) -> None:
        self._inner = inner
        self._notify = notify
        self._notified = False

    async def access_token(self) -> str:
        """The inner source's token; consent errors are reported, then re-raised."""
        try:
            return await self._inner.access_token()
        except ConsentRequiredError as error:
            if not self._notified:
                self._notified = True
                log.info("integrations.consent.notify", provider=error.provider)
                await self._notify(error)
            raise


# Design notes:
# - Non-interactive is an explicit mode of the token sources rather than a
#   check for "is there a console" inside them: the composition root knows
#   which front end it builds, the auth code should not guess (pythonw, a
#   redirected stdout and a CI runner all look different).
# - The remedy is the error message itself so every existing path that
#   turns an IntegrationError into words (tool failure notes, the agent)
#   speaks it without knowing about consent.
# - The connection log is a separate key, not a field inside the token
#   JSON, because google-auth and msal own those documents' shape.
