"""What every connect command shares: output, masking, redaction, errors.

Everything the connect commands print goes through an :class:`Echo`. The
rule is simple: tokens and client secrets are never printed, identifiers
that help the user recognise an account (an email address, a client id)
are shown partly masked, and :class:`RedactingEcho` replaces any known
secret value that still slips into a line (an error body echoing a header,
say) as a second line of defence.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Final, Protocol

from jarvis.utils.errors import JarvisError

Echo = Callable[[str], None]
"""Writes one line for the user (``typer.echo`` in the CLI, a list in tests)."""

README_HINT: Final = 'See "Connect your accounts" in README.md.'


class ConnectError(JarvisError):
    """A setup problem the user can fix; the message says how."""


class BlockingSignIn(Protocol):
    """A token source that signs in on the calling thread (``GoogleAuth``, ``MicrosoftAuth``)."""

    def acquire_blocking(self) -> str:
        """Run or reuse the consent and return an access token."""


REDACTED: Final = "[hidden]"
_VISIBLE_EDGE: Final = 2
_MIN_MASKABLE: Final = 6
_MIN_SECRET_CHARS: Final = 8
"""Shorter values are not redacted: replacing "abc" everywhere would mangle output."""


def mask_identifier(value: str) -> str:
    """``"1234567890-abc.apps.googleusercontent.com"`` -> ``"12****om"``.

    Enough to tell two client ids apart, not enough to reuse one.
    """
    if len(value) < _MIN_MASKABLE:
        return "*" * len(value)
    return value[:_VISIBLE_EDGE] + "****" + value[-_VISIBLE_EDGE:]


def mask_email(address: str) -> str:
    """``"bojidar@gmail.com"`` -> ``"bo****@gmail.com"``; non-addresses are masked whole."""
    local, at, domain = address.partition("@")
    if not at:
        return mask_identifier(address)
    return local[:_VISIBLE_EDGE] + "****@" + domain


class RedactingEcho:
    """An :class:`Echo` that replaces known secret values before printing.

    Args:
        echo: Where the cleaned lines go.
        secrets: Values that must never appear (bot token, client secret).
    """

    def __init__(self, echo: Echo, secrets: Iterable[str | None] = ()) -> None:
        self._echo = echo
        self._secrets: list[str] = []
        for secret in secrets:
            self.add_secret(secret)

    def add_secret(self, secret: str | None) -> None:
        """Also hide ``secret`` from now on (an access token once it exists)."""
        if secret and len(secret) >= _MIN_SECRET_CHARS:
            self._secrets.append(secret)

    def __call__(self, line: str) -> None:
        """Print ``line`` with every secret replaced by ``[hidden]``."""
        for secret in self._secrets:
            line = line.replace(secret, REDACTED)
        self._echo(line)


# Design notes:
# - Masking keeps both edges of an id: users recognise "the one ending in
#   om" faster than a prefix alone, and four stars say "shortened" without
#   revealing the length.
# - Redaction is by value, not by pattern: the connect commands know the
#   exact secrets in play (they loaded them from settings), which catches
#   them in any format, whereas a token-looking regex would miss some and
#   mangle harmless hex.
