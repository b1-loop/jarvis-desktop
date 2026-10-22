"""Where OAuth tokens live and how providers obtain an access token.

Two small abstractions keep every provider free of credential plumbing:

* :class:`TokenStore` persists whatever blob an auth library wants to keep
  between runs (Google's authorized-user JSON, msal's serialised cache). The
  file implementation writes under ``Settings.data_dir`` so tokens never end
  up in the repository or next to the code.
* :class:`TokenSource` yields a bearer token on demand. REST providers only
  depend on this, so tests hand them a :class:`StaticTokenSource` with an
  obviously fake value and never touch an OAuth flow.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from pydantic import SecretStr

from jarvis.utils.logging import get_logger

log = get_logger(__name__)

_TOKEN_FILE_MODE = 0o600


class TokenStore(Protocol):
    """Persistence for auth-library state, keyed by provider.

    Methods are synchronous because they are only called from the worker
    thread that runs the (blocking) auth library; the event loop never
    touches the disk.
    """

    def load(self, key: str) -> dict[str, Any] | None:
        """Return the stored JSON object for ``key``, or ``None``."""

    def save(self, key: str, data: dict[str, Any]) -> None:
        """Persist ``data`` for ``key``, replacing any previous value."""

    def clear(self, key: str) -> None:
        """Forget ``key`` (after a revoked token, for example)."""


class FileTokenStore:
    """JSON files under ``<directory>/<key>.json``, owner-readable only."""

    def __init__(self, directory: Path) -> None:
        self._directory = directory

    def path_for(self, key: str) -> Path:
        """Where ``key`` is stored; auth libraries with their own cache use it."""
        return self._directory / f"{key}.json"

    def load(self, key: str) -> dict[str, Any] | None:
        """Read the stored object; a missing or corrupt file counts as absent."""
        path = self.path_for(key)
        if not path.exists():
            return None
        try:
            data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            log.warning("integrations.token_store.unreadable", key=key, error=repr(error))
            return None
        return data

    def save(self, key: str, data: dict[str, Any]) -> None:
        """Write atomically and restrict permissions (a no-op on NTFS)."""
        path = self.path_for(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_name(path.name + ".part")
        partial.write_text(json.dumps(data), encoding="utf-8")
        partial.chmod(_TOKEN_FILE_MODE)
        partial.replace(path)

    def clear(self, key: str) -> None:
        """Delete the file if present."""
        self.path_for(key).unlink(missing_ok=True)


class MemoryTokenStore:
    """Dictionary-backed store for tests and for runs that must not persist."""

    def __init__(self) -> None:
        self.data: dict[str, dict[str, Any]] = {}

    def load(self, key: str) -> dict[str, Any] | None:
        """Return a copy so callers cannot mutate the store by accident."""
        stored = self.data.get(key)
        return dict(stored) if stored is not None else None

    def save(self, key: str, data: dict[str, Any]) -> None:
        """Store a copy of ``data``."""
        self.data[key] = dict(data)

    def clear(self, key: str) -> None:
        """Forget ``key``."""
        self.data.pop(key, None)


@runtime_checkable
class TokenSource(Protocol):
    """Anything that can produce a bearer token for the next request."""

    async def access_token(self) -> str:
        """A currently valid access token.

        Raises:
            AuthError: If no token can be obtained.
        """


class StaticTokenSource:
    """A long-lived token from settings (Home Assistant, Slack, Discord)."""

    def __init__(self, token: SecretStr) -> None:
        self._token = token

    async def access_token(self) -> str:
        """The configured token, unchanged."""
        return self._token.get_secret_value()


class ThreadedTokenSource:
    """Base for sources whose acquisition runs a blocking OAuth library.

    Subclasses implement :meth:`_acquire_blocking`; it runs in a worker
    thread under a lock so two concurrent requests never start two browser
    or device-code flows.
    """

    def __init__(self) -> None:
        self._lock = asyncio.Lock()

    async def access_token(self) -> str:
        """Serialise acquisition and keep the event loop free."""
        async with self._lock:
            return await asyncio.to_thread(self._acquire_blocking)

    def _acquire_blocking(self) -> str:
        raise NotImplementedError


# Design notes:
# - Tokens are not SecretStr inside the store: the auth libraries hand us plain
#   dicts and expect them back. The store never logs values, and the logging
#   redactor catches any key containing "token" as a second line of defence.
# - chmod 0o600 is best effort. NTFS ignores POSIX bits; the real protection
#   on Windows is that data_dir lives in the user's profile.
# - ThreadedTokenSource holds an asyncio.Lock across the thread hop so a
#   burst of tool calls on a fresh install produces one consent prompt.
