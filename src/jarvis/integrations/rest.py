"""Authenticated JSON calls with uniform error mapping.

Every REST-backed provider (Gmail, Google Calendar, Microsoft Graph, Spotify,
Discord, Home Assistant) needs the same four things: a bearer token on each
request, HTTP statuses turned into the ``IntegrationError`` hierarchy, a
record of the last failure for the dashboard, and no token in any log line.
:class:`RestSession` does exactly that once, so the providers themselves are
little more than URL shaping and response parsing.
"""

from __future__ import annotations

from http import HTTPStatus
from typing import Any

import httpx

from jarvis.integrations.base import (
    AuthError,
    IntegrationError,
    ProviderError,
    ProviderStatus,
    RateLimitedError,
)
from jarvis.integrations.tokens import TokenSource
from jarvis.utils.logging import get_logger

log = get_logger(__name__)

_BODY_PREVIEW_CHARS = 200
_AUTH_STATUSES = frozenset({HTTPStatus.UNAUTHORIZED, HTTPStatus.FORBIDDEN})


class RestSession:
    """One provider's view of an injected ``httpx.AsyncClient``.

    Args:
        http: Shared client; tests pass one built on ``httpx.MockTransport``.
        provider: Name used in errors and logs.
        base_url: Prefix for relative paths.
        token_source: Where the bearer token comes from.
        auth_scheme: ``Authorization`` scheme, ``Bearer`` for everyone but Discord.
    """

    def __init__(
        self,
        http: httpx.AsyncClient,
        *,
        provider: str,
        base_url: str,
        token_source: TokenSource,
        auth_scheme: str = "Bearer",
    ) -> None:
        self._http = http
        self._provider = provider
        self._base_url = base_url.rstrip("/")
        self._token_source = token_source
        self._auth_scheme = auth_scheme
        self.last_error: str | None = None
        self.authenticated = False

    def status(self) -> ProviderStatus:
        """Status for a configured real provider using this session."""
        return ProviderStatus(
            name=self._provider,
            configured=True,
            authenticated=self.authenticated,
            last_error=self.last_error,
        )

    async def request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json: Any = None,
        headers: dict[str, str] | None = None,
    ) -> Any:
        """Send one request and return the decoded JSON body (``None`` if empty).

        Raises:
            AuthError: On 401/403 or when no token can be obtained.
            RateLimitedError: On 429, with ``Retry-After`` when present.
            ProviderError: On any other non-2xx status or transport failure.
        """
        try:
            response = await self._send(method, path, params, json, headers)
            self._check(response)
        except IntegrationError as error:
            self.last_error = str(error)
            log.warning("integrations.request.failed", provider=self._provider, error=str(error))
            raise
        self.last_error = None
        self.authenticated = True
        return _decode(response)

    async def _send(
        self,
        method: str,
        path: str,
        params: dict[str, Any] | None,
        json: Any,
        headers: dict[str, str] | None,
    ) -> httpx.Response:
        token = await self._token_source.access_token()
        all_headers = {"Authorization": f"{self._auth_scheme} {token}", **(headers or {})}
        url = path if path.startswith("http") else f"{self._base_url}{path}"
        try:
            return await self._http.request(
                method, url, params=params, json=json, headers=all_headers
            )
        except httpx.HTTPError as error:
            raise ProviderError(self._provider, f"request failed: {error!r}") from error

    def _check(self, response: httpx.Response) -> None:
        status = response.status_code
        if response.is_success:
            return
        detail = f"HTTP {status} from {response.request.method} {response.request.url.path}"
        preview = response.text[:_BODY_PREVIEW_CHARS]
        if status in _AUTH_STATUSES:
            raise AuthError(self._provider, f"{detail}: {preview}")
        if status == HTTPStatus.TOO_MANY_REQUESTS:
            raise RateLimitedError(
                self._provider, detail, retry_after=_retry_after(response.headers)
            )
        raise ProviderError(self._provider, f"{detail}: {preview}", status_code=status)


def _retry_after(headers: httpx.Headers) -> float | None:
    value = headers.get("Retry-After")
    try:
        return float(value) if value is not None else None
    except ValueError:
        return None


def _decode(response: httpx.Response) -> Any:
    if response.status_code == HTTPStatus.NO_CONTENT or not response.content:
        return None
    return response.json()


# Design notes:
# - The token is fetched per request rather than cached here, because the
#   TokenSource already knows when it must refresh; caching twice invites
#   stale tokens after a refresh.
# - 403 maps to AuthError even though some services use it for "not found
#   for you": in every integrated API the practical remedy is the same, grant
#   the scope or re-consent, and that is what the agent should tell the user.
# - Only a 200-character body preview goes into the error: enough to see
#   "insufficient scope", not enough to echo a whole payload into the logs.
