"""One authenticated GET whose outcome is data, not an exception.

The connect commands need the full response to explain a failure ("the
Gmail API is not enabled in your project", "admin consent required"),
whereas :class:`~jarvis.integrations.rest.RestSession` deliberately keeps
only a short preview and raises. :func:`probe` returns status, decoded body
and the network error, if any, and never raises for HTTP or transport
failures.
"""

from __future__ import annotations

from dataclasses import dataclass
from http import HTTPStatus
from typing import Any

import httpx


@dataclass(frozen=True, slots=True)
class ProbeResult:
    """Outcome of one probe.

    Attributes:
        status: HTTP status, ``0`` when no response arrived.
        body: Decoded JSON body, or ``None`` when it was not JSON.
        text: The raw body text (for matching error codes in messages).
        network_error: Why no response arrived, when ``status`` is 0.
        retry_after: Seconds the server asked to wait (429), when given.
    """

    status: int
    body: Any = None
    text: str = ""
    network_error: str | None = None
    retry_after: float | None = None

    @property
    def ok(self) -> bool:
        """True for a 2xx response."""
        return HTTPStatus.OK <= self.status < HTTPStatus.MULTIPLE_CHOICES

    def field(self, name: str, default: Any = None) -> Any:
        """``body[name]`` when the body is a JSON object, else ``default``."""
        return self.body.get(name, default) if isinstance(self.body, dict) else default


async def probe(
    http: httpx.AsyncClient,
    url: str,
    authorization: str,
    *,
    params: dict[str, Any] | None = None,
) -> ProbeResult:
    """GET ``url`` with the ``Authorization`` header and describe what came back."""
    try:
        response = await http.get(url, params=params, headers={"Authorization": authorization})
    except httpx.HTTPError as error:
        return ProbeResult(status=0, network_error=type(error).__name__)
    try:
        body: Any = response.json()
    except ValueError:
        body = None
    return ProbeResult(
        status=response.status_code,
        body=body,
        text=response.text,
        retry_after=_retry_after(response.headers.get("Retry-After"), body),
    )


def _retry_after(header: str | None, body: Any) -> float | None:
    value: Any = header
    if value is None and isinstance(body, dict):
        value = body.get("retry_after")  # Discord puts it in the body as well
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


# Design notes:
# - Only the exception's type name is kept for a network failure: httpx's
#   messages can include the full URL with query parameters, and the type
#   ("ConnectTimeout") is what the user needs to read anyway.
# - The Authorization header is passed in whole so Google/Microsoft
#   ("Bearer ...") and Discord ("Bot ...") share one helper.
