"""Connect Google (Gmail + Calendar) from a terminal.

Three steps, each with a precise message when it fails:

1. :func:`validate_client_secrets` checks ``GOOGLE_CLIENT_SECRETS_FILE``
   before anything opens: present, JSON, and a *Desktop app* OAuth client.
   The installed-app flow needs exactly that type (a loopback redirect);
   a Web client or a service-account key fails later with an obscure
   ``redirect_uri_mismatch`` in the browser, so it is caught here.
2. :func:`build_sign_in` returns the same
   :class:`~jarvis.integrations.google_auth.GoogleAuth` the providers use,
   interactive and with exactly the scopes they need; the CLI runs its
   consent in the foreground and the token lands in the same store.
3. :func:`verify_google` makes one cheap call per product (Gmail profile,
   calendar list) and explains the common failures: an API not enabled in
   the Cloud project, or a consent that granted fewer permissions.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import httpx

from jarvis.integrations.connect.common import (
    README_HINT,
    BlockingSignIn,
    ConnectError,
    Echo,
    mask_email,
    mask_identifier,
)
from jarvis.integrations.connect.probe import ProbeResult, probe
from jarvis.integrations.factory import google_scopes

if TYPE_CHECKING:
    from jarvis.config import Settings
    from jarvis.integrations.tokens import TokenStore

GMAIL_PROFILE_URL: Final = "https://gmail.googleapis.com/gmail/v1/users/me/profile"
CALENDAR_LIST_URL: Final = "https://www.googleapis.com/calendar/v3/users/me/calendarList"
_CALENDAR_PAGE: Final = "50"

SCOPE_PURPOSES: Final[dict[str, str]] = {
    "https://www.googleapis.com/auth/gmail.readonly": "read your mail",
    "https://www.googleapis.com/auth/gmail.compose": "create reply drafts and send them",
    "https://www.googleapis.com/auth/calendar.readonly": "read your calendars",
    "https://www.googleapis.com/auth/calendar.events": "create and change events",
}

_API_DISABLED_MARKERS: Final = ("SERVICE_DISABLED", "accessNotConfigured", "has not been used")
_SCOPE_MARKERS: Final = ("insufficient", "SCOPE_INSUFFICIENT")


@dataclass(frozen=True, slots=True)
class ClientSecrets:
    """The parts of a valid Desktop-app client file worth showing.

    Attributes:
        client_id: The OAuth client id (shown masked).
        project_id: The Google Cloud project, when the file names one.
    """

    client_id: str
    project_id: str | None


def validate_client_secrets(path: Path | None) -> ClientSecrets:
    """Check that ``path`` is a Desktop-app OAuth client JSON.

    Raises:
        ConnectError: Naming what is wrong and what to do instead.
    """
    if path is None:
        raise ConnectError(
            "GOOGLE_CLIENT_SECRETS_FILE is not set. Download the OAuth client JSON to "
            "data\\google_client.json and add GOOGLE_CLIENT_SECRETS_FILE=data\\google_client.json "
            f"to .env. {README_HINT}"
        )
    if not path.is_file():
        raise ConnectError(f"GOOGLE_CLIENT_SECRETS_FILE points to {path}, which does not exist.")
    return _parse_client_secrets(path, _read_json(path))


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ConnectError(f"{path} is not readable JSON ({type(error).__name__}).") from error
    if not isinstance(data, dict):
        raise ConnectError(f"{path} is not an OAuth client file. {README_HINT}")
    return data


def _parse_client_secrets(path: Path, data: dict[str, Any]) -> ClientSecrets:
    if "web" in data:
        raise ConnectError(
            f"{path} is a 'Web application' OAuth client. Jarvis signs in with a local "
            "redirect, which needs an OAuth client of type 'Desktop app': create one in the "
            f"Google Cloud Console and download its JSON instead. {README_HINT}"
        )
    if data.get("type") == "service_account":
        raise ConnectError(
            f"{path} is a service-account key, not an OAuth client. Create an OAuth client ID "
            f"of type 'Desktop app' and download that JSON. {README_HINT}"
        )
    installed = data.get("installed")
    if not isinstance(installed, dict):
        raise ConnectError(f"{path} is not an OAuth client file. {README_HINT}")
    missing = [key for key in ("client_id", "client_secret") if not installed.get(key)]
    if missing:
        raise ConnectError(f"{path} lacks {', '.join(missing)}; download the JSON again.")
    return ClientSecrets(str(installed["client_id"]), installed.get("project_id"))


def describe_plan(settings: Settings, secrets: ClientSecrets, echo: Echo) -> None:
    """Say which client is used and what each requested permission is for."""
    project = f", project {secrets.project_id}" if secrets.project_id else ""
    echo(f"Google OAuth client {mask_identifier(secrets.client_id)} (Desktop app{project})")
    echo(f"  from {settings.google_client_secrets_file}")
    echo("Jarvis asks for:")
    for scope in google_scopes():
        echo(f"  - {SCOPE_PURPOSES.get(scope, scope)}  ({scope.rsplit('/', 1)[-1]})")
    echo("Unless a valid sign-in is stored (use --force to redo it), your browser now opens")
    echo("Google's consent page. Choose your account and allow every")
    echo("permission; if Google warns that the app is unverified, that is your own app in")
    echo("testing: continue. If no browser opens, copy the URL printed below.")


def build_sign_in(settings: Settings, store: TokenStore) -> BlockingSignIn:
    """The real, interactive ``GoogleAuth`` with the providers' scopes."""
    from jarvis.integrations.google_auth import GoogleAuth  # noqa: PLC0415 - optional extra

    return GoogleAuth.from_settings(settings, google_scopes(), store, interactive=True)


async def verify_google(http: httpx.AsyncClient, token: str, echo: Echo) -> bool:
    """One Gmail and one Calendar call; prints what worked and why not."""
    authorization = f"Bearer {token}"
    gmail = await probe(http, GMAIL_PROFILE_URL, authorization)
    if gmail.ok:
        address = mask_email(str(gmail.field("emailAddress", "")))
        total = int(gmail.field("messagesTotal", 0))
        echo(f"Gmail: OK, signed in as {address} ({total:,} messages).")
    else:
        echo(f"Gmail: FAILED. {explain_google_failure('Gmail API', gmail)}")
    calendars = await probe(
        http, CALENDAR_LIST_URL, authorization, params={"maxResults": _CALENDAR_PAGE}
    )
    if calendars.ok:
        count = len(calendars.field("items", []))
        echo(f"Google Calendar: OK, {count} calendar(s) visible.")
    else:
        echo(f"Google Calendar: FAILED. {explain_google_failure('Google Calendar API', calendars)}")
    return gmail.ok and calendars.ok


def explain_google_failure(api: str, result: ProbeResult) -> str:
    """The likely cause and the fix for a failed verification call."""
    if result.status == 0:
        return f"Could not reach Google ({result.network_error}); check the connection."
    if any(marker in result.text for marker in _API_DISABLED_MARKERS):
        return (
            f"The {api} is not enabled in your Google Cloud project. Enable it under "
            "APIs & Services > Library, wait a minute, then run this again."
        )
    if any(marker in result.text for marker in _SCOPE_MARKERS):
        return (
            "Google granted fewer permissions than Jarvis asked for. Run again with --force "
            "and leave every permission ticked on the consent screen."
        )
    if result.status == httpx.codes.UNAUTHORIZED:
        return "Google rejected the new token; run again with --force."
    return f"{api} answered HTTP {result.status}."


# Design notes:
# - The client JSON is validated by shape, not by asking Google: a wrong
#   type is the most common setup mistake, it is visible in the file, and
#   catching it before the browser opens saves a confusing round trip.
# - Verification uses raw probes rather than the Gmail/Calendar providers
#   because the full error body is needed to tell "API disabled" from
#   "scope not granted"; the providers keep only a short preview by design.
# - The address is shown masked: enough to confirm the right account was
#   chosen in the browser, nothing more in the terminal scrollback.
