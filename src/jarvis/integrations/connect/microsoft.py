"""Connect Microsoft (Outlook mail + calendar, optionally Teams) from a terminal.

The device-code flow suits a terminal well: Jarvis shows a short code and a
URL, the user signs in in any browser, and msal's poll returns the tokens.
:class:`TerminalDeviceCodePrompt` makes that hard to miss: the code and URL
in a box, the code copied to the clipboard and the page opened in the
browser when possible.

Teams is opt-in (``--teams`` or ``JARVIS_TEAMS_ENABLED=true``) because the
Teams chat API (``Chat.Read``, ``ChatMessage.Send``) is only available to
work or school accounts; asking a personal account for those scopes gains
nothing and can fail the sign-in. Verification uses mail and calendar
calls, not ``/me``: reading the profile needs ``User.Read``, a scope Jarvis
does not otherwise use, so the account name comes from the sign-in itself.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import webbrowser
from collections.abc import Callable
from typing import TYPE_CHECKING, Final, Protocol

import httpx

from jarvis.integrations.connect.common import (
    README_HINT,
    ConnectError,
    Echo,
    mask_email,
    mask_identifier,
)
from jarvis.integrations.connect.probe import ProbeResult, probe
from jarvis.integrations.factory import microsoft_scopes

if TYPE_CHECKING:
    from jarvis.config import Settings
    from jarvis.integrations.microsoft_graph import DeviceCode
    from jarvis.integrations.tokens import TokenStore

PERSONAL_TENANTS: Final = frozenset({"common", "consumers"})
TEAMS_PERSONAL_NOTE: Final = (
    "Teams: personal Microsoft accounts (outlook.com, hotmail.com, live.com) cannot use "
    "the Teams chat API. Use --teams only with a work or school account."
)
_CLIPBOARD_TIMEOUT_SECONDS: Final = 5
_SECONDS_PER_MINUTE: Final = 60
_BOX_WIDTH: Final = 64

Copier = Callable[[str], bool]
Opener = Callable[[str], bool]


class MicrosoftSignIn(Protocol):
    """What the CLI needs from ``MicrosoftAuth``."""

    def acquire_blocking(self) -> str:
        """Sign in (silently or by device code) and return an access token."""

    def signed_in_username(self) -> str | None:
        """The account that signed in."""


def copy_to_clipboard(text: str) -> bool:
    """Put ``text`` on the Windows clipboard with ``clip.exe``; False if unavailable."""
    clip = shutil.which("clip") if sys.platform == "win32" else None
    if clip is None:
        return False
    try:
        # Fixed executable, no shell, and the input is the short ASCII user code.
        subprocess.run(  # noqa: S603
            [clip], input=text.encode("ascii"), check=True, timeout=_CLIPBOARD_TIMEOUT_SECONDS
        )
    except (OSError, subprocess.SubprocessError, UnicodeEncodeError):
        return False
    return True


def open_in_browser(url: str) -> bool:
    """Open ``url`` in the default browser; False when no browser could be started."""
    return webbrowser.open(url, new=2)


def render_device_code(code: DeviceCode, *, copied: bool, opened: bool) -> list[str]:
    """The boxed instructions; the code and URL are what the eye should land on."""
    rule = "=" * _BOX_WIDTH
    opened_note = "  (opened in your browser)" if opened else ""
    copied_note = "  (copied to the clipboard)" if copied else ""
    lines = [
        rule,
        "  Microsoft sign-in",
        f"  1. Go to:         {code.verification_uri}{opened_note}",
        f"  2. Enter code:    {code.user_code}{copied_note}",
        "  3. Sign in with the account Jarvis should use and accept the permissions.",
    ]
    if code.expires_in:
        minutes = max(1, code.expires_in // _SECONDS_PER_MINUTE)
        lines.append(f"  The code expires in {minutes} minutes. Waiting for you...")
    return [*lines, rule]


class TerminalDeviceCodePrompt:
    """Shows a device code in the terminal, copies it and opens the page.

    Args:
        echo: Where the instructions are printed.
        copy: Clipboard seam (tests pass a recorder).
        open_url: Browser seam (tests never open a real one).
    """

    def __init__(
        self, echo: Echo, *, copy: Copier = copy_to_clipboard, open_url: Opener = open_in_browser
    ) -> None:
        self._echo = echo
        self._copy = copy
        self._open = open_url

    def __call__(self, code: DeviceCode) -> None:
        """Render ``code``; clipboard and browser are best effort."""
        copied = self._copy(code.user_code)
        opened = self._open(code.verification_uri)
        for line in render_device_code(code, copied=copied, opened=opened):
            self._echo(line)


def wants_teams(settings: Settings, teams: bool | None) -> bool:
    """``--teams`` / ``--no-teams`` when given, else ``JARVIS_TEAMS_ENABLED``."""
    return settings.teams_enabled if teams is None else teams


def describe_plan(settings: Settings, *, teams: bool, echo: Echo) -> None:
    """Which app and tenant are used, what is asked for, and the Teams caveat.

    Raises:
        ConnectError: If ``MS_CLIENT_ID`` is not set.
    """
    if settings.ms_client_id is None:
        raise ConnectError(
            "MS_CLIENT_ID is not set. Register an app in Microsoft Entra and put its "
            f"Application (client) ID in .env as MS_CLIENT_ID. {README_HINT}"
        )
    tenant = settings.ms_tenant_id
    kind = "personal and work accounts" if tenant in PERSONAL_TENANTS else "your organisation"
    echo(f"Microsoft app {mask_identifier(settings.ms_client_id)}, tenant {tenant} ({kind})")
    echo("Jarvis asks for: " + ", ".join(microsoft_scopes(teams=teams)))
    if teams:
        echo(TEAMS_PERSONAL_NOTE)
    if teams != settings.teams_enabled:
        state = "true" if teams else "false"
        echo(f"Note: set JARVIS_TEAMS_ENABLED={state} in .env so the app asks for the same.")


def build_sign_in(
    settings: Settings, store: TokenStore, *, teams: bool, echo: Echo
) -> MicrosoftSignIn:
    """The real, interactive ``MicrosoftAuth`` with a terminal prompt."""
    from jarvis.integrations.microsoft_graph import MicrosoftAuth  # noqa: PLC0415 - optional extra

    return MicrosoftAuth.from_settings(
        settings,
        microsoft_scopes(teams=teams),
        store,
        prompt=TerminalDeviceCodePrompt(echo),
        interactive=True,
    )


async def verify_microsoft(
    http: httpx.AsyncClient, token: str, echo: Echo, *, username: str | None, teams: bool
) -> bool:
    """One mail, one calendar and (with Teams) one chat call; prints the outcome."""
    from jarvis.integrations.microsoft_graph import GRAPH_BASE_URL  # noqa: PLC0415 - needs msal

    authorization = f"Bearer {token}"
    if username:
        echo(f"Signed in as {mask_email(username)}.")
    inbox = await probe(
        http,
        f"{GRAPH_BASE_URL}/me/mailFolders/inbox",
        authorization,
        params={"$select": "totalItemCount,unreadItemCount"},
    )
    _report(echo, "Outlook mail", inbox, _inbox_summary)
    calendars = await probe(
        http, f"{GRAPH_BASE_URL}/me/calendars", authorization, params={"$select": "name"}
    )
    _report(echo, "Outlook calendar", calendars, _calendar_summary)
    results = [inbox, calendars]
    if teams:
        chats = await probe(http, f"{GRAPH_BASE_URL}/me/chats", authorization, params={"$top": 1})
        _report(echo, "Teams chats", chats, lambda _r: "chats are readable.")
        results.append(chats)
    return all(result.ok for result in results)


def _report(
    echo: Echo, label: str, result: ProbeResult, summary: Callable[[ProbeResult], str]
) -> None:
    if result.ok:
        echo(f"{label}: OK, {summary(result)}")
    else:
        echo(f"{label}: FAILED. {explain_graph_failure(label, result)}")


def _inbox_summary(result: ProbeResult) -> str:
    unread = int(result.field("unreadItemCount", 0))
    total = int(result.field("totalItemCount", 0))
    return f"inbox has {total:,} messages ({unread:,} unread)."


def _calendar_summary(result: ProbeResult) -> str:
    return f"{len(result.field('value', []))} calendar(s) visible."


def explain_graph_failure(label: str, result: ProbeResult) -> str:
    """The likely cause and the fix for a failed Graph verification call."""
    if result.status == 0:
        return f"Could not reach Microsoft Graph ({result.network_error}); check the connection."
    error = result.field("error", {})
    code = str(error.get("code", "")) if isinstance(error, dict) else ""
    if label.startswith("Teams"):
        return (
            f"Graph answered HTTP {result.status} {code}. {TEAMS_PERSONAL_NOTE} "
            "Otherwise check that Chat.Read and ChatMessage.Send are granted."
        )
    if code == "MailboxNotEnabledForRESTAPI":
        return "This account has no mailbox that Microsoft Graph can reach."
    if result.status == httpx.codes.UNAUTHORIZED:
        return "Microsoft rejected the new token; run again with --force."
    if result.status == httpx.codes.FORBIDDEN:
        return (
            "Access denied. Check that the app registration lists the delegated Graph "
            "permissions Mail.ReadWrite, Mail.Send and Calendars.ReadWrite; on a work "
            "tenant an administrator may have to grant consent."
        )
    return f"Graph answered HTTP {result.status} {code}".rstrip() + "."


# Design notes:
# - Clipboard through clip.exe rather than the native Win32 layer: this
#   command must work in a fresh checkout before the C++ build exists, and
#   clip.exe ships with every Windows install. Failure is silent because the
#   code is printed anyway.
# - The prompt runs on the thread that polls msal (the main thread here),
#   so printing directly is safe; nothing else writes to the terminal then.
# - The device_code secret never reaches this module: DeviceCode carries
#   only the user-facing fields.
