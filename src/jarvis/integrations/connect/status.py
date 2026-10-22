"""What is configured and connected, read from settings and the token store.

``connect_accounts.py status`` answers "will the desktop app be able to use
my accounts?" without any network call and without printing a secret: it
reads ``.env`` (through ``Settings``) for *configured*, the stored token
documents for *connected* and expiry, and the
:class:`~jarvis.integrations.consent.ConnectionLog` for the last problem.
Values are inspected, never echoed: an access token becomes "valid until
14:05", a refresh token "present", an account name is masked.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Final

from jarvis.integrations.connect.common import mask_email
from jarvis.integrations.consent import ConnectionLog
from jarvis.integrations.factory import google_scopes, microsoft_scopes

if TYPE_CHECKING:
    from jarvis.config import Settings
    from jarvis.integrations.tokens import TokenStore

GOOGLE_KEY: Final = "google"
MICROSOFT_KEY: Final = "microsoft"
DISCORD_CHECK_KEY: Final = "discord-check"
GRAPH_SCOPE_PREFIX: Final = "https://graph.microsoft.com/"


@dataclass(frozen=True, slots=True)
class ServiceStatus:
    """One service's state, safe to print.

    Attributes:
        service: ``google``, ``microsoft`` or ``discord``.
        configured: The settings it needs are present.
        connected: Usable without asking the user again.
        details: Extra lines (account, expiry, granted scopes), already masked.
        last_error: The latest recorded problem and when, if any.
    """

    service: str
    configured: bool
    connected: bool
    details: tuple[str, ...] = ()
    last_error: tuple[str, str] | None = None


def collect_status(settings: Settings, store: TokenStore) -> list[ServiceStatus]:
    """Status of every connectable service, in a fixed order."""
    errors = ConnectionLog(store)
    return [
        google_status(settings, store, errors),
        microsoft_status(settings, store, errors),
        discord_status(settings, store, errors),
    ]


def google_status(settings: Settings, store: TokenStore, errors: ConnectionLog) -> ServiceStatus:
    """Gmail + Calendar: secrets file, stored token, scopes and expiry."""
    path = settings.google_client_secrets_file
    out: list[str] = []
    if path is not None and not path.is_file():
        out.append(f"GOOGLE_CLIENT_SECRETS_FILE points to a missing file: {path}")
    info = store.load(GOOGLE_KEY) or {}
    missing = sorted(set(google_scopes()) - set(info.get("scopes", [])))
    connected = bool(info.get("refresh_token")) and not missing
    if info and missing:
        out.append("stored sign-in lacks: " + ", ".join(s.rsplit("/", 1)[-1] for s in missing))
    if info.get("refresh_token"):
        out.append("refresh token: present (renews access silently)")
    if expiry := info.get("expiry"):
        out.append(_expiry_line(_parse_iso(str(expiry))))
    return ServiceStatus(
        GOOGLE_KEY, path is not None, connected, tuple(out), errors.last(GOOGLE_KEY)
    )


def microsoft_status(settings: Settings, store: TokenStore, errors: ConnectionLog) -> ServiceStatus:
    """Outlook (+ Teams): client id, cached account, granted scopes and expiry."""
    out: list[str] = []
    cache = _msal_cache(store.load(MICROSOFT_KEY))
    for account in cache.get("Account", {}).values():
        out.append(f"account: {mask_email(str(account.get('username', '')))}")
    granted = _granted_graph_scopes(cache)
    wanted = {s.lower() for s in microsoft_scopes(teams=settings.teams_enabled)}
    has_refresh = bool(cache.get("RefreshToken"))
    if has_refresh:
        out.append("refresh token: present (renews access silently)")
    if granted and not wanted <= granted:
        out.append("stored sign-in lacks: " + ", ".join(sorted(wanted - granted)))
    if expiries := _access_expiries(cache):
        out.append(_expiry_line(max(expiries)))
    out.append(f"Teams: {'enabled' if settings.teams_enabled else 'off (JARVIS_TEAMS_ENABLED)'}")
    connected = has_refresh and (not granted or wanted <= granted)
    return ServiceStatus(
        MICROSOFT_KEY,
        settings.ms_client_id is not None,
        connected,
        tuple(out),
        errors.last(MICROSOFT_KEY),
    )


def discord_status(settings: Settings, store: TokenStore, errors: ConnectionLog) -> ServiceStatus:
    """Bot token and server id from settings; "connected" is the last check's result."""
    out: list[str] = []
    out.append(f"bot token: {'set (hidden)' if settings.discord_bot_token else 'not set'}")
    out.append(f"server id: {'set' if settings.discord_guild_id else 'not set'}")
    check = store.load(DISCORD_CHECK_KEY) or {}
    if check:
        out.append(
            f"last check {check.get('at', '?')}: bot {check.get('bot')} in {check.get('guild')}"
        )
    configured = settings.discord_bot_token is not None and settings.discord_guild_id is not None
    return ServiceStatus("discord", configured, bool(check), tuple(out), errors.last("discord"))


def render_status(statuses: list[ServiceStatus]) -> list[str]:
    """Lines for the terminal, one block per service."""
    lines: list[str] = []
    for status in statuses:
        configured = "configured" if status.configured else "not configured"
        connected = "connected" if status.connected else "not connected"
        lines.append(f"{status.service}: {configured}, {connected}")
        lines.extend(f"    {detail}" for detail in status.details)
        if status.last_error is not None:
            message, at = status.last_error
            lines.append(f"    last error ({at}): {message}")
    return lines


def _msal_cache(stored: dict[str, Any] | None) -> dict[str, Any]:
    if not stored:
        return {}
    try:
        cache = json.loads(str(stored.get("cache") or "{}"))
    except ValueError:
        return {}
    return cache if isinstance(cache, dict) else {}


def _granted_graph_scopes(cache: dict[str, Any]) -> set[str]:
    """Scopes in the cached access tokens, lower-cased and without the Graph prefix."""
    granted: set[str] = set()
    for entry in cache.get("AccessToken", {}).values():
        for scope in str(entry.get("target", "")).split():
            granted.add(scope.lower().removeprefix(GRAPH_SCOPE_PREFIX))
    return granted


def _access_expiries(cache: dict[str, Any]) -> list[datetime]:
    expiries: list[datetime] = []
    for entry in cache.get("AccessToken", {}).values():
        try:
            expiries.append(datetime.fromtimestamp(int(entry["expires_on"]), tz=UTC))
        except (KeyError, TypeError, ValueError):
            continue
    return expiries


def _parse_iso(value: str) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _expiry_line(expiry: datetime | None) -> str:
    if expiry is None:
        return "access token: expiry unknown"
    local = expiry.astimezone().strftime("%Y-%m-%d %H:%M")
    if expiry <= datetime.now(UTC):
        return f"access token: expired {local} (renewed on next use)"
    return f"access token: valid until {local}"


# Design notes:
# - Offline on purpose: status must be instant and must not trigger a
#   refresh (which would rotate tokens) just to be displayed. "connect"
#   is the command that proves an account works.
# - Microsoft scopes are compared case-insensitively and without the
#   https://graph.microsoft.com/ prefix, because msal records the target
#   in whatever form the token endpoint returned.
# - Access tokens last about an hour; the refresh token is what makes a
#   service "connected", so expiry is informational only.
