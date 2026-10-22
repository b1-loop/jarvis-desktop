"""Check a Discord bot setup and say exactly what is wrong.

Discord needs no OAuth consent here: the provider authenticates with the
bot token from ``.env``. What goes wrong instead is setup, and Discord's
answers are terse (``401``, ``50001 Missing Access``). :func:`diagnose`
walks the same path the provider does and turns each answer into a
:class:`Finding` with the fix:

1. ``GET /users/@me``: is the token a bot token at all?
2. ``GET /applications/@me``: is the Message Content intent on? Without it
   Discord returns messages with empty text, over REST as well.
3. ``GET /guilds/{id}``: is the bot a member of ``DISCORD_GUILD_ID``?
4. The bot's roles: View Channels, Read Message History, Send Messages.
5. ``GET /guilds/{id}/channels`` and one message read per text channel.

The token is only ever placed in the ``Authorization`` header.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final

import httpx
from pydantic import SecretStr

from jarvis.integrations.connect.probe import ProbeResult, probe
from jarvis.integrations.discord import BASE_URL

MESSAGE_CONTENT_FLAGS: Final = (1 << 18) | (1 << 19)
"""GATEWAY_MESSAGE_CONTENT_LIMITED | GATEWAY_MESSAGE_CONTENT application flags."""

ADMINISTRATOR: Final = 1 << 3
REQUIRED_PERMISSIONS: Final[dict[str, int]] = {
    "View Channels": 1 << 10,
    "Send Messages": 1 << 11,
    "Read Message History": 1 << 16,
}
UNKNOWN_GUILD: Final = 10004
MISSING_ACCESS: Final = 50001
_NOT_A_MEMBER_STATUSES: Final = frozenset({httpx.codes.FORBIDDEN, httpx.codes.NOT_FOUND})
GUILD_TEXT_CHANNEL: Final = 0
MAX_CHANNELS_CHECKED: Final = 10
"""Enough to show the pattern without tripping the rate limit on big servers."""

REISSUE_HELP: Final = (
    "Copy the token again from the Developer Portal: your application > Bot > Reset Token. "
    "It is not the Application ID, the Public Key or the OAuth2 Client Secret."
)
INVITE_HELP: Final = (
    "Invite the bot: Developer Portal > OAuth2 > URL Generator, scope 'bot', permissions "
    "View Channels, Read Message History and Send Messages; open the URL and pick the server."
)
GUILD_ID_HELP: Final = (
    "In Discord enable Developer Mode (User Settings > Advanced), right-click the server "
    "icon > Copy Server ID, and set DISCORD_GUILD_ID to that number."
)


@dataclass(frozen=True, slots=True)
class Finding:
    """One checked item.

    Attributes:
        ok: Whether this part of the setup works.
        title: What was checked, e.g. "Bot token".
        detail: What was found, or the fix when ``ok`` is False.
    """

    ok: bool
    title: str
    detail: str


@dataclass(frozen=True, slots=True)
class DiscordCheck:
    """All findings of one diagnosis plus the names worth remembering."""

    findings: tuple[Finding, ...]
    bot_name: str | None = None
    guild_name: str | None = None

    @property
    def ok(self) -> bool:
        """True when nothing failed."""
        return all(finding.ok for finding in self.findings)


class _Api:
    """GETs against the Discord API with the bot token."""

    def __init__(self, http: httpx.AsyncClient, token: SecretStr) -> None:
        self._http = http
        self._authorization = f"Bot {token.get_secret_value()}"

    async def get(self, path: str, params: dict[str, Any] | None = None) -> ProbeResult:
        return await probe(self._http, f"{BASE_URL}{path}", self._authorization, params=params)


async def diagnose(
    http: httpx.AsyncClient, token: SecretStr | None, guild_id: str | None
) -> DiscordCheck:
    """Run every check that the previous ones make meaningful."""
    if token is None:
        return DiscordCheck((Finding(False, "Bot token", "DISCORD_BOT_TOKEN is not set in .env."),))
    api = _Api(http, token)
    me = await api.get("/users/@me")
    if not me.ok:
        return DiscordCheck((explain_token_failure(me),))
    bot_name = str(me.field("username", "bot"))
    findings = [Finding(True, "Bot token", f"valid, the bot is {bot_name}."), await _intent(api)]
    guild_name: str | None = None
    if guild_id is None or not guild_id.strip().isdigit():
        detail = f"DISCORD_GUILD_ID is not set to a server id. {GUILD_ID_HELP}"
        findings.append(Finding(False, "Server", detail))
    else:
        guild = await api.get(f"/guilds/{guild_id}")
        findings.append(explain_guild(guild))
        if guild.ok:
            guild_name = str(guild.field("name", guild_id))
            findings.extend(await _guild_checks(api, guild_id, str(me.field("id", ""))))
    return DiscordCheck(tuple(findings), bot_name=bot_name, guild_name=guild_name)


def explain_token_failure(result: ProbeResult) -> Finding:
    """``/users/@me`` failed: the token, the network or the rate limit."""
    if result.status == 0:
        return Finding(False, "Bot token", f"could not reach Discord ({result.network_error}).")
    if result.status == httpx.codes.TOO_MANY_REQUESTS:
        return Finding(False, "Bot token", _rate_limited(result))
    if result.status == httpx.codes.UNAUTHORIZED:
        return Finding(False, "Bot token", f"Discord rejected the token (HTTP 401). {REISSUE_HELP}")
    return Finding(False, "Bot token", f"unexpected HTTP {result.status} from Discord.")


def explain_intent(result: ProbeResult) -> Finding:
    """Read the Message Content intent from the application flags."""
    if not result.ok:
        return Finding(False, "Message Content intent", f"could not check (HTTP {result.status}).")
    if int(result.field("flags", 0)) & MESSAGE_CONTENT_FLAGS:
        return Finding(True, "Message Content intent", "enabled.")
    return Finding(
        False,
        "Message Content intent",
        "off, so Discord returns messages with empty text. Turn on Developer Portal > "
        "Bot > Privileged Gateway Intents > Message Content Intent and save.",
    )


def explain_guild(result: ProbeResult) -> Finding:
    """``/guilds/{id}`` answered: member, not invited, or a wrong id."""
    if result.ok:
        return Finding(True, "Server", f"the bot is a member of {result.field('name', 'it')}.")
    code = result.field("code")
    if result.status == httpx.codes.TOO_MANY_REQUESTS:
        return Finding(False, "Server", _rate_limited(result))
    if code in {UNKNOWN_GUILD, MISSING_ACCESS} or result.status in _NOT_A_MEMBER_STATUSES:
        return Finding(
            False,
            "Server",
            f"the bot is not in that server, or DISCORD_GUILD_ID is wrong. {INVITE_HELP} "
            f"{GUILD_ID_HELP}",
        )
    return Finding(False, "Server", f"unexpected HTTP {result.status} from Discord.")


def explain_permissions(granted: int) -> Finding:
    """Server-wide permissions of the bot's roles against what Jarvis needs."""
    if granted & ADMINISTRATOR:
        return Finding(True, "Permissions", "Administrator (covers everything).")
    missing = [name for name, bit in REQUIRED_PERMISSIONS.items() if not granted & bit]
    if not missing:
        return Finding(True, "Permissions", ", ".join(REQUIRED_PERMISSIONS) + ".")
    return Finding(
        False,
        "Permissions",
        f"the bot's roles lack {', '.join(missing)}. Grant them in Server Settings > Roles "
        "(the bot's role), or re-invite it with those permissions ticked.",
    )


def explain_channels(names: dict[str, ProbeResult]) -> Finding:
    """Which text channels the bot could read one message from."""
    if not names:
        return Finding(False, "Channels", "the server has no text channels the bot can see.")
    readable = [name for name, result in names.items() if result.ok]
    blocked = [name for name, result in names.items() if not result.ok]
    if not blocked:
        return Finding(True, "Channels", "readable: " + ", ".join(f"#{n}" for n in readable))
    return Finding(
        False,
        "Channels",
        "cannot read " + ", ".join(f"#{n}" for n in blocked) + ". A channel's own permission "
        "settings can hide it from the bot: allow View Channel and Read Message History "
        "for the bot's role there."
        + (f" Readable: {', '.join(f'#{n}' for n in readable)}." if readable else ""),
    )


def guild_permissions(guild_id: str, roles: list[dict[str, Any]], member_roles: list[str]) -> int:
    """OR of ``@everyone`` (whose id is the guild id) and the member's roles."""
    wanted = {guild_id, *member_roles}
    granted = 0
    for role in roles:
        if str(role.get("id")) in wanted:
            granted |= int(role.get("permissions", 0))
    return granted


async def _intent(api: _Api) -> Finding:
    return explain_intent(await api.get("/applications/@me"))


async def _guild_checks(api: _Api, guild_id: str, bot_id: str) -> list[Finding]:
    findings: list[Finding] = []
    member = await api.get(f"/guilds/{guild_id}/members/{bot_id}")
    roles = await api.get(f"/guilds/{guild_id}/roles")
    if member.ok and roles.ok and isinstance(roles.body, list):
        granted = guild_permissions(guild_id, roles.body, list(member.field("roles", [])))
        findings.append(explain_permissions(granted))
    channels = await api.get(f"/guilds/{guild_id}/channels")
    if not channels.ok or not isinstance(channels.body, list):
        detail = f"could not list channels (HTTP {channels.status})."
        return [*findings, Finding(False, "Channels", detail)]
    text = [c for c in channels.body if c.get("type") == GUILD_TEXT_CHANNEL][:MAX_CHANNELS_CHECKED]
    reads = {
        str(c.get("name", c["id"])): await api.get(f"/channels/{c['id']}/messages", {"limit": 1})
        for c in text
    }
    return [*findings, explain_channels(reads)]


def _rate_limited(result: ProbeResult) -> str:
    wait = f" in {result.retry_after:.0f} s" if result.retry_after else " in a minute"
    return f"Discord is rate limiting these checks; run this again{wait}."


# Design notes:
# - Permissions are computed from the server-wide roles (everyone + the
#   bot's roles), not per channel: channel overwrites are many and
#   order-dependent, and the per-channel read check below already shows
#   their effect as "cannot read #x".
# - Each check runs only when the previous one makes it meaningful: with a
#   bad token every later call would also say 401 and bury the real cause.
# - The application-flags check is the reliable intent test; inferring it
#   from empty message text would misfire on image-only messages.
