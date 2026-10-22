import httpx
import pytest
from pydantic import SecretStr

from jarvis.integrations.connect.discord import (
    ADMINISTRATOR,
    REQUIRED_PERMISSIONS,
    DiscordCheck,
    diagnose,
    explain_channels,
    explain_guild,
    explain_intent,
    explain_permissions,
    explain_token_failure,
    guild_permissions,
)
from jarvis.integrations.connect.probe import ProbeResult
from tests.integrations.conftest import Router

TOKEN = SecretStr("fake-bot-token-for-tests")
GUILD = "111111111111111111"
BOT = "222222222222222222"
ROLE = "333333333333333333"
ALL_NEEDED = sum(REQUIRED_PERMISSIONS.values())
MESSAGE_CONTENT = 1 << 19


def healthy(router: Router, *, flags: int = MESSAGE_CONTENT, permissions: int = ALL_NEEDED) -> None:
    router.add("GET", "/api/v10/users/@me", {"id": BOT, "username": "jarvis-bot", "bot": True})
    router.add("GET", "/api/v10/applications/@me", {"id": "app", "flags": flags})
    router.add("GET", f"/api/v10/guilds/{GUILD}", {"id": GUILD, "name": "Home"})
    router.add("GET", f"/api/v10/guilds/{GUILD}/members/{BOT}", {"roles": [ROLE]})
    router.add(
        "GET",
        f"/api/v10/guilds/{GUILD}/roles",
        [{"id": GUILD, "permissions": "0"}, {"id": ROLE, "permissions": str(permissions)}],
    )
    router.add(
        "GET",
        f"/api/v10/guilds/{GUILD}/channels",
        [
            {"id": "c1", "name": "general", "type": 0},
            {"id": "c2", "name": "Voice", "type": 2},
        ],
    )
    router.add("GET", "/api/v10/channels/c1/messages", [])


def titles(check: DiscordCheck) -> dict[str, bool]:
    return {finding.title: finding.ok for finding in check.findings}


async def test_a_healthy_setup_passes_every_check(router: Router) -> None:
    healthy(router)

    check = await diagnose(router.client(), TOKEN, GUILD)

    assert check.ok, check.findings
    assert (check.bot_name, check.guild_name) == ("jarvis-bot", "Home")
    assert titles(check) == {
        "Bot token": True,
        "Message Content intent": True,
        "Server": True,
        "Permissions": True,
        "Channels": True,
    }
    assert {r.headers["Authorization"] for r in router.requests} == {
        f"Bot {TOKEN.get_secret_value()}"
    }
    assert all(r.url.path != "/api/v10/channels/c2/messages" for r in router.requests)


async def test_a_wrong_token_stops_after_the_first_call(router: Router) -> None:
    router.add("GET", "/api/v10/users/@me", {"message": "401: Unauthorized", "code": 0}, status=401)

    check = await diagnose(router.client(), TOKEN, GUILD)

    assert not check.ok
    assert len(router.requests) == 1
    assert "Reset Token" in check.findings[0].detail
    assert TOKEN.get_secret_value() not in check.findings[0].detail


async def test_no_token_and_no_guild_are_reported(router: Router) -> None:
    none = await diagnose(router.client(), None, GUILD)
    healthy(router)
    no_guild = await diagnose(router.client(), TOKEN, None)

    assert "DISCORD_BOT_TOKEN is not set" in none.findings[0].detail
    assert "Copy Server ID" in next(f.detail for f in no_guild.findings if f.title == "Server")


async def test_intent_off_and_missing_permissions_are_found(router: Router) -> None:
    healthy(router, flags=0, permissions=REQUIRED_PERMISSIONS["View Channels"])

    check = await diagnose(router.client(), TOKEN, GUILD)

    assert titles(check)["Message Content intent"] is False
    permissions = next(f for f in check.findings if f.title == "Permissions")
    assert "Send Messages, Read Message History" in permissions.detail


async def test_a_bot_that_was_not_invited_is_explained(router: Router) -> None:
    router.add("GET", "/api/v10/users/@me", {"id": BOT, "username": "jarvis-bot"})
    router.add("GET", "/api/v10/applications/@me", {"flags": MESSAGE_CONTENT})
    router.add(
        "GET", f"/api/v10/guilds/{GUILD}", {"message": "Missing Access", "code": 50001}, status=403
    )

    check = await diagnose(router.client(), TOKEN, GUILD)

    server = next(f for f in check.findings if f.title == "Server")
    assert not server.ok
    assert "URL Generator" in server.detail
    assert "Permissions" not in titles(check)


@pytest.mark.parametrize(
    ("result", "expected"),
    [
        (ProbeResult(status=0, network_error="ConnectError"), "could not reach Discord"),
        (ProbeResult(status=429, retry_after=3.0), "run this again in 3 s"),
        (ProbeResult(status=401), "HTTP 401"),
        (ProbeResult(status=502), "unexpected HTTP 502"),
    ],
)
def test_token_failure_mapping(result: ProbeResult, expected: str) -> None:
    assert expected in explain_token_failure(result).detail


def test_guild_mapping_covers_unknown_guild_and_rate_limits() -> None:
    unknown = explain_guild(ProbeResult(status=404, body={"code": 10004}))
    limited = explain_guild(ProbeResult(status=429))

    assert "DISCORD_GUILD_ID is wrong" in unknown.detail
    assert "rate limiting" in limited.detail


def test_intent_accepts_the_limited_flag_and_reports_check_failures() -> None:
    limited = explain_intent(ProbeResult(status=200, body={"flags": 1 << 18}))
    failed = explain_intent(ProbeResult(status=500))

    assert limited.ok
    assert not failed.ok and "HTTP 500" in failed.detail


def test_permissions_from_roles() -> None:
    roles = [
        {"id": GUILD, "permissions": str(REQUIRED_PERMISSIONS["View Channels"])},
        {"id": ROLE, "permissions": str(REQUIRED_PERMISSIONS["Send Messages"])},
        {"id": "other", "permissions": str(ADMINISTRATOR)},
    ]

    granted = guild_permissions(GUILD, roles, [ROLE])

    assert granted == REQUIRED_PERMISSIONS["View Channels"] | REQUIRED_PERMISSIONS["Send Messages"]
    assert explain_permissions(ADMINISTRATOR).ok
    assert not explain_permissions(granted).ok


def test_channel_findings_name_blocked_and_readable_channels() -> None:
    finding = explain_channels(
        {
            "general": ProbeResult(status=200, body=[]),
            "secret": ProbeResult(status=403, body={"code": 50001}),
        }
    )

    assert not finding.ok
    assert "#secret" in finding.detail and "Readable: #general" in finding.detail
    assert not explain_channels({}).ok


async def test_network_failures_are_findings_not_exceptions() -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    client = httpx.AsyncClient(transport=httpx.MockTransport(refuse))

    check = await diagnose(client, TOKEN, GUILD)

    assert "could not reach Discord (ConnectError)" in check.findings[0].detail
