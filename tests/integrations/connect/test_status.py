import json
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path

from jarvis.config import Settings
from jarvis.integrations.connect.status import collect_status, render_status
from jarvis.integrations.consent import ConnectionLog
from jarvis.integrations.factory import google_scopes
from jarvis.integrations.tokens import MemoryTokenStore

GOOGLE_ACCESS = "ya29.fake-google-access-token"
GOOGLE_REFRESH = "1//fake-google-refresh-token"
MS_ACCESS = "eyJ0eXAi.fake-ms-access-token"
MS_REFRESH = "M.R3_BAY.fake-ms-refresh-token"
DISCORD_TOKEN = "fake-discord-bot-token-value"


def msal_cache(expires: datetime, target: str) -> str:
    return json.dumps(
        {
            "Account": {"a": {"username": "bojidar@outlook.com"}},
            "AccessToken": {
                "t": {
                    "secret": MS_ACCESS,
                    "target": target,
                    "expires_on": str(int(expires.timestamp())),
                }
            },
            "RefreshToken": {"r": {"secret": MS_REFRESH}},
        }
    )


def connected_store() -> MemoryTokenStore:
    store = MemoryTokenStore()
    later = datetime.now(UTC) + timedelta(hours=1)
    store.save(
        "google",
        {
            "token": GOOGLE_ACCESS,
            "refresh_token": GOOGLE_REFRESH,
            "client_secret": "fake-client-secret",
            "scopes": list(google_scopes()),
            "expiry": later.replace(tzinfo=None).isoformat() + "Z",
        },
    )
    target = "https://graph.microsoft.com/Mail.ReadWrite Mail.Send calendars.readwrite openid"
    store.save("microsoft", {"cache": msal_cache(later, target)})
    store.save("discord-check", {"at": "2026-10-22T10:00:00+00:00", "bot": "jb", "guild": "Home"})
    return store


def settings_for(make_settings: Callable[..., Settings], tmp_path: Path) -> Settings:
    client = tmp_path / "google_client.json"
    client.write_text("{}", encoding="utf-8")
    return make_settings(
        google_client_secrets_file=client,
        ms_client_id="fake-client-id",
        discord_bot_token=DISCORD_TOKEN,
        discord_guild_id="111",
    )


def test_connected_services_show_state_without_any_secret(
    make_settings: Callable[..., Settings], tmp_path: Path
) -> None:
    statuses = collect_status(settings_for(make_settings, tmp_path), connected_store())
    text = "\n".join(render_status(statuses))

    assert [(s.service, s.configured, s.connected) for s in statuses] == [
        ("google", True, True),
        ("microsoft", True, True),
        ("discord", True, True),
    ]
    assert "access token: valid until" in text
    assert "account: bo****@outlook.com" in text
    assert "bot token: set (hidden)" in text
    for secret in (GOOGLE_ACCESS, GOOGLE_REFRESH, MS_ACCESS, MS_REFRESH, DISCORD_TOKEN):
        assert secret not in text
    assert "fake-client-secret" not in text
    assert "bojidar@outlook.com" not in text


def test_nothing_configured_reads_plainly(make_settings: Callable[..., Settings]) -> None:
    lines = render_status(collect_status(make_settings(), MemoryTokenStore()))

    assert lines[0] == "google: not configured, not connected"
    assert "microsoft: not configured, not connected" in lines
    assert "discord: not configured, not connected" in lines


def test_problems_are_listed(make_settings: Callable[..., Settings], tmp_path: Path) -> None:
    store = MemoryTokenStore()
    store.save("google", {"refresh_token": GOOGLE_REFRESH, "scopes": [google_scopes()[0]]})
    earlier = datetime.now(UTC) - timedelta(hours=2)
    store.save("microsoft", {"cache": msal_cache(earlier, "Mail.ReadWrite")})
    ConnectionLog(store).record("google", "Google isn't connected yet; run ...")
    settings = make_settings(
        google_client_secrets_file=tmp_path / "missing.json", ms_client_id="fake"
    )

    statuses = collect_status(settings, store)
    text = "\n".join(render_status(statuses))

    assert not statuses[0].connected
    assert not statuses[1].connected
    assert "points to a missing file" in text
    assert "stored sign-in lacks: calendar.events, calendar.readonly, gmail.compose" in text
    assert "stored sign-in lacks: calendars.readwrite, mail.send" in text
    assert "access token: expired" in text
    assert "last error (" in text and "Google isn't connected yet" in text


def test_a_corrupt_msal_cache_counts_as_not_connected(
    make_settings: Callable[..., Settings],
) -> None:
    store = MemoryTokenStore()
    store.save("microsoft", {"cache": "{not json"})

    microsoft = collect_status(make_settings(ms_client_id="fake"), store)[1]

    assert microsoft.configured and not microsoft.connected
