import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner, Result

from jarvis.config import Settings
from jarvis.integrations.base import AuthError
from jarvis.integrations.connect.cli import (
    EXIT_CHECK_FAILED,
    EXIT_INTERRUPTED,
    EXIT_SETUP_PROBLEM,
    ConnectDeps,
    connect_app,
)
from jarvis.integrations.connect.common import Echo
from jarvis.integrations.consent import ConnectionLog
from jarvis.integrations.tokens import MemoryTokenStore, TokenStore
from tests.integrations.conftest import Router

ACCESS = "fake-access-token-never-printed"
DISCORD_TOKEN = "fake-discord-bot-token-never-printed"
GUILD = "111111111111111111"

runner = CliRunner()


@dataclass
class FakeSignIn:
    store: TokenStore | None = None
    error: BaseException | None = None
    calls: list[dict[str, Any]] = field(default_factory=list)
    stored_before: list[bool] = field(default_factory=list)
    key: str = "google"

    def acquire_blocking(self) -> str:
        if self.store is not None:
            self.stored_before.append(self.store.load(self.key) is not None)
        if self.error is not None:
            raise self.error
        return ACCESS

    def signed_in_username(self) -> str | None:
        return "me@outlook.com"


@dataclass
class Harness:
    make_settings: Callable[..., Settings]
    router: Router
    tmp_path: Path
    overrides: dict[str, Any] = field(default_factory=dict)
    store: MemoryTokenStore = field(default_factory=MemoryTokenStore)
    google: FakeSignIn = field(default_factory=FakeSignIn)
    microsoft: FakeSignIn = field(default_factory=lambda: FakeSignIn(key="microsoft"))
    microsoft_teams: list[bool] = field(default_factory=list)

    def run(self, *args: str) -> Result:
        self.google.store = self.microsoft.store = self.store

        def microsoft_sign_in(
            _settings: Settings, _store: TokenStore, teams: bool, _echo: Echo
        ) -> FakeSignIn:
            self.microsoft_teams.append(teams)
            return self.microsoft

        deps = ConnectDeps(
            settings=lambda: self.make_settings(data_dir=self.tmp_path, **self.overrides),
            store=lambda _settings: self.store,
            http=self.router.client,
            google_sign_in=lambda _settings, _store: self.google,
            microsoft_sign_in=microsoft_sign_in,
        )
        return runner.invoke(connect_app, list(args), obj=deps)

    def google_client(self) -> None:
        client = {"installed": {"client_id": "123-fake.apps", "client_secret": "fake-secret-x"}}
        path = self.tmp_path / "google_client.json"
        path.write_text(json.dumps(client), encoding="utf-8")
        self.overrides["google_client_secrets_file"] = path


@pytest.fixture
def harness(make_settings: Callable[..., Settings], router: Router, tmp_path: Path) -> Harness:
    return Harness(make_settings, router, tmp_path)


def google_ok(router: Router) -> None:
    router.add("GET", "/gmail/v1/users/me/profile", {"emailAddress": "me@gmail.com"})
    router.add("GET", "/calendar/v3/users/me/calendarList", {"items": []})


def test_no_arguments_shows_the_commands(harness: Harness) -> None:
    result = harness.run()

    assert "google" in result.output and "disconnect" in result.output


def test_google_without_settings_is_a_setup_problem(harness: Harness) -> None:
    result = harness.run("google")

    assert result.exit_code == EXIT_SETUP_PROBLEM
    assert "GOOGLE_CLIENT_SECRETS_FILE is not set" in result.output
    assert harness.google.stored_before == []  # never got as far as signing in


def test_google_signs_in_verifies_and_hides_the_token(harness: Harness) -> None:
    harness.google_client()
    google_ok(harness.router)

    result = harness.run("google")

    assert result.exit_code == 0, result.output
    assert "Gmail: OK, signed in as me****@gmail.com" in result.output
    assert "Google is connected" in result.output
    assert ACCESS not in result.output
    assert "fake-secret-x" not in result.output
    assert "12****ps" in result.output


def test_google_force_forgets_the_stored_sign_in_first(harness: Harness) -> None:
    harness.google_client()
    google_ok(harness.router)
    harness.store.save("google", {"refresh_token": "old"})

    result = harness.run("google", "--force")

    assert result.exit_code == 0, result.output
    assert harness.google.stored_before == [False]


def test_google_check_failure_exits_1_and_is_recorded(harness: Harness) -> None:
    harness.google_client()
    harness.router.add("GET", "/gmail/v1/users/me/profile", {"error": "x"}, status=500)
    harness.router.add("GET", "/calendar/v3/users/me/calendarList", {"items": []})

    result = harness.run("google")

    assert result.exit_code == EXIT_CHECK_FAILED
    assert ConnectionLog(harness.store).last("google") is not None


def test_a_declined_consent_is_reported_without_a_traceback(harness: Harness) -> None:
    harness.google_client()
    harness.google.error = AuthError("google", "OAuth consent failed: access_denied")

    result = harness.run("google")

    assert result.exit_code == EXIT_SETUP_PROBLEM
    assert "Not connected: google: OAuth consent failed: access_denied" in result.output


def test_microsoft_teams_flag_reaches_the_sign_in(harness: Harness) -> None:
    harness.overrides["ms_client_id"] = "fake-client-id"
    harness.router.add("GET", "/v1.0/me/mailFolders/inbox", {"totalItemCount": 2})
    harness.router.add("GET", "/v1.0/me/calendars", {"value": []})
    harness.router.add("GET", "/v1.0/me/chats", {"value": []})

    teams = harness.run("microsoft", "--teams")
    plain = harness.run("microsoft")

    assert teams.exit_code == plain.exit_code == 0, teams.output + plain.output
    assert harness.microsoft_teams == [True, False]
    assert "Teams chats: OK" in teams.output
    assert "personal Microsoft accounts" in teams.output
    assert "Signed in as me****@outlook.com." in plain.output
    assert ACCESS not in teams.output + plain.output


def test_microsoft_without_a_client_id_is_a_setup_problem(harness: Harness) -> None:
    result = harness.run("microsoft")

    assert result.exit_code == EXIT_SETUP_PROBLEM
    assert "MS_CLIENT_ID is not set" in result.output
    assert harness.microsoft_teams == []


def test_ctrl_c_during_sign_in_cancels_cleanly(harness: Harness) -> None:
    harness.overrides["ms_client_id"] = "fake-client-id"
    harness.microsoft.error = KeyboardInterrupt()

    result = harness.run("microsoft")

    assert result.exit_code == EXIT_INTERRUPTED
    assert "Cancelled" in result.output


def discord_ok(router: Router) -> None:
    router.add("GET", "/api/v10/users/@me", {"id": "9", "username": "jarvis-bot"})
    router.add("GET", "/api/v10/applications/@me", {"flags": 1 << 19})
    router.add("GET", f"/api/v10/guilds/{GUILD}", {"name": "Home"})
    router.add("GET", f"/api/v10/guilds/{GUILD}/members/9", {"roles": []})
    router.add("GET", f"/api/v10/guilds/{GUILD}/roles", [{"id": GUILD, "permissions": "8"}])
    router.add(
        "GET", f"/api/v10/guilds/{GUILD}/channels", [{"id": "c", "name": "general", "type": 0}]
    )
    router.add("GET", "/api/v10/channels/c/messages", [])


def test_discord_success_is_remembered_for_status(harness: Harness) -> None:
    harness.overrides.update(discord_bot_token=DISCORD_TOKEN, discord_guild_id=GUILD)
    discord_ok(harness.router)

    result = harness.run("discord")
    status = harness.run("status")

    assert result.exit_code == 0, result.output
    assert "[OK  ] Bot token: valid, the bot is jarvis-bot." in result.output
    assert harness.store.load("discord-check") is not None
    assert "discord: configured, connected" in status.output
    assert DISCORD_TOKEN not in result.output + status.output


def test_discord_failure_exits_1_and_records_the_cause(harness: Harness) -> None:
    harness.overrides.update(discord_bot_token=DISCORD_TOKEN, discord_guild_id=GUILD)
    harness.store.save("discord-check", {"at": "earlier"})
    harness.router.add("GET", "/api/v10/users/@me", {"message": "401: Unauthorized"}, status=401)

    result = harness.run("discord")

    assert result.exit_code == EXIT_CHECK_FAILED
    assert "[FAIL] Bot token" in result.output
    assert harness.store.load("discord-check") is None
    last = ConnectionLog(harness.store).last("discord")
    assert last is not None and last[0].startswith("Bot token:")


def test_status_prints_every_service_without_secrets(harness: Harness) -> None:
    harness.store.save("google", {"token": ACCESS, "refresh_token": "fake-refresh-value"})

    result = harness.run("status")

    assert result.exit_code == 0
    assert result.output.splitlines()[0].startswith("google: not configured")
    assert ACCESS not in result.output and "fake-refresh-value" not in result.output


def test_disconnect_deletes_only_that_service(harness: Harness) -> None:
    harness.store.save("google", {"refresh_token": "x"})
    ConnectionLog(harness.store).record("google", "old")
    harness.store.save("microsoft", {"cache": "{}"})

    result = harness.run("disconnect", "GOOGLE")

    assert result.exit_code == 0, result.output
    assert harness.store.load("google") is None
    assert ConnectionLog(harness.store).last("google") is None
    assert harness.store.load("microsoft") is not None
    assert "myaccount.google.com/permissions" in result.output


def test_disconnect_rejects_unknown_services(harness: Harness) -> None:
    result = harness.run("disconnect", "spotify")

    assert result.exit_code != 0
    assert harness.store.data == {}
