from collections.abc import Callable
from pathlib import Path

import httpx
import pytest

from jarvis.config import Settings
from jarvis.integrations.discord import DiscordProvider
from jarvis.integrations.factory import (
    IntegrationFactory,
    build_calendar_providers,
    build_chat_providers,
    build_email_providers,
    build_music_providers,
    build_smart_home_providers,
)
from jarvis.integrations.gmail import GmailProvider
from jarvis.integrations.google_calendar import GoogleCalendarProvider
from jarvis.integrations.home_assistant import HomeAssistantProvider
from jarvis.integrations.outlook import OutlookProvider
from jarvis.integrations.slack import SlackProvider
from jarvis.integrations.spotify import SpotifyProvider
from jarvis.integrations.stubs import (
    StubCalendarProvider,
    StubChatProvider,
    StubEmailProvider,
    StubMusicProvider,
    StubSmartHomeProvider,
)
from jarvis.integrations.teams import TeamsProvider
from jarvis.integrations.tokens import MemoryTokenStore


@pytest.fixture
def http() -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(lambda _r: httpx.Response(599)))


def test_demo_mode_without_credentials_gives_every_service_its_stub(
    make_settings: Callable[..., Settings], http: httpx.AsyncClient
) -> None:
    settings = make_settings(demo_integrations=True)

    assert type(build_email_providers(settings, http)[0]) is StubEmailProvider
    assert type(build_calendar_providers(settings, http)[0]) is StubCalendarProvider
    assert type(build_music_providers(settings, http)[0]) is StubMusicProvider
    assert type(build_chat_providers(settings, http)[0]) is StubChatProvider
    assert type(build_smart_home_providers(settings, http)[0]) is StubSmartHomeProvider


def test_google_settings_select_gmail_and_calendar_with_one_shared_sign_in(
    make_settings: Callable[..., Settings], http: httpx.AsyncClient, tmp_path: Path
) -> None:
    settings = make_settings(google_client_secrets_file=tmp_path / "client.json")
    factory = IntegrationFactory(settings, http, MemoryTokenStore())

    email = factory.email()
    calendar = factory.calendar()

    assert [type(p) for p in email] == [GmailProvider]
    assert [type(p) for p in calendar] == [GoogleCalendarProvider]
    assert factory._google() is factory._google()
    assert set(factory._google().scopes) == {
        *GmailProvider.scopes,
        *GoogleCalendarProvider.scopes,
    }


def test_microsoft_settings_select_outlook_for_mail_calendar_and_teams_for_chat(
    make_settings: Callable[..., Settings], http: httpx.AsyncClient
) -> None:
    settings = make_settings(ms_client_id="00000000-fake-client-id")
    factory = IntegrationFactory(settings, http, MemoryTokenStore())

    assert [type(p) for p in factory.email()] == [OutlookProvider]
    assert [type(p) for p in factory.calendar()] == [OutlookProvider]
    assert [type(p) for p in factory.chat()] == [TeamsProvider]
    assert factory._microsoft() is factory._microsoft()
    assert set(factory._microsoft().scopes) == {
        *OutlookProvider.scopes,
        *TeamsProvider.scopes,
    }


def test_both_mail_accounts_are_returned_together(
    make_settings: Callable[..., Settings], http: httpx.AsyncClient, tmp_path: Path
) -> None:
    settings = make_settings(
        google_client_secrets_file=tmp_path / "client.json", ms_client_id="fake-client"
    )

    providers = build_email_providers(settings, http, MemoryTokenStore())

    assert [type(p) for p in providers] == [GmailProvider, OutlookProvider]


def test_spotify_requires_both_id_and_secret(
    make_settings: Callable[..., Settings], http: httpx.AsyncClient, tmp_path: Path
) -> None:
    half = make_settings(spotify_client_id="fake-id", data_dir=tmp_path)
    full = make_settings(
        spotify_client_id="fake-id", spotify_client_secret="fake-secret", data_dir=tmp_path
    )

    assert build_music_providers(half, http) == []
    assert type(build_music_providers(full, http)[0]) is SpotifyProvider
    assert (tmp_path / "tokens").is_dir()


def test_chat_lists_every_configured_network(
    make_settings: Callable[..., Settings], http: httpx.AsyncClient
) -> None:
    settings = make_settings(
        slack_bot_token="xoxb-fake", discord_bot_token="fake-discord", ms_client_id="fake"
    )

    providers = build_chat_providers(settings, http, MemoryTokenStore())

    assert [type(p) for p in providers] == [SlackProvider, DiscordProvider, TeamsProvider]


def test_home_assistant_requires_url_and_token(
    make_settings: Callable[..., Settings], http: httpx.AsyncClient
) -> None:
    url_only = make_settings(home_assistant_url="http://ha.local:8123")
    both = make_settings(home_assistant_url="http://ha.local:8123", home_assistant_token="fake")

    assert build_smart_home_providers(url_only, http) == []
    assert type(build_smart_home_providers(both, http)[0]) is HomeAssistantProvider


def test_default_token_store_lives_under_data_dir(
    make_settings: Callable[..., Settings], http: httpx.AsyncClient, tmp_path: Path
) -> None:
    factory = IntegrationFactory(make_settings(data_dir=tmp_path), http)

    assert factory._store.path_for("google") == tmp_path / "tokens" / "google.json"


def test_without_credentials_no_service_pretends_to_be_connected(
    make_settings: Callable[..., Settings], http: httpx.AsyncClient
) -> None:
    # Regression: stub sample data used to be served by default, and the
    # proactive meeting trigger announced a fixture meeting as real.
    settings = make_settings()

    assert build_email_providers(settings, http) == []
    assert build_calendar_providers(settings, http) == []
    assert build_music_providers(settings, http) == []
    assert build_chat_providers(settings, http) == []
    assert build_smart_home_providers(settings, http) == []
