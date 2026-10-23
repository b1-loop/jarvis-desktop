from collections.abc import Callable
from pathlib import Path

import httpx
import pytest

from jarvis.config import Settings
from jarvis.integrations.base import ConsentRequiredError
from jarvis.integrations.consent import NotifyOnConsent
from jarvis.integrations.discord import DiscordProvider
from jarvis.integrations.factory import IntegrationFactory
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
    factory = IntegrationFactory(make_settings(demo_integrations=True), http)

    assert type(factory.email()[0]) is StubEmailProvider
    assert type(factory.calendar()[0]) is StubCalendarProvider
    assert type(factory.music()[0]) is StubMusicProvider
    assert type(factory.chat()[0]) is StubChatProvider
    assert type(factory.smart_home()[0]) is StubSmartHomeProvider


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


def test_microsoft_settings_select_outlook_and_leave_teams_off_by_default(
    make_settings: Callable[..., Settings], http: httpx.AsyncClient
) -> None:
    settings = make_settings(ms_client_id="00000000-fake-client-id")
    factory = IntegrationFactory(settings, http, MemoryTokenStore())

    assert [type(p) for p in factory.email()] == [OutlookProvider]
    assert [type(p) for p in factory.calendar()] == [OutlookProvider]
    assert factory.chat() == []
    assert factory._microsoft() is factory._microsoft()
    assert set(factory._microsoft().scopes) == set(OutlookProvider.scopes)


def test_teams_enabled_adds_teams_chat_and_its_scopes_to_the_one_sign_in(
    make_settings: Callable[..., Settings], http: httpx.AsyncClient
) -> None:
    settings = make_settings(ms_client_id="00000000-fake-client-id", teams_enabled=True)
    factory = IntegrationFactory(settings, http, MemoryTokenStore())

    assert [type(p) for p in factory.chat()] == [TeamsProvider]
    assert set(factory._microsoft().scopes) == {
        *OutlookProvider.scopes,
        *TeamsProvider.scopes,
    }


def test_sign_in_is_non_interactive_unless_asked(
    make_settings: Callable[..., Settings], http: httpx.AsyncClient, tmp_path: Path
) -> None:
    settings = make_settings(
        google_client_secrets_file=tmp_path / "client.json", ms_client_id="fake-client"
    )
    quiet = IntegrationFactory(settings, http, MemoryTokenStore())
    terminal = IntegrationFactory(settings, http, MemoryTokenStore(), interactive_sign_in=True)

    assert not quiet._google().interactive
    assert not quiet._microsoft().interactive
    assert terminal._google().interactive
    assert terminal._microsoft().interactive


def test_a_consent_notifier_wraps_both_sign_ins(
    make_settings: Callable[..., Settings], http: httpx.AsyncClient, tmp_path: Path
) -> None:
    async def notify(_error: ConsentRequiredError) -> None:
        return None

    settings = make_settings(
        google_client_secrets_file=tmp_path / "client.json", ms_client_id="fake-client"
    )
    factory = IntegrationFactory(settings, http, MemoryTokenStore(), consent_notifier=notify)

    assert isinstance(factory._google(), NotifyOnConsent)
    assert isinstance(factory._microsoft(), NotifyOnConsent)


def test_both_mail_accounts_are_returned_together(
    make_settings: Callable[..., Settings], http: httpx.AsyncClient, tmp_path: Path
) -> None:
    settings = make_settings(
        google_client_secrets_file=tmp_path / "client.json", ms_client_id="fake-client"
    )

    providers = IntegrationFactory(settings, http, MemoryTokenStore()).email()

    assert [type(p) for p in providers] == [GmailProvider, OutlookProvider]


def test_spotify_requires_both_id_and_secret(
    make_settings: Callable[..., Settings], http: httpx.AsyncClient, tmp_path: Path
) -> None:
    half = make_settings(spotify_client_id="fake-id", data_dir=tmp_path)
    full = make_settings(
        spotify_client_id="fake-id", spotify_client_secret="fake-secret", data_dir=tmp_path
    )

    assert IntegrationFactory(half, http).music() == []
    assert type(IntegrationFactory(full, http).music()[0]) is SpotifyProvider
    assert (tmp_path / "tokens").is_dir()


def test_chat_lists_every_configured_network(
    make_settings: Callable[..., Settings], http: httpx.AsyncClient
) -> None:
    settings = make_settings(
        slack_bot_token="xoxb-fake",
        discord_bot_token="fake-discord",
        ms_client_id="fake",
        teams_enabled=True,
    )

    providers = IntegrationFactory(settings, http, MemoryTokenStore()).chat()

    assert [type(p) for p in providers] == [SlackProvider, DiscordProvider, TeamsProvider]


def test_home_assistant_requires_url_and_token(
    make_settings: Callable[..., Settings], http: httpx.AsyncClient
) -> None:
    url_only = make_settings(home_assistant_url="http://ha.local:8123")
    both = make_settings(home_assistant_url="http://ha.local:8123", home_assistant_token="fake")

    assert IntegrationFactory(url_only, http).smart_home() == []
    assert type(IntegrationFactory(both, http).smart_home()[0]) is HomeAssistantProvider


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
    factory = IntegrationFactory(make_settings(), http)

    assert factory.email() == []
    assert factory.calendar() == []
    assert factory.music() == []
    assert factory.chat() == []
    assert factory.smart_home() == []
