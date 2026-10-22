"""Chooses real or stub providers from settings.

The rule is simple and logged: a real provider is built when its settings
are present, the ``Stub*`` provider otherwise. Nothing else in Jarvis knows
which one it got. The factory also owns the two shared sign-ins (Google for
Gmail + Calendar, Microsoft for Outlook + Teams) so each is created once
with the union of scopes and the user consents once.

Real provider modules are imported only when selected, so a checkout
without the ``integrations`` extra still runs with stubs.
"""

# ruff: noqa: PLC0415  -- the lazy imports above are the point of this module.

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

import httpx

from jarvis.integrations.base import (
    CalendarProvider,
    ChatProvider,
    EmailProvider,
    MusicProvider,
    Provider,
    SmartHomeProvider,
)
from jarvis.integrations.stubs import (
    StubCalendarProvider,
    StubChatProvider,
    StubEmailProvider,
    StubMusicProvider,
    StubSmartHomeProvider,
)
from jarvis.integrations.tokens import FileTokenStore, TokenSource, TokenStore
from jarvis.utils.logging import get_logger

if TYPE_CHECKING:
    from jarvis.config import Settings

log = get_logger(__name__)

TOKENS_SUBDIR = "tokens"


class IntegrationFactory:
    """Builds providers for one ``Settings`` and one shared HTTP client.

    Args:
        settings: Source of credentials; ``None`` fields select stubs.
        http: Shared client injected into every REST provider.
        token_store: Persistence for OAuth state; defaults to JSON files under
            ``settings.data_dir / "tokens"``.
    """

    def __init__(
        self, settings: Settings, http: httpx.AsyncClient, token_store: TokenStore | None = None
    ) -> None:
        self._settings = settings
        self._http = http
        self._store = token_store or FileTokenStore(settings.data_dir / TOKENS_SUBDIR)
        self._google_auth: TokenSource | None = None
        self._microsoft_auth: TokenSource | None = None

    def email(self) -> list[EmailProvider]:
        """Gmail and/or Outlook, else the stub."""
        providers: list[EmailProvider] = []
        if self._settings.google_client_secrets_file is not None:
            from jarvis.integrations.gmail import GmailProvider

            providers.append(GmailProvider(self._http, self._google()))
        if self._settings.ms_client_id is not None:
            from jarvis.integrations.outlook import OutlookProvider

            providers.append(OutlookProvider(self._http, self._microsoft()))
        return _chosen("email", providers, StubEmailProvider)

    def calendar(self) -> list[CalendarProvider]:
        """Google Calendar and/or Outlook, else the stub."""
        providers: list[CalendarProvider] = []
        if self._settings.google_client_secrets_file is not None:
            from jarvis.integrations.google_calendar import GoogleCalendarProvider

            providers.append(GoogleCalendarProvider(self._http, self._google()))
        if self._settings.ms_client_id is not None:
            from jarvis.integrations.outlook import OutlookProvider

            providers.append(OutlookProvider(self._http, self._microsoft()))
        return _chosen("calendar", providers, StubCalendarProvider)

    def music(self) -> list[MusicProvider]:
        """Spotify when both client id and secret are set, else the stub."""
        providers: list[MusicProvider] = []
        settings = self._settings
        if settings.spotify_client_id is not None and settings.spotify_client_secret is not None:
            from jarvis.integrations.spotify import SpotifyAuth, SpotifyProvider

            cache_path = settings.data_dir / TOKENS_SUBDIR / "spotify.json"
            auth = SpotifyAuth.from_settings(settings, cache_path)
            providers.append(SpotifyProvider(self._http, auth))
        return _chosen("music", providers, StubMusicProvider)

    def chat(self) -> list[ChatProvider]:
        """Slack, Discord and/or Teams, else the stub."""
        providers: list[ChatProvider] = []
        if self._settings.slack_bot_token is not None:
            from jarvis.integrations.slack import SlackProvider

            providers.append(SlackProvider.from_settings(self._settings))
        if self._settings.discord_bot_token is not None:
            from jarvis.integrations.discord import DiscordProvider

            providers.append(DiscordProvider.from_settings(self._settings, self._http))
        if self._settings.ms_client_id is not None:
            from jarvis.integrations.teams import TeamsProvider

            providers.append(TeamsProvider(self._http, self._microsoft()))
        return _chosen("chat", providers, StubChatProvider)

    def smart_home(self) -> list[SmartHomeProvider]:
        """Home Assistant when URL and token are set, else the stub."""
        providers: list[SmartHomeProvider] = []
        settings = self._settings
        if settings.home_assistant_url is not None and settings.home_assistant_token is not None:
            from jarvis.integrations.home_assistant import HomeAssistantProvider

            providers.append(HomeAssistantProvider.from_settings(settings, self._http))
        return _chosen("smart_home", providers, StubSmartHomeProvider)

    def _google(self) -> TokenSource:
        """One Google sign-in covering Gmail and Calendar."""
        if self._google_auth is None:
            from jarvis.integrations.gmail import GMAIL_SCOPES
            from jarvis.integrations.google_auth import GoogleAuth
            from jarvis.integrations.google_calendar import CALENDAR_SCOPES

            scopes = (*GMAIL_SCOPES, *CALENDAR_SCOPES)
            self._google_auth = GoogleAuth.from_settings(self._settings, scopes, self._store)
        return self._google_auth

    def _microsoft(self) -> TokenSource:
        """One Microsoft sign-in covering Outlook and Teams."""
        if self._microsoft_auth is None:
            from jarvis.integrations.microsoft_graph import MicrosoftAuth
            from jarvis.integrations.outlook import OUTLOOK_SCOPES
            from jarvis.integrations.teams import TEAMS_SCOPES

            scopes = (*OUTLOOK_SCOPES, *TEAMS_SCOPES)
            self._microsoft_auth = MicrosoftAuth.from_settings(self._settings, scopes, self._store)
        return self._microsoft_auth


def _chosen[P: Provider](service: str, providers: Sequence[P], stub: Callable[[], P]) -> list[P]:
    chosen = list(providers) or [stub()]
    for provider in chosen:
        log.info(
            "integrations.provider.selected",
            service=service,
            provider=provider.provider_name,
            stub=not providers,
        )
    return chosen


def build_email_providers(
    settings: Settings, http: httpx.AsyncClient, token_store: TokenStore | None = None
) -> list[EmailProvider]:
    """One-shot convenience over :meth:`IntegrationFactory.email`."""
    return IntegrationFactory(settings, http, token_store).email()


def build_calendar_providers(
    settings: Settings, http: httpx.AsyncClient, token_store: TokenStore | None = None
) -> list[CalendarProvider]:
    """One-shot convenience over :meth:`IntegrationFactory.calendar`."""
    return IntegrationFactory(settings, http, token_store).calendar()


def build_music_providers(
    settings: Settings, http: httpx.AsyncClient, token_store: TokenStore | None = None
) -> list[MusicProvider]:
    """One-shot convenience over :meth:`IntegrationFactory.music`."""
    return IntegrationFactory(settings, http, token_store).music()


def build_chat_providers(
    settings: Settings, http: httpx.AsyncClient, token_store: TokenStore | None = None
) -> list[ChatProvider]:
    """One-shot convenience over :meth:`IntegrationFactory.chat`."""
    return IntegrationFactory(settings, http, token_store).chat()


def build_smart_home_providers(
    settings: Settings, http: httpx.AsyncClient, token_store: TokenStore | None = None
) -> list[SmartHomeProvider]:
    """One-shot convenience over :meth:`IntegrationFactory.smart_home`."""
    return IntegrationFactory(settings, http, token_store).smart_home()


# Design notes:
# - Lists, not single providers: a user with both Gmail and Outlook wants
#   "my unread mail" to cover both, and the tool layer can merge results.
# - The composition root should hold one IntegrationFactory so Gmail and
#   Calendar share a GoogleAuth; the build_* functions exist for callers that
#   need a single service and accept a possible second consent prompt.
# - Stub selection is logged at INFO with stub=True so a run without keys is
#   visibly a demo run, never a silent fake.
