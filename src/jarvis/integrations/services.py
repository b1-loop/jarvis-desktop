"""The bundle of external services that integration tools receive.

Tools reach email, calendar, music, chat, smart home, web search, web pages
and YouTube only through :class:`IntegrationServices` on ``ToolContext``. The
bundle holds interfaces (the ABCs in ``base.py`` and the Protocols in
``web.py`` / ``youtube.py``), never a concrete provider type, so a tool is
tested against ``Stub*`` providers and the composition root alone decides
what is wired in. :func:`build_integration_services` is that decision in one
call: it delegates to :class:`~jarvis.integrations.factory.IntegrationFactory`
for the account-backed services and picks the search backend from settings.

The bundle also owns :class:`EmailDraftBook`, the per-session memory of the
drafts Jarvis created, because ``draft_email_reply`` and ``send_email`` are
separate plugins that may not import each other yet must agree on what a
draft id refers to.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Final

import httpx

from jarvis.integrations.base import (
    CalendarProvider,
    ChatProvider,
    EmailDraft,
    EmailProvider,
    MusicProvider,
    SmartHomeProvider,
)
from jarvis.integrations.factory import IntegrationFactory
from jarvis.integrations.web import (
    BraveSearchProvider,
    DuckDuckGoSearchProvider,
    WebSearchProvider,
)
from jarvis.integrations.youtube import TranscriptFetcher, YouTubeTranscriptApiFetcher
from jarvis.utils.errors import JarvisError
from jarvis.utils.logging import get_logger

if TYPE_CHECKING:
    from jarvis.config import Settings
    from jarvis.integrations.consent import ConsentNotifier
    from jarvis.integrations.tokens import TokenStore

log = get_logger(__name__)

REF_SEPARATOR: Final = ":"
"""Joins a provider name and a provider-local id into one reference string."""


class UnknownReferenceError(JarvisError):
    """A reference names no known provider, or an item this session never saw.

    Attributes:
        reference: The reference as the caller supplied it.
    """

    def __init__(self, reference: str, reason: str) -> None:
        self.reference = reference
        super().__init__(f"unknown reference {reference!r}: {reason}")


def make_reference(provider_name: str, item_id: str) -> str:
    """``("gmail", "18c2")`` -> ``"gmail:18c2"``.

    Ids are only unique within one provider, and a user with Gmail *and*
    Outlook gets merged listings, so every id handed to the LLM carries the
    provider that issued it.
    """
    return f"{provider_name}{REF_SEPARATOR}{item_id}"


def split_reference(reference: str) -> tuple[str, str]:
    """Inverse of :func:`make_reference`; splits on the first separator only.

    Raises:
        UnknownReferenceError: If there is no provider prefix or no id.
    """
    provider_name, separator, item_id = reference.partition(REF_SEPARATOR)
    if not separator or not provider_name or not item_id:
        raise UnknownReferenceError(reference, "expected '<provider>:<id>'")
    return provider_name, item_id


@dataclass(frozen=True, slots=True)
class StoredDraft:
    """A draft this session created, with the provider that holds it."""

    provider: EmailProvider
    draft: EmailDraft


class EmailDraftBook:
    """Drafts created in this session, keyed by their reference string.

    The email interface can create and send a draft but not look one up, so
    the draft object returned by ``draft_reply`` is kept here until it is
    sent. Drafts are not persisted: after a restart the user dictates the
    reply again, which is cheaper than a draft cache that can go stale.
    """

    def __init__(self) -> None:
        self._drafts: dict[str, StoredDraft] = {}

    def remember(self, provider: EmailProvider, draft: EmailDraft) -> str:
        """Store ``draft`` and return the reference the LLM will use for it."""
        reference = make_reference(provider.provider_name, draft.id)
        self._drafts[reference] = StoredDraft(provider, draft)
        return reference

    def get(self, reference: str) -> StoredDraft:
        """The stored draft for ``reference``.

        Raises:
            UnknownReferenceError: If this session created no such draft.
        """
        try:
            return self._drafts[reference]
        except KeyError:
            reason = "no draft with that id in this session"
            raise UnknownReferenceError(reference, reason) from None

    def forget(self, reference: str) -> None:
        """Drop a draft once it has been sent."""
        self._drafts.pop(reference, None)

    def __len__(self) -> int:
        return len(self._drafts)


@dataclass(frozen=True, slots=True)
class IntegrationServices:
    """Every external service a tool may use, as interfaces.

    Attributes:
        email: Email providers; several when Gmail and Outlook are both set up.
        calendar: Calendar providers, merged by the calendar tools.
        music: Music providers; the first one controls playback.
        chat: Chat providers (Slack, Discord, Teams).
        smart_home: Smart-home providers.
        web_search: Search backend, ``None`` when web access is not wired.
        transcript_fetcher: YouTube caption source, ``None`` when not wired.
        web_client: HTTP client for reading web pages, ``None`` when not wired.
        drafts: Email drafts created this session (see :class:`EmailDraftBook`).
    """

    email: tuple[EmailProvider, ...] = ()
    calendar: tuple[CalendarProvider, ...] = ()
    music: tuple[MusicProvider, ...] = ()
    chat: tuple[ChatProvider, ...] = ()
    smart_home: tuple[SmartHomeProvider, ...] = ()
    web_search: WebSearchProvider | None = None
    transcript_fetcher: TranscriptFetcher | None = None
    web_client: httpx.AsyncClient | None = None
    drafts: EmailDraftBook = field(default_factory=EmailDraftBook)


def build_web_search_provider(settings: Settings, http: httpx.AsyncClient) -> WebSearchProvider:
    """Brave when ``BRAVE_SEARCH_API_KEY`` is set, keyless DuckDuckGo otherwise (ADR-0012)."""
    if settings.brave_search_api_key is not None:
        log.info("integrations.provider.selected", service="web_search", provider="brave")
        return BraveSearchProvider(http, settings.brave_search_api_key.get_secret_value())
    log.info("integrations.provider.selected", service="web_search", provider="duckduckgo")
    return DuckDuckGoSearchProvider(http)


def build_integration_services(
    settings: Settings,
    http: httpx.AsyncClient,
    *,
    token_store: TokenStore | None = None,
    transcript_fetcher: TranscriptFetcher | None = None,
    interactive_sign_in: bool = False,
    consent_notifier: ConsentNotifier | None = None,
) -> IntegrationServices:
    """Assemble the whole bundle for the composition root in one call.

    Args:
        settings: Credentials and options; absent credentials select stubs.
        http: The one shared client for every REST provider, search and page
            reads. Build it with :func:`jarvis.integrations.web.create_web_client`
            so it carries the honest User-Agent and timeouts.
        token_store: OAuth persistence; defaults to ``data_dir / "tokens"``.
        transcript_fetcher: Caption source; defaults to the real
            ``youtube-transcript-api`` fetcher (imported lazily on first use).
        interactive_sign_in: Allow browser / device-code consent (a terminal
            session only); see :class:`IntegrationFactory`.
        consent_notifier: Shown the first consent error of each sign-in.
    """
    factory = IntegrationFactory(
        settings,
        http,
        token_store,
        interactive_sign_in=interactive_sign_in,
        consent_notifier=consent_notifier,
    )
    return IntegrationServices(
        email=_frozen(factory.email()),
        calendar=_frozen(factory.calendar()),
        music=_frozen(factory.music()),
        chat=_frozen(factory.chat()),
        smart_home=_frozen(factory.smart_home()),
        web_search=build_web_search_provider(settings, http),
        transcript_fetcher=transcript_fetcher or YouTubeTranscriptApiFetcher(),
        web_client=http,
    )


def _frozen[P](providers: Sequence[P]) -> tuple[P, ...]:
    return tuple(providers)


# Design notes:
# - One bundle field on ToolContext instead of one field per service keeps
#   the context stable as services are added and lets the composition root
#   (and tests) pass "no integrations at all" as a single None. Rationale in
#   ADR-0018.
# - Tuples, not lists: the dataclass is frozen and shared by every tool call
#   in a session, so nothing may append a provider behind the root's back.
#   The draft book is the one deliberately mutable member; it is session
#   state that two plugins must share.
# - One factory instance builds every account-backed service so Gmail and
#   Calendar (and Outlook and Teams) share a single sign-in, as factory.py
#   requires.
# - The draft reference format is the same "<provider>:<id>" as message
#   references, so the LLM sees one convention for every id it passes back.
