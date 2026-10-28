from collections.abc import Callable
from dataclasses import FrozenInstanceError
from pathlib import Path

import httpx
import pytest

from jarvis.config import Settings
from jarvis.integrations.base import EmailDraft
from jarvis.integrations.services import (
    EmailDraftBook,
    IntegrationServices,
    UnknownReferenceError,
    build_integration_services,
    build_web_search_provider,
    make_reference,
    split_reference,
)
from jarvis.integrations.stubs import (
    StubCalendarProvider,
    StubChatProvider,
    StubEmailProvider,
    StubMusicProvider,
    StubSmartHomeProvider,
)
from jarvis.integrations.web import BraveSearchProvider, DuckDuckGoSearchProvider
from jarvis.integrations.youtube import YouTubeTranscriptApiFetcher


def test_references_round_trip_and_split_on_the_first_separator() -> None:
    reference = make_reference("outlook", "AAMk:x/y=")

    assert reference == "outlook:AAMk:x/y="
    assert split_reference(reference) == ("outlook", "AAMk:x/y=")


@pytest.mark.parametrize("reference", ["no-separator", ":id-only", "provider:"])
def test_malformed_references_are_rejected(reference: str) -> None:
    with pytest.raises(UnknownReferenceError, match="expected '<provider>:<id>'"):
        split_reference(reference)


def test_draft_book_remembers_and_forgets() -> None:
    book = EmailDraftBook()
    provider = StubEmailProvider()
    draft = EmailDraft(id="d1", to=("a@example.com",), subject="Re: x", body="Hi")

    reference = book.remember(provider, draft)

    assert reference == "stub-email:d1"
    assert book.get(reference).draft is draft
    assert book.get(reference).provider is provider
    assert len(book) == 1
    book.forget(reference)
    assert len(book) == 0
    with pytest.raises(UnknownReferenceError, match="no draft with that id"):
        book.get(reference)


def test_the_bundle_is_frozen_and_empty_by_default() -> None:
    services = IntegrationServices()

    assert services.email == ()
    assert services.web_search is None
    with pytest.raises(FrozenInstanceError):
        services.email = (StubEmailProvider(),)  # type: ignore[misc]


async def test_builder_selects_stubs_in_demo_mode_without_credentials(
    make_settings: Callable[..., Settings], tmp_path: Path
) -> None:
    settings = make_settings(data_dir=tmp_path, demo_integrations=True)
    async with httpx.AsyncClient() as http:
        services = build_integration_services(settings, http)

    assert [type(p) for p in services.email] == [StubEmailProvider]
    assert [type(p) for p in services.calendar] == [StubCalendarProvider]
    assert [type(p) for p in services.music] == [StubMusicProvider]
    assert [type(p) for p in services.chat] == [StubChatProvider]
    assert [type(p) for p in services.smart_home] == [StubSmartHomeProvider]
    assert isinstance(services.web_search, DuckDuckGoSearchProvider)
    assert isinstance(services.transcript_fetcher, YouTubeTranscriptApiFetcher)
    assert services.web_client is http
    assert len(services.drafts) == 0


async def test_builder_accepts_an_injected_transcript_fetcher(
    make_settings: Callable[..., Settings], tmp_path: Path
) -> None:
    fetcher = YouTubeTranscriptApiFetcher()
    async with httpx.AsyncClient() as http:
        services = build_integration_services(
            make_settings(data_dir=tmp_path), http, transcript_fetcher=fetcher
        )

    assert services.transcript_fetcher is fetcher


async def test_brave_is_chosen_when_its_key_is_set(make_settings: Callable[..., Settings]) -> None:
    async with httpx.AsyncClient() as http:
        provider = build_web_search_provider(make_settings(BRAVE_SEARCH_API_KEY="k"), http)

    assert isinstance(provider, BraveSearchProvider)


def test_a_bundle_without_web_search_leaves_it_unset() -> None:
    bundle = IntegrationServices(email=(StubEmailProvider(),), chat=(StubChatProvider(),))

    assert bundle.web_search is None  # web tools then answer "not configured"


async def test_builder_leaves_unconnected_services_empty_by_default(
    make_settings: Callable[..., Settings], tmp_path: Path
) -> None:
    async with httpx.AsyncClient() as http:
        services = build_integration_services(make_settings(data_dir=tmp_path), http)

    assert services.email == services.calendar == services.music == ()
    assert services.chat == services.smart_home == ()
    assert isinstance(services.web_search, DuckDuckGoSearchProvider)  # keyless, always real
