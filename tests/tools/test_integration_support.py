from collections.abc import Callable
from datetime import UTC, datetime, timedelta, timezone

import pytest

from jarvis.agent.registry import ToolRegistry
from jarvis.config import Settings
from jarvis.integrations.base import ProviderError
from jarvis.integrations.services import IntegrationServices, UnknownReferenceError
from jarvis.integrations.stubs import StubChatProvider, StubEmailProvider
from jarvis.tools._integration_support import (
    SETUP_VARIABLES,
    Service,
    count_phrase,
    display_name,
    first_line,
    gather_each,
    not_configured,
    provider_named,
    resolve_reference,
    spoken_clock,
    spoken_day,
    spoken_list,
    trim,
    web_client,
)
from tests.tools.integration_helpers import integration_context

NOW = datetime(2026, 10, 21, 15, 42, tzinfo=timezone(timedelta(hours=3)))  # a Wednesday


def test_support_module_is_not_a_plugin() -> None:
    registry = ToolRegistry.discover()

    assert not any(f.module.endswith("_integration_support") for f in registry.failures)
    assert all(type(t).__module__ != "jarvis.tools._integration_support" for t in registry)


@pytest.mark.parametrize("service", list(Service))
def test_not_configured_names_the_variables(service: Service) -> None:
    result = not_configured(service)

    assert not result.ok
    assert result.spoken.startswith("That integration isn't configured")
    for name in SETUP_VARIABLES[service]:
        assert name in result.spoken
        assert name in result.detail
    assert result.data == {
        "configured": False,
        "service": service.value,
        "env_vars": list(SETUP_VARIABLES[service]),
    }


def test_accessors_treat_missing_services_as_empty(
    make_settings: Callable[..., Settings],
) -> None:
    assert web_client(integration_context(make_settings(), None)) is None
    assert web_client(integration_context(make_settings(), IntegrationServices())) is None


async def test_gather_each_reraises_programming_errors() -> None:
    class Buggy(StubChatProvider):
        provider_name = "buggy"

    async def call(provider: StubChatProvider) -> int:
        if provider.provider_name == "buggy":
            raise KeyError("bug")
        return 1

    with pytest.raises(KeyError):
        await gather_each([StubChatProvider(), Buggy()], call)


async def test_gather_each_raises_when_every_provider_fails() -> None:
    async def call(provider: StubChatProvider) -> int:
        raise ProviderError(provider.provider_name, "down")

    with pytest.raises(ProviderError, match="down"):
        await gather_each([StubChatProvider()], call)


async def test_gather_each_with_no_providers_is_empty() -> None:
    async def call(provider: StubChatProvider) -> int:
        return 1

    gathered = await gather_each([], call)

    assert gathered.results == []
    assert gathered.failure_note() == ""


def test_references_resolve_to_the_issuing_provider() -> None:
    email = StubEmailProvider()

    assert resolve_reference([email], "stub-email:abc:def") == (email, "abc:def")
    with pytest.raises(UnknownReferenceError, match="available: stub-email"):
        provider_named([email], "gmail")


@pytest.mark.parametrize(
    ("text", "limit", "expected"),
    [
        ("short", 10, "short"),
        ("one two three four", 10, "one two…"),
        ("unbreakablewordhere", 8, "unbreak…"),
    ],
)
def test_trim_cuts_on_word_boundaries(text: str, limit: int, expected: str) -> None:
    assert trim(text, limit) == expected


def test_text_helpers() -> None:
    assert spoken_list([]) == ""
    assert spoken_list(["a"]) == "a"
    assert spoken_list(["a", "b", "c"]) == "a, b and c"
    assert count_phrase(0, "email") == "no emails"
    assert count_phrase(1, "email") == "1 email"
    assert count_phrase(3, "match", "matches") == "3 matches"
    assert first_line("\n\n  Hello there \nsecond") == "Hello there"
    assert first_line("") == ""
    assert display_name('"Maya L" <maya@example.com>') == "Maya L"
    assert display_name("<bot@example.com>") == "bot@example.com"


@pytest.mark.parametrize(
    ("moment", "expected"),
    [
        (NOW.replace(hour=18), "today"),
        (NOW + timedelta(days=1), "tomorrow"),
        (NOW + timedelta(days=2), "on Friday"),
        (NOW + timedelta(days=20), "on 10 November"),
    ],
)
def test_spoken_day(moment: datetime, expected: str) -> None:
    assert spoken_day(moment, NOW) == expected


def test_spoken_clock_converts_to_the_users_zone() -> None:
    assert spoken_clock(datetime(2026, 10, 21, 9, 5, tzinfo=UTC), NOW) == "12:05 PM"
    assert spoken_clock(datetime(2026, 10, 21, 21, 0, tzinfo=UTC), NOW) == "12:00 AM"
