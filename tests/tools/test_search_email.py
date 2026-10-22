from collections.abc import Callable

import pytest

from jarvis.agent.registry import ToolRegistry
from jarvis.config import Settings
from jarvis.integrations.services import IntegrationServices
from jarvis.integrations.stubs import StubEmailProvider
from jarvis.tools.search_email import SearchEmailTool
from tests.tools.integration_helpers import BrokenEmailProvider, integration_context


@pytest.fixture
def settings(make_settings: Callable[..., Settings]) -> Settings:
    return make_settings()


def test_is_discovered_and_read_only() -> None:
    tool = ToolRegistry.discover().get("search_email")

    assert isinstance(tool, SearchEmailTool)
    assert not tool.requires_confirmation


async def test_answers_a_question_from_message_bodies(settings: Settings) -> None:
    provider = StubEmailProvider()
    context = integration_context(settings, IntegrationServices(email=(provider,)))

    result = await SearchEmailTool().call({"query": "venue"}, context)

    assert result.ok
    assert result.spoken == (
        "I found 1 email; the latest is from Maya Lindqvist about Design review moved to Thursday."
    )
    assert "Answer the user's question" in result.detail
    assert "booked on Wednesday" in result.detail
    assert result.data is not None
    assert result.data["query"] == "venue"
    assert result.data["messages"][0]["id"] == "stub-email:stub-mail-1"
    assert ("search", {"query": "venue", "limit": 5}) in provider.calls


async def test_searches_read_mail_too(settings: Settings) -> None:
    context = integration_context(settings, IntegrationServices(email=(StubEmailProvider(),)))

    result = await SearchEmailTool().call({"query": "C++26"}, context)

    assert result.data is not None
    assert result.data["messages"][0]["unread"] is False


async def test_no_hits(settings: Settings) -> None:
    services = IntegrationServices(email=(StubEmailProvider(), BrokenEmailProvider()))

    result = await SearchEmailTool().call(
        {"query": "invoice"}, integration_context(settings, services)
    )

    assert result.ok
    assert result.spoken == "I found no email matching invoice."
    assert "broken-email" in result.detail


async def test_empty_query_is_rejected(settings: Settings) -> None:
    context = integration_context(settings, IntegrationServices(email=(StubEmailProvider(),)))

    result = await SearchEmailTool().call({"query": ""}, context)

    assert not result.ok
    assert "query" in result.detail
