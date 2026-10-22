from collections.abc import Callable
from dataclasses import replace

import pytest

from jarvis.agent.registry import ToolRegistry
from jarvis.config import Settings
from jarvis.integrations.services import IntegrationServices
from jarvis.integrations.stubs import StubEmailProvider
from jarvis.tools.summarize_inbox import SummarizeInboxTool
from tests.tools.integration_helpers import BodylessEmailProvider, integration_context


@pytest.fixture
def settings(make_settings: Callable[..., Settings]) -> Settings:
    return make_settings()


def test_is_discovered_and_read_only() -> None:
    tool = ToolRegistry.discover().get("summarize_inbox")

    assert isinstance(tool, SummarizeInboxTool)
    assert not tool.requires_confirmation


async def test_returns_bodies_and_sender_counts_for_the_llm(settings: Settings) -> None:
    provider = StubEmailProvider()
    context = integration_context(settings, IntegrationServices(email=(provider,)))

    result = await SummarizeInboxTool().call({"body_chars": 100}, context)

    assert result.ok
    assert result.spoken == "You have 2 unread emails, from Maya Lindqvist and GitHub."
    assert result.detail.startswith("Summarise these unread emails")
    assert "The venue is booked on Wednesday." in result.detail  # full body, not snippet
    assert result.data is not None
    assert result.data["unread_count"] == 2
    assert result.data["by_sender"] == {"Maya Lindqvist": 1, "GitHub": 1}
    assert result.data["messages"][0]["body"].endswith("Maya")
    assert [call[0] for call in provider.calls] == ["list_unread", "get", "get"]


async def test_many_senders_are_counted_not_listed(settings: Settings) -> None:
    template = StubEmailProvider().messages[0]
    messages = [replace(template, id=f"m{i}", sender=f"Person {i} <p{i}@x.org>") for i in range(5)]
    context = integration_context(
        settings, IntegrationServices(email=(StubEmailProvider(messages),))
    )

    result = await SummarizeInboxTool().call({}, context)

    assert (
        result.spoken == "You have 5 unread emails, from Person 0, Person 1, Person 2 and 2 others."
    )


async def test_a_body_that_cannot_be_fetched_keeps_its_snippet(settings: Settings) -> None:
    listing = [replace(m, body="") for m in StubEmailProvider().messages]
    provider = BodylessEmailProvider(listing)
    context = integration_context(settings, IntegrationServices(email=(provider,)))

    result = await SummarizeInboxTool().call({}, context)

    assert result.ok
    assert "The venue is booked." in result.detail  # the snippet's wording
    assert result.data is not None
    assert "body" not in result.data["messages"][0]


async def test_clear_inbox(settings: Settings) -> None:
    context = integration_context(settings, IntegrationServices(email=(StubEmailProvider([]),)))

    result = await SummarizeInboxTool().call({}, context)

    assert result.spoken == "Your inbox is clear: no unread email."
