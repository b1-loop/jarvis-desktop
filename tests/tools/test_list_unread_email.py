from collections.abc import Callable
from dataclasses import replace

import pytest

from jarvis.agent.registry import ToolRegistry
from jarvis.config import Settings
from jarvis.integrations.services import IntegrationServices
from jarvis.integrations.stubs import StubEmailProvider
from jarvis.tools.list_unread_email import ListUnreadEmailTool
from tests.tools.integration_helpers import BrokenEmailProvider, integration_context


@pytest.fixture
def settings(make_settings: Callable[..., Settings]) -> Settings:
    return make_settings()


def test_is_discovered_and_read_only() -> None:
    tool = ToolRegistry.discover().get("list_unread_email")

    assert isinstance(tool, ListUnreadEmailTool)
    assert not tool.requires_confirmation


async def test_lists_unread_newest_first_with_references(settings: Settings) -> None:
    provider = StubEmailProvider()
    context = integration_context(settings, IntegrationServices(email=(provider,)))

    result = await ListUnreadEmailTool().call({"limit": 5}, context)

    assert result.ok
    assert result.spoken == (
        "You have 2 unread emails: Maya Lindqvist about Design review moved to Thursday "
        "and GitHub about [jarvis-desktop] CI failed on feat/audio-engine."
    )
    assert result.data is not None
    ids = [message["id"] for message in result.data["messages"]]
    assert ids == ["stub-email:stub-mail-1", "stub-email:stub-mail-2"]
    assert "[stub-email:stub-mail-1]" in result.detail
    assert provider.calls == [("list_unread", {"limit": 5})]


async def test_mentions_at_most_three_and_counts_the_rest(settings: Settings) -> None:
    template = StubEmailProvider().messages[0]
    messages = [replace(template, id=f"m{i}") for i in range(5)]
    context = integration_context(
        settings, IntegrationServices(email=(StubEmailProvider(messages),))
    )

    result = await ListUnreadEmailTool().call({}, context)

    assert result.spoken.endswith(", plus 2 more.")


async def test_empty_inbox(settings: Settings) -> None:
    context = integration_context(settings, IntegrationServices(email=(StubEmailProvider([]),)))

    result = await ListUnreadEmailTool().call({}, context)

    assert result.ok
    assert result.spoken == "You have no unread email."


async def test_one_failing_mailbox_does_not_hide_the_other(settings: Settings) -> None:
    services = IntegrationServices(email=(BrokenEmailProvider(), StubEmailProvider()))

    result = await ListUnreadEmailTool().call({}, integration_context(settings, services))

    assert result.ok
    assert "Some providers failed: broken-email" in result.detail
    assert result.data is not None
    assert result.data["failures"] == [
        {"provider": "broken-email", "error": "broken-email: 503 service unavailable"}
    ]


async def test_every_mailbox_failing_is_a_failed_result(settings: Settings) -> None:
    services = IntegrationServices(email=(BrokenEmailProvider(),))

    result = await ListUnreadEmailTool().call({}, integration_context(settings, services))

    assert not result.ok
    assert "503" in result.detail


async def test_limit_is_bounded(settings: Settings) -> None:
    context = integration_context(settings, IntegrationServices(email=(StubEmailProvider(),)))

    result = await ListUnreadEmailTool().call({"limit": 500}, context)

    assert not result.ok
    assert "limit" in result.detail
