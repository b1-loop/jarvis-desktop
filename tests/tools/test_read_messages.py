from collections.abc import Callable

import pytest

from jarvis.agent.registry import ToolRegistry
from jarvis.config import Settings
from jarvis.integrations.services import IntegrationServices
from jarvis.integrations.stubs import StubChatProvider
from jarvis.tools.read_messages import ReadMessagesTool
from tests.tools.integration_helpers import integration_context


class SecondChat(StubChatProvider):
    provider_name = "discord"


@pytest.fixture
def settings(make_settings: Callable[..., Settings]) -> Settings:
    return make_settings()


def test_is_discovered_and_read_only() -> None:
    tool = ToolRegistry.discover().get("read_messages")

    assert isinstance(tool, ReadMessagesTool)
    assert not tool.requires_confirmation


async def test_unread_by_default_and_calls_out_mentions(settings: Settings) -> None:
    provider = StubChatProvider()
    context = integration_context(settings, IntegrationServices(chat=(provider,)))

    result = await ReadMessagesTool().call({}, context)

    assert result.ok
    assert result.spoken == (
        "You have 2 unread messages. The latest is from Tom in general: "
        "Lunch at 12:30? Pizza place around the corner. Maya mentioned you in dev."
    )
    assert "[stub-chat #dev (C-dev)]" in result.detail
    assert "(mentions you)" in result.detail
    assert result.data is not None
    assert [m["author"] for m in result.data["messages"]] == ["Tom", "Maya"]
    assert provider.calls[0] == (
        "list_recent",
        {"channel": None, "limit": 15, "unread_only": True, "mentions_only": False},
    )


async def test_mentions_filter(settings: Settings) -> None:
    context = integration_context(settings, IntegrationServices(chat=(StubChatProvider(),)))

    result = await ReadMessagesTool().call({"filter": "mentions"}, context)

    assert result.spoken.startswith("You have 1 message mentioning you. The latest is from Maya")
    assert "mentioned you in" not in result.spoken


async def test_recent_in_one_channel_on_one_provider(settings: Settings) -> None:
    first, second = StubChatProvider(), SecondChat([])
    context = integration_context(settings, IntegrationServices(chat=(first, second)))

    result = await ReadMessagesTool().call(
        {"filter": "recent", "channel": "dev", "provider": "stub-chat"}, context
    )

    assert result.spoken.startswith("You have 2 recent messages.")
    assert second.calls == []


async def test_nothing_new(settings: Settings) -> None:
    context = integration_context(settings, IntegrationServices(chat=(StubChatProvider([]),)))

    result = await ReadMessagesTool().call({}, context)

    assert result.spoken == "You have no unread messages."


async def test_unknown_filter_is_rejected(settings: Settings) -> None:
    context = integration_context(settings, IntegrationServices(chat=(StubChatProvider(),)))

    result = await ReadMessagesTool().call({"filter": "starred"}, context)

    assert not result.ok
    assert "filter" in result.detail
