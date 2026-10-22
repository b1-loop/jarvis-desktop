from collections.abc import Callable

import pytest

from jarvis.agent.confirmation import DECLINED_SPOKEN, ConfirmationPolicy, NotNeeded, Required
from jarvis.agent.registry import ToolRegistry
from jarvis.config import Settings
from jarvis.integrations.services import IntegrationServices
from jarvis.integrations.stubs import StubChatProvider
from jarvis.tools.send_message import SendMessageInput, SendMessageTool
from tests.agent.fakes import FakeConfirmation
from tests.tools.integration_helpers import integration_context


class SecondChat(StubChatProvider):
    provider_name = "discord"


@pytest.fixture
def settings(make_settings: Callable[..., Settings]) -> Settings:
    return make_settings()


def test_is_discovered_and_confirms() -> None:
    tool = ToolRegistry.discover().get("send_message")

    assert isinstance(tool, SendMessageTool)
    assert tool.requires_confirmation


def test_prompt_reads_back_channel_and_text() -> None:
    tool = SendMessageTool()

    assert tool.confirmation_prompt(SendMessageInput(channel="dev", text="On it.")) == (
        'Post to dev: "On it."?'
    )
    assert tool.confirmation_prompt(
        SendMessageInput(channel="dev", text="On it.", provider="slack")
    ) == ('Post to dev on slack: "On it."?')


async def test_confirmed_message_is_posted(settings: Settings) -> None:
    provider = StubChatProvider()
    asker = FakeConfirmation(answers=[True])
    context = integration_context(
        settings, IntegrationServices(chat=(provider,)), confirmation=asker
    )

    result = await ConfirmationPolicy().execute_with_policy(
        SendMessageTool(), {"channel": "C-dev", "text": "Reviewing now."}, context
    )

    assert result.ok
    assert result.spoken == "Posted to C-dev."
    assert asker.prompts == ['Post to C-dev: "Reviewing now."?']
    assert provider.calls == [("send", {"channel": "C-dev", "text": "Reviewing now."})]


async def test_declined_message_is_not_posted(settings: Settings) -> None:
    provider = StubChatProvider()
    context = integration_context(
        settings,
        IntegrationServices(chat=(provider,)),
        confirmation=FakeConfirmation(answers=[False]),
    )

    result = await ConfirmationPolicy().execute_with_policy(
        SendMessageTool(), {"channel": "C-dev", "text": "x"}, context
    )

    assert result.spoken == DECLINED_SPOKEN
    assert provider.calls == []


async def test_several_providers_need_a_name_and_no_pointless_confirmation(
    settings: Settings,
) -> None:
    first, second = StubChatProvider(), SecondChat()
    asker = FakeConfirmation(answers=[])
    context = integration_context(
        settings, IntegrationServices(chat=(first, second)), confirmation=asker
    )
    tool = SendMessageTool()
    ambiguous = {"channel": "dev", "text": "hi"}

    decision = ConfirmationPolicy().decide(tool, tool.parse(ambiguous), context)
    result = await ConfirmationPolicy().execute_with_policy(tool, ambiguous, context)

    assert isinstance(decision, NotNeeded)
    assert not result.ok
    assert result.spoken == "Which service should I post on: stub-chat, discord?"
    assert first.calls == second.calls == []
    named = tool.parse(ambiguous | {"provider": "discord"})
    assert isinstance(ConfirmationPolicy().decide(tool, named, context), Required)


async def test_named_provider_is_used(settings: Settings) -> None:
    first, second = StubChatProvider(), SecondChat()
    context = integration_context(settings, IntegrationServices(chat=(first, second)))

    result = await SendMessageTool().call(
        {"channel": "general", "text": "hi", "provider": "discord"}, context
    )

    assert result.ok
    assert first.calls == []
    assert second.calls == [("send", {"channel": "general", "text": "hi"})]
