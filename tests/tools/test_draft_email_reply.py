from collections.abc import Callable

import pytest

from jarvis.agent.registry import ToolRegistry
from jarvis.config import Settings
from jarvis.integrations.services import IntegrationServices
from jarvis.integrations.stubs import StubEmailProvider
from jarvis.tools.draft_email_reply import DraftEmailReplyTool
from tests.tools.integration_helpers import integration_context

REPLY = "Thursday 14:00 works for me.\n\nSee you there,\nB."


@pytest.fixture
def settings(make_settings: Callable[..., Settings]) -> Settings:
    return make_settings()


def test_is_discovered_and_needs_no_confirmation() -> None:
    tool = ToolRegistry.discover().get("draft_email_reply")

    assert isinstance(tool, DraftEmailReplyTool)
    assert not tool.requires_confirmation  # sending is what gets confirmed


async def test_creates_a_draft_and_reads_it_back(settings: Settings) -> None:
    provider = StubEmailProvider()
    services = IntegrationServices(email=(provider,))

    result = await DraftEmailReplyTool().call(
        {"message_id": "stub-email:stub-mail-1", "body": REPLY},
        integration_context(settings, services),
    )

    assert result.ok
    assert result.spoken == (
        'I\'ve drafted a reply to Maya Lindqvist: "Thursday 14:00 works for me." Shall I send it?'
    )
    assert result.data == {
        "draft_id": "stub-email:stub-draft-1",
        "to": ["Maya Lindqvist <maya@example.com>"],
        "subject": "Re: Design review moved to Thursday",
        "first_line": "Thursday 14:00 works for me.",
        "body": REPLY,
    }
    assert "call send_email" in result.detail
    assert provider.sent == []  # nothing left the machine
    assert services.drafts.get("stub-email:stub-draft-1").provider is provider
    assert [call[0] for call in provider.calls] == ["get", "draft_reply"]


async def test_unknown_provider_prefix_fails_cleanly(settings: Settings) -> None:
    services = IntegrationServices(email=(StubEmailProvider(),))

    result = await DraftEmailReplyTool().call(
        {"message_id": "gmail:123", "body": "hi"}, integration_context(settings, services)
    )

    assert not result.ok
    assert "no such provider" in result.detail


async def test_unknown_message_fails_cleanly(settings: Settings) -> None:
    services = IntegrationServices(email=(StubEmailProvider(),))

    result = await DraftEmailReplyTool().call(
        {"message_id": "stub-email:nope", "body": "hi"}, integration_context(settings, services)
    )

    assert not result.ok
    assert "no message with id 'nope'" in result.detail
    assert len(services.drafts) == 0


async def test_empty_body_is_rejected(settings: Settings) -> None:
    services = IntegrationServices(email=(StubEmailProvider(),))

    result = await DraftEmailReplyTool().call(
        {"message_id": "stub-email:stub-mail-1", "body": ""},
        integration_context(settings, services),
    )

    assert not result.ok
    assert "body" in result.detail
