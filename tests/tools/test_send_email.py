from collections.abc import Callable
from typing import Any

import pytest

from jarvis.agent.confirmation import DECLINED_SPOKEN, ConfirmationPolicy
from jarvis.agent.registry import ToolRegistry
from jarvis.config import Settings
from jarvis.integrations.services import IntegrationServices
from jarvis.integrations.stubs import StubEmailProvider
from jarvis.tools.draft_email_reply import DraftEmailReplyTool
from jarvis.tools.send_email import SendEmailInput, SendEmailTool
from tests.agent.fakes import FakeConfirmation
from tests.tools.integration_helpers import integration_context


@pytest.fixture
def settings(make_settings: Callable[..., Settings]) -> Settings:
    return make_settings()


@pytest.fixture
def provider() -> StubEmailProvider:
    return StubEmailProvider()


@pytest.fixture
def services(provider: StubEmailProvider) -> IntegrationServices:
    return IntegrationServices(email=(provider,))


async def _draft(settings: Settings, services: IntegrationServices) -> dict[str, Any]:
    result = await DraftEmailReplyTool().call(
        {"message_id": "stub-email:stub-mail-1", "body": "Thursday works.\nB."},
        integration_context(settings, services),
    )
    assert result.data is not None
    return {key: result.data[key] for key in ("draft_id", "to", "subject", "first_line")}


def test_is_discovered_and_always_confirms() -> None:
    tool = ToolRegistry.discover().get("send_email")

    assert isinstance(tool, SendEmailTool)
    assert tool.requires_confirmation


def test_prompt_reads_recipient_subject_and_first_line() -> None:
    params = SendEmailInput(
        draft_id="stub-email:d1",
        to=["Maya Lindqvist <maya@example.com>"],
        subject="Re: Venue",
        first_line="Thursday works.",
    )

    assert SendEmailTool().confirmation_prompt(params) == (
        'Send the email to Maya Lindqvist with the subject "Re: Venue", starting "Thursday works."?'
    )


async def test_confirmed_send_goes_out_once(
    settings: Settings, services: IntegrationServices, provider: StubEmailProvider
) -> None:
    arguments = await _draft(settings, services)
    asker = FakeConfirmation(answers=[True])
    context = integration_context(settings, services, confirmation=asker)

    result = await ConfirmationPolicy().execute_with_policy(SendEmailTool(), arguments, context)

    assert result.ok
    assert result.spoken == "Sent to Maya Lindqvist."
    assert asker.prompts == [
        'Send the email to Maya Lindqvist with the subject "Re: Design review moved to '
        'Thursday", starting "Thursday works."?'
    ]
    assert [draft.id for draft in provider.sent] == ["stub-draft-1"]
    again = await SendEmailTool().call(arguments, context)
    assert not again.ok  # the draft is forgotten after sending
    assert "can't find that draft" in again.spoken
    assert len(provider.sent) == 1


async def test_declined_send_sends_nothing(
    settings: Settings, services: IntegrationServices, provider: StubEmailProvider
) -> None:
    arguments = await _draft(settings, services)
    context = integration_context(
        settings, services, confirmation=FakeConfirmation(answers=[False])
    )

    result = await ConfirmationPolicy().execute_with_policy(SendEmailTool(), arguments, context)

    assert result.spoken == DECLINED_SPOKEN
    assert provider.sent == []


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        ("to", ["someone.else@example.com"], "recipients"),
        ("subject", "Something else", "subject"),
        ("first_line", "Friday works.", "first line"),
    ],
)
async def test_mismatched_restatement_is_refused(
    settings: Settings,
    services: IntegrationServices,
    field: str,
    value: object,
    reason: str,
) -> None:
    arguments = await _draft(settings, services) | {field: value}

    result = await SendEmailTool().call(arguments, integration_context(settings, services))

    assert not result.ok
    assert result.spoken == "The draft doesn't match what I read out, so I didn't send it."
    assert reason in result.detail
    (provider,) = services.email
    assert isinstance(provider, StubEmailProvider)
    assert provider.sent == []


async def test_recipient_comparison_ignores_case_and_spacing(
    settings: Settings, services: IntegrationServices, provider: StubEmailProvider
) -> None:
    arguments = await _draft(settings, services)
    arguments["to"] = ["maya lindqvist   <MAYA@example.com>"]

    result = await SendEmailTool().call(arguments, integration_context(settings, services))

    assert result.ok
    assert len(provider.sent) == 1


async def test_without_email_integration(settings: Settings) -> None:
    result = await SendEmailTool().call(
        {"draft_id": "x:1", "to": ["a@b.co"], "subject": "", "first_line": ""},
        integration_context(settings, IntegrationServices()),
    )

    assert not result.ok
    assert "isn't configured" in result.spoken
