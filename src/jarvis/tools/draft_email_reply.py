"""Tool: prepare a reply to an email as a draft, without sending it.

First half of "reply to Maya and say Thursday works": the reply is created
as a draft in the user's mailbox and read back, and only ``send_email`` (with
the user's spoken confirmation) sends it. The draft is remembered in the
session's :class:`~jarvis.integrations.services.EmailDraftBook` under a
``provider:id`` reference that ``send_email`` takes.
"""

from __future__ import annotations

from typing import Final

from pydantic import Field

from jarvis.agent.tool import Tool, ToolCategory, ToolContext, ToolInput, ToolResult
from jarvis.integrations.base import EmailDraft
from jarvis.tools._integration_support import (
    Service,
    display_name,
    email_providers,
    first_line,
    not_configured,
    resolve_reference,
    spoken_list,
    trim,
)

MAX_BODY_CHARS: Final = 10_000
MAX_REFERENCE_CHARS: Final = 512
_SPOKEN_PREVIEW_CHARS: Final = 120


class DraftEmailReplyInput(ToolInput):
    """Arguments for :class:`DraftEmailReplyTool`."""

    message_id: str = Field(
        min_length=3,
        max_length=MAX_REFERENCE_CHARS,
        description=(
            "Id of the email being answered, exactly as returned by list_unread_email, "
            "summarize_inbox or search_email (for example 'gmail:18c2f...')."
        ),
    )
    body: str = Field(
        min_length=1,
        max_length=MAX_BODY_CHARS,
        description=(
            "The complete plain-text reply, written in the user's voice and ready to send. "
            "Do not include the quoted original or a subject line."
        ),
    )


class DraftEmailReplyTool(Tool[DraftEmailReplyInput]):
    """Creates a reply draft and returns it for the user's spoken approval."""

    name = "draft_email_reply"
    description = (
        "Draft a reply to an email without sending it. Use whenever the user wants to answer "
        "an email; read the draft back and ask whether to send it. Returns draft_id, to, "
        "subject and first_line, which send_email needs verbatim once the user approves."
    )
    Input = DraftEmailReplyInput
    category = ToolCategory.COMMUNICATION

    async def run(self, params: DraftEmailReplyInput, context: ToolContext) -> ToolResult:
        """Fetch the original, create the draft on its provider, remember it."""
        providers = email_providers(context)
        if context.integrations is None or not providers:
            return not_configured(Service.EMAIL)
        provider, message_id = resolve_reference(providers, params.message_id)
        original = await provider.get(message_id)
        draft = await provider.draft_reply(original, params.body)
        reference = context.integrations.drafts.remember(provider, draft)
        return _result(reference, draft)


def _result(reference: str, draft: EmailDraft) -> ToolResult:
    recipients = spoken_list([display_name(address) for address in draft.to])
    opening = first_line(draft.body)
    preview = trim(opening, _SPOKEN_PREVIEW_CHARS)
    return ToolResult.success(
        f'I\'ve drafted a reply to {recipients}: "{preview}" Shall I send it?',
        detail=(
            f"Draft {reference} created, not sent.\nTo: {', '.join(draft.to)}\n"
            f"Subject: {draft.subject}\n\n{draft.body}\n\n"
            "Read it back to the user. Only if they approve, call send_email with draft_id, "
            "to, subject and first_line exactly as given here."
        ),
        data={
            "draft_id": reference,
            "to": list(draft.to),
            "subject": draft.subject,
            "first_line": opening,
            "body": draft.body,
        },
    )


# Design notes:
# - No requires_confirmation: a draft sits in the user's own Drafts folder,
#   reaches nobody and is the very thing the user is asked to approve. Asking
#   before drafting *and* before sending would make one reply cost two
#   confirmations. Sending is the side effect, and send_email confirms it.
# - The original is fetched with ``get`` before drafting because providers
#   thread replies from the full message (Message-ID, thread id), which a
#   listing entry may lack.
