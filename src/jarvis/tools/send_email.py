"""Tool: send a draft created earlier by ``draft_email_reply``.

The only email tool that lets anything leave the machine, so it always
requires the user's spoken confirmation. The confirmation prompt is built
from the tool's arguments (the policy calls ``confirmation_prompt`` with the
validated input only), so the arguments restate the draft's recipients,
subject and first line, and ``run`` refuses to send if they do not match
the stored draft. The user therefore always approves exactly what is sent.
"""

from __future__ import annotations

from typing import Annotated, Final

from pydantic import Field

from jarvis.agent.tool import Tool, ToolCategory, ToolContext, ToolInput, ToolResult
from jarvis.integrations.base import EmailDraft
from jarvis.integrations.services import UnknownReferenceError
from jarvis.tools._integration_support import (
    Service,
    display_name,
    email_providers,
    first_line,
    not_configured,
    spoken_list,
    trim,
)

MAX_RECIPIENTS: Final = 20
MAX_SUBJECT_CHARS: Final = 300
MAX_LINE_CHARS: Final = 1000
_PROMPT_LINE_CHARS: Final = 120

Recipient = Annotated[str, Field(min_length=3, max_length=320)]


class SendEmailInput(ToolInput):
    """Arguments for :class:`SendEmailTool`; all copied from ``draft_email_reply``."""

    draft_id: str = Field(
        min_length=3, max_length=512, description="draft_id returned by draft_email_reply."
    )
    to: list[Recipient] = Field(
        min_length=1,
        max_length=MAX_RECIPIENTS,
        description="Recipients exactly as returned by draft_email_reply.",
    )
    subject: str = Field(
        max_length=MAX_SUBJECT_CHARS,
        description="Subject exactly as returned by draft_email_reply.",
    )
    first_line: str = Field(
        max_length=MAX_LINE_CHARS,
        description="first_line exactly as returned by draft_email_reply.",
    )


class SendEmailTool(Tool[SendEmailInput]):
    """Sends a previously drafted email after the user confirms it."""

    name = "send_email"
    description = (
        "Send an email draft that draft_email_reply created, after the user has approved it. "
        "Pass draft_id, to, subject and first_line exactly as draft_email_reply returned "
        "them. The user is asked to confirm before anything is sent."
    )
    Input = SendEmailInput
    category = ToolCategory.COMMUNICATION
    requires_confirmation = True

    def confirmation_prompt(self, params: SendEmailInput) -> str:
        """Read back recipient, subject and opening line so the user knows what goes out."""
        recipients = spoken_list([display_name(address) for address in params.to])
        opening = trim(params.first_line, _PROMPT_LINE_CHARS)
        return (
            f'Send the email to {recipients} with the subject "{params.subject}", '
            f'starting "{opening}"?'
        )

    async def run(self, params: SendEmailInput, context: ToolContext) -> ToolResult:
        """Check the arguments against the stored draft, then send it."""
        if context.integrations is None or not email_providers(context):
            return not_configured(Service.EMAIL)
        drafts = context.integrations.drafts
        try:
            stored = drafts.get(params.draft_id)
        except UnknownReferenceError as error:
            return ToolResult.failure(
                "I can't find that draft any more, so nothing was sent. Shall I draft it again?",
                detail=str(error),
            )
        if mismatch := _mismatch(stored.draft, params):
            return ToolResult.failure(
                "The draft doesn't match what I read out, so I didn't send it.",
                detail=f"Refused to send {params.draft_id}: {mismatch}.",
            )
        sent_id = await stored.provider.send(stored.draft)
        drafts.forget(params.draft_id)
        recipients = spoken_list([display_name(address) for address in stored.draft.to])
        return ToolResult.success(
            f"Sent to {recipients}.",
            detail=f"Draft {params.draft_id} sent as message {sent_id}.",
            data={"draft_id": params.draft_id, "sent_id": sent_id, "to": list(stored.draft.to)},
        )


def _mismatch(draft: EmailDraft, params: SendEmailInput) -> str | None:
    """Why the restated fields differ from the draft, or ``None`` if they agree."""
    if {_normalise(a) for a in draft.to} != {_normalise(a) for a in params.to}:
        return f"recipients {list(params.to)} differ from the draft's {list(draft.to)}"
    if draft.subject.strip() != params.subject.strip():
        return f"subject {params.subject!r} differs from the draft's {draft.subject!r}"
    if first_line(draft.body) != params.first_line.strip():
        return "the first line differs from the draft's"
    return None


def _normalise(address: str) -> str:
    return " ".join(address.lower().split())


# Design notes:
# - Restating fields is the price of a confirmation API that sees only the
#   arguments. The alternative (looking the draft up inside
#   confirmation_prompt) would need the context there, i.e. a change to the
#   Tool contract for every plugin. The mismatch check turns the restatement
#   into a safety property instead of a trust exercise.
# - A sent draft is forgotten so a repeated "send it" cannot send twice.
