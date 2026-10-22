"""Tool: the user's unread email across every connected mailbox.

Answers "any new mail?" and "who wrote to me?". Gmail and Outlook are
queried concurrently and merged newest first; each item carries a
``provider:id`` reference that ``draft_email_reply`` and ``draft_document``
accept, so the LLM can act on a message it just listed.
"""

from __future__ import annotations

from typing import Final

from pydantic import Field, JsonValue

from jarvis.agent.tool import Tool, ToolCategory, ToolContext, ToolInput, ToolResult
from jarvis.integrations.base import EmailMessage, EmailProvider
from jarvis.tools._integration_support import (
    Service,
    count_phrase,
    display_name,
    email_item,
    email_line,
    email_providers,
    flatten,
    gather_each,
    not_configured,
    spoken_list,
    trim,
)

MAX_LIMIT: Final = 25
_SPOKEN_MESSAGES: Final = 3
_SPOKEN_SUBJECT_CHARS: Final = 60


class ListUnreadEmailInput(ToolInput):
    """Arguments for :class:`ListUnreadEmailTool`."""

    limit: int = Field(
        default=10,
        ge=1,
        le=MAX_LIMIT,
        description="Maximum number of unread messages to return across all mailboxes.",
    )


class ListUnreadEmailTool(Tool[ListUnreadEmailInput]):
    """Lists unread mail, newest first, from every email provider."""

    name = "list_unread_email"
    description = (
        "List the user's unread emails across all connected mailboxes (Gmail, Outlook), "
        "newest first. Use for 'do I have new mail?' or 'who emailed me?'. Returns sender, "
        "subject, snippet, time and an id usable with draft_email_reply. For a content "
        "summary of the inbox use summarize_inbox instead."
    )
    Input = ListUnreadEmailInput
    category = ToolCategory.COMMUNICATION

    async def run(self, params: ListUnreadEmailInput, context: ToolContext) -> ToolResult:
        """Fan out ``list_unread`` and merge by receipt time."""
        providers = email_providers(context)
        if not providers:
            return not_configured(Service.EMAIL)
        gathered = await gather_each(providers, lambda p: p.list_unread(params.limit))
        pairs = sorted(flatten(gathered), key=lambda pair: pair[1].received_at, reverse=True)
        pairs = pairs[: params.limit]
        items: list[JsonValue] = [email_item(provider, message) for provider, message in pairs]
        lines = [email_line(provider, message) for provider, message in pairs]
        detail = "\n".join([f"{len(pairs)} unread message(s):", *lines, gathered.failure_note()])
        return ToolResult.success(
            _spoken(pairs),
            detail=detail.strip(),
            data={"messages": items, "failures": gathered.failure_data()},
        )


def _spoken(pairs: list[tuple[EmailProvider, EmailMessage]]) -> str:
    if not pairs:
        return "You have no unread email."
    mentions = [
        f"{display_name(message.sender)} about {trim(message.subject, _SPOKEN_SUBJECT_CHARS)}"
        for _, message in pairs[:_SPOKEN_MESSAGES]
    ]
    more = len(pairs) - len(mentions)
    tail = f", plus {more} more" if more else ""
    return f"You have {count_phrase(len(pairs), 'unread email')}: {spoken_list(mentions)}{tail}."


# Design notes:
# - ``limit`` applies to the merged list, and each provider is asked for the
#   same ``limit``, so the newest N overall are correct without paging.
# - A failing mailbox is reported in ``detail`` and ``data.failures`` while
#   the other mailbox's mail is still returned (see gather_each).
