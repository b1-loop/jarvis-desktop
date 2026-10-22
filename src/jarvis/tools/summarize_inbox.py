"""Tool: unread email with bodies, structured for the LLM to summarise.

The tool does not summarise; the model does, because only the model knows
what the user asked ("anything urgent?", "what does Maya want?"). The tool's
job is to hand over the right material within a budget: the unread
messages, newest first, each with a trimmed body, plus per-sender counts so
"you have five newsletters" is cheap to say.
"""

from __future__ import annotations

from collections import Counter
from typing import Final

from pydantic import Field, JsonValue

from jarvis.agent.tool import Tool, ToolCategory, ToolContext, ToolInput, ToolResult
from jarvis.integrations.base import EmailMessage, EmailProvider
from jarvis.tools._integration_support import (
    Service,
    count_phrase,
    display_name,
    email_block,
    email_item,
    email_providers,
    flatten,
    gather_each,
    not_configured,
    spoken_list,
    with_bodies,
)

MAX_MESSAGES: Final = 30
MIN_BODY_CHARS: Final = 100
MAX_BODY_CHARS: Final = 2000
_SPOKEN_SENDERS: Final = 3
_INSTRUCTION: Final = (
    "Summarise these unread emails for the user in a few spoken sentences: lead with "
    "anything that needs action or a reply, group newsletters and notifications, and "
    "mention senders by name."
)


class SummarizeInboxInput(ToolInput):
    """Arguments for :class:`SummarizeInboxTool`."""

    limit: int = Field(
        default=15, ge=1, le=MAX_MESSAGES, description="How many unread messages to include."
    )
    body_chars: int = Field(
        default=600,
        ge=MIN_BODY_CHARS,
        le=MAX_BODY_CHARS,
        description="Characters of each message body to include; lower keeps the answer short.",
    )


class SummarizeInboxTool(Tool[SummarizeInboxInput]):
    """Returns unread messages with bodies for an LLM-written summary."""

    name = "summarize_inbox"
    description = (
        "Fetch the user's unread emails with their (trimmed) bodies so you can summarise the "
        "inbox: what is important, who needs a reply, what can wait. Use for 'summarise my "
        "inbox' or 'anything urgent in my email?'. Returns structured messages with ids "
        "usable with draft_email_reply; you write the summary."
    )
    Input = SummarizeInboxInput
    category = ToolCategory.COMMUNICATION
    keywords = ("mail", "mejl", "inkorg", "inkorgen", "sammanfatta")

    async def run(self, params: SummarizeInboxInput, context: ToolContext) -> ToolResult:
        """List unread mail, fetch bodies, and package it with an instruction."""
        providers = email_providers(context)
        if not providers:
            return not_configured(Service.EMAIL)
        gathered = await gather_each(providers, lambda p: p.list_unread(params.limit))
        pairs = sorted(flatten(gathered), key=lambda pair: pair[1].received_at, reverse=True)
        pairs = await with_bodies(pairs[: params.limit], params.limit)
        senders = Counter(display_name(message.sender) for _, message in pairs)
        items: list[JsonValue] = [
            email_item(provider, message, body_chars=params.body_chars)
            for provider, message in pairs
        ]
        return ToolResult.success(
            _spoken(pairs, senders),
            detail=_detail(pairs, params.body_chars, gathered.failure_note()),
            data={
                "unread_count": len(pairs),
                "by_sender": dict(senders.most_common()),
                "messages": items,
                "failures": gathered.failure_data(),
            },
        )


def _spoken(pairs: list[tuple[EmailProvider, EmailMessage]], senders: Counter[str]) -> str:
    if not pairs:
        return "Your inbox is clear: no unread email."
    names = [name for name, _ in senders.most_common(_SPOKEN_SENDERS)]
    others = len(senders) - len(names)
    tail = f" and {count_phrase(others, 'other')}" if others else ""
    joined = ", ".join(names) + tail if others else spoken_list(names)
    return f"You have {count_phrase(len(pairs), 'unread email')}, from {joined}."


def _detail(
    pairs: list[tuple[EmailProvider, EmailMessage]], body_chars: int, failure_note: str
) -> str:
    blocks = [_INSTRUCTION]
    blocks.extend(email_block(provider, message, body_chars) for provider, message in pairs)
    if failure_note:
        blocks.append(failure_note)
    return "\n\n".join(blocks)


# Design notes:
# - Bodies are fetched for every listed message (concurrently) because a
#   snippet is often a greeting; ``body_chars`` and ``limit`` bound the cost
#   to at most 30 x 2000 characters, well inside the context budget.
# - The instruction travels in ``detail`` rather than the system prompt so
#   the summarising guidance lives next to the data it applies to.
