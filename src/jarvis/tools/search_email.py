"""Tool: search email with a free-text query and return the matching messages.

Backs questions such as "what did the client write about the invoice?". The
query goes to every mailbox's own search (Gmail and Outlook both accept
plain words), hits are merged newest first, and the best few come with
their bodies so the LLM can answer the question rather than only list
subjects.
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
    email_block,
    email_item,
    email_providers,
    flatten,
    gather_each,
    not_configured,
    trim,
    with_bodies,
)

MAX_QUERY_CHARS: Final = 200
MAX_LIMIT: Final = 20
BODIES_FETCHED: Final = 3
"""How many of the top hits are fetched in full; the rest keep their snippet."""
_BODY_CHARS: Final = 1500
_SPOKEN_SUBJECT_CHARS: Final = 60


class SearchEmailInput(ToolInput):
    """Arguments for :class:`SearchEmailTool`."""

    query: str = Field(
        min_length=1,
        max_length=MAX_QUERY_CHARS,
        description=(
            "Words to search for, e.g. 'invoice acme' or a sender's name. Gmail operators "
            "such as from: or newer_than:7d work on Gmail; plain words work everywhere."
        ),
    )
    limit: int = Field(default=5, ge=1, le=MAX_LIMIT, description="Maximum number of hits.")


class SearchEmailTool(Tool[SearchEmailInput]):
    """Searches every mailbox and returns hits, the best with bodies."""

    name = "search_email"
    description = (
        "Search the user's email (read and unread) across all mailboxes. Use to answer "
        "questions about past mail, e.g. 'what did the client write about the invoice?' or "
        "'find the email from Maya about the venue'. Returns matching messages newest first; "
        "the top three include their body text so you can answer from it."
    )
    Input = SearchEmailInput
    category = ToolCategory.COMMUNICATION
    keywords = ("mail", "mejl", "mejlen", "epost", "sök")

    async def run(self, params: SearchEmailInput, context: ToolContext) -> ToolResult:
        """Search each provider, merge, and fetch the leading bodies."""
        providers = email_providers(context)
        if not providers:
            return not_configured(Service.EMAIL)
        gathered = await gather_each(providers, lambda p: p.search(params.query, params.limit))
        pairs = sorted(flatten(gathered), key=lambda pair: pair[1].received_at, reverse=True)
        pairs = await with_bodies(pairs[: params.limit], BODIES_FETCHED)
        items: list[JsonValue] = [
            email_item(provider, message, body_chars=_BODY_CHARS) for provider, message in pairs
        ]
        return ToolResult.success(
            _spoken(params.query, pairs),
            detail=_detail(params.query, pairs, gathered.failure_note()),
            data={"query": params.query, "messages": items, "failures": gathered.failure_data()},
        )


def _spoken(query: str, pairs: list[tuple[EmailProvider, EmailMessage]]) -> str:
    if not pairs:
        return f"I found no email matching {query}."
    newest = pairs[0][1]
    subject = trim(newest.subject, _SPOKEN_SUBJECT_CHARS)
    found = count_phrase(len(pairs), "email")
    return f"I found {found}; the latest is from {display_name(newest.sender)} about {subject}."


def _detail(query: str, pairs: list[tuple[EmailProvider, EmailMessage]], failure_note: str) -> str:
    lines = [f"Results for {query!r}, newest first. Answer the user's question from these."]
    lines.extend(email_block(provider, message, _BODY_CHARS) for provider, message in pairs)
    if failure_note:
        lines.append(failure_note)
    return "\n\n".join(lines)


# Design notes:
# - The query is passed through unchanged: both providers accept plain
#   words, and rewriting into one vendor's syntax would break the other.
#   The description tells the model Gmail operators exist, not that they
#   are required.
# - Only the top BODIES_FETCHED hits are fetched in full; deeper hits are
#   rarely the answer and each body is one more API round trip.
