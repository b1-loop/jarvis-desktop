"""Tool: recent, unread or mentioning chat messages from Slack, Discord and Teams.

"Anything I missed on Slack?" and "did anyone mention me?" become one call
across every connected chat provider (or one, if the user names it). Each
message carries its channel id so ``send_message`` can reply in the right
place.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Final

from pydantic import Field, JsonValue

from jarvis.agent.tool import Tool, ToolCategory, ToolContext, ToolInput, ToolResult
from jarvis.integrations.base import ChatMessage, ChatProvider
from jarvis.tools._integration_support import (
    Service,
    chat_item,
    chat_providers,
    count_phrase,
    flatten,
    gather_each,
    not_configured,
    provider_named,
    trim,
)

MAX_LIMIT: Final = 50
_SPOKEN_TEXT_CHARS: Final = 100


class MessageFilter(StrEnum):
    """Which messages to read."""

    RECENT = "recent"
    UNREAD = "unread"
    MENTIONS = "mentions"


_NOUNS: Final = {
    MessageFilter.RECENT: ("recent message", "recent messages"),
    MessageFilter.UNREAD: ("unread message", "unread messages"),
    MessageFilter.MENTIONS: ("message mentioning you", "messages mentioning you"),
}


class ReadMessagesInput(ToolInput):
    """Arguments for :class:`ReadMessagesTool`."""

    filter: MessageFilter = Field(
        default=MessageFilter.UNREAD,
        description="'unread' for what the user missed, 'mentions' for messages naming them, "
        "'recent' for the latest regardless of read state.",
    )
    channel: str | None = Field(
        default=None,
        min_length=1,
        max_length=128,
        description="Channel or conversation id or name; omit for every reachable channel.",
    )
    provider: str | None = Field(
        default=None,
        max_length=32,
        description="Only this provider ('slack', 'discord', 'teams'); omit for all.",
    )
    limit: int = Field(default=15, ge=1, le=MAX_LIMIT, description="Maximum messages overall.")


class ReadMessagesTool(Tool[ReadMessagesInput]):
    """Reads chat messages across providers, newest first."""

    name = "read_messages"
    description = (
        "Read chat messages from Slack, Discord and Teams: unread ones (default), ones that "
        "mention the user, or the most recent. Use for 'anything new on Slack?' or 'did "
        "anyone mention me?'. Returns author, channel, text and time, newest first, with the "
        "channel id and provider that send_message needs to reply."
    )
    Input = ReadMessagesInput
    category = ToolCategory.COMMUNICATION

    async def run(self, params: ReadMessagesInput, context: ToolContext) -> ToolResult:
        """Query the chosen providers with the filter and merge by time."""
        providers = chat_providers(context)
        if not providers:
            return not_configured(Service.CHAT)
        chosen = (provider_named(providers, params.provider),) if params.provider else providers
        gathered = await gather_each(
            chosen,
            lambda p: p.list_recent(
                params.channel,
                params.limit,
                unread_only=params.filter is MessageFilter.UNREAD,
                mentions_only=params.filter is MessageFilter.MENTIONS,
            ),
        )
        pairs = sorted(flatten(gathered), key=lambda pair: pair[1].sent_at, reverse=True)
        pairs = pairs[: params.limit]
        items: list[JsonValue] = [chat_item(provider, message) for provider, message in pairs]
        return ToolResult.success(
            _spoken(pairs, params.filter),
            detail=_detail(pairs, gathered.failure_note()),
            data={
                "filter": params.filter.value,
                "messages": items,
                "failures": gathered.failure_data(),
            },
        )


def _spoken(pairs: list[tuple[ChatProvider, ChatMessage]], kind: MessageFilter) -> str:
    found = f"You have {count_phrase(len(pairs), *_NOUNS[kind])}."
    if not pairs:
        return found
    latest = pairs[0][1]
    text = trim(latest.text, _SPOKEN_TEXT_CHARS)
    spoken = f"{found} The latest is from {latest.author} in {latest.channel_name}: {text}"
    mentions = [message for _, message in pairs if message.mentions_me]
    if mentions and kind is not MessageFilter.MENTIONS:
        spoken += f" {mentions[0].author} mentioned you in {mentions[0].channel_name}."
    return spoken


def _detail(pairs: list[tuple[ChatProvider, ChatMessage]], failure_note: str) -> str:
    lines = [f"{len(pairs)} message(s), newest first:"]
    for provider, message in pairs:
        where = f"{provider.provider_name} #{message.channel_name} ({message.channel_id})"
        flag = " (mentions you)" if message.mentions_me else ""
        when = f"{message.sent_at:%Y-%m-%d %H:%M}"
        lines.append(f"- [{where}] {when} {message.author}{flag}: {message.text}")
    lines.append(failure_note)
    return "\n".join(lines).strip()


# Design notes:
# - Unread is the default filter because "what did I miss" is the common
#   question; providers without unread state (Discord bots, see ADR-0009)
#   return nothing for it, and the description steers the model to 'recent'.
# - A mention is called out even in an unread listing: it is the one
#   message the user most likely needs to hear about first.
