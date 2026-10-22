"""Tool: post a chat message to Slack, Discord or Teams after confirmation.

Posting speaks for the user in front of other people, so the call is always
confirmed, and the prompt reads back the channel, the provider and the
text itself: a misheard channel name or a dictation slip is caught before
it is public.
"""

from __future__ import annotations

from typing import Final

from pydantic import Field

from jarvis.agent.tool import Tool, ToolCategory, ToolContext, ToolInput, ToolResult
from jarvis.tools._integration_support import (
    Service,
    chat_item,
    chat_providers,
    not_configured,
    provider_named,
    trim,
)

MAX_TEXT_CHARS: Final = 4000
_PROMPT_TEXT_CHARS: Final = 200


class SendMessageInput(ToolInput):
    """Arguments for :class:`SendMessageTool`."""

    channel: str = Field(
        min_length=1,
        max_length=128,
        description="Channel or conversation id (preferred, from read_messages) or name.",
    )
    text: str = Field(
        min_length=1, max_length=MAX_TEXT_CHARS, description="The message to post, verbatim."
    )
    provider: str | None = Field(
        default=None,
        max_length=32,
        description="'slack', 'discord' or 'teams'; required when several are connected.",
    )


class SendMessageTool(Tool[SendMessageInput]):
    """Posts a message after the user confirms channel and text."""

    name = "send_message"
    description = (
        "Post a message to a Slack, Discord or Teams channel or chat, e.g. to reply to "
        "something from read_messages. The user is asked to confirm the channel and text "
        "first. Pass provider when more than one chat service is connected."
    )
    Input = SendMessageInput
    category = ToolCategory.COMMUNICATION
    requires_confirmation = True

    def needs_confirmation(self, params: SendMessageInput, context: ToolContext) -> bool:
        """Always ask, except when ``run`` will refuse before posting anything.

        That is the case with no chat provider, or several and none named;
        asking "post this?" and then answering "which service?" would make
        the user confirm something that cannot happen.
        """
        providers = chat_providers(context)
        return bool(providers) and (params.provider is not None or len(providers) == 1)

    def confirmation_prompt(self, params: SendMessageInput) -> str:
        """Read back where the message goes and exactly what it says."""
        where = f"{params.channel} on {params.provider}" if params.provider else params.channel
        return f'Post to {where}: "{trim(params.text, _PROMPT_TEXT_CHARS)}"?'

    async def run(self, params: SendMessageInput, context: ToolContext) -> ToolResult:
        """Pick the provider (refusing to guess between several) and post."""
        providers = chat_providers(context)
        if not providers:
            return not_configured(Service.CHAT)
        if params.provider is None and len(providers) > 1:
            names = [provider.provider_name for provider in providers]
            return ToolResult.failure(
                f"Which service should I post on: {', '.join(names)}?",
                detail="Several chat providers are connected; call again with provider set.",
                data={"providers": list(names)},
            )
        provider = provider_named(providers, params.provider) if params.provider else providers[0]
        message = await provider.send(params.channel, params.text)
        return ToolResult.success(
            f"Posted to {message.channel_name}.",
            detail=f"Message {message.id} posted to {message.channel_id} "
            f"on {provider.provider_name}.",
            data={"message": chat_item(provider, message)},
        )


# Design notes:
# - With several providers and none named, the tool asks instead of
#   guessing: the same channel name can exist on Slack and Discord, and the
#   confirmation prompt (built before run) could not have said which.
# - The full text is in the prompt up to 200 characters; longer messages
#   are cut on a word boundary because a minute of read-back is worse UX
#   than a trimmed preview the user already dictated.
