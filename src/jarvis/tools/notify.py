"""Tool: show a desktop notification.

Useful when the answer should stay visible after Jarvis stops talking ("put
the Wi-Fi password on screen") or when the user asked for a silent cue.
"""

from __future__ import annotations

from typing import Final

from pydantic import Field

from jarvis.agent.tool import Tool, ToolCategory, ToolContext, ToolInput, ToolResult
from jarvis.tools._common import require_platform

MAX_TITLE_CHARS: Final = 64
MAX_MESSAGE_CHARS: Final = 256
"""Windows truncates balloon text around here; longer text would just be cut."""


class NotifyInput(ToolInput):
    """Arguments for :class:`NotifyTool`."""

    title: str = Field(
        default="Jarvis", min_length=1, max_length=MAX_TITLE_CHARS, description="Heading."
    )
    message: str = Field(
        min_length=1, max_length=MAX_MESSAGE_CHARS, description="Body text of the notification."
    )


class NotifyTool(Tool[NotifyInput]):
    """Pops up a system notification."""

    name = "notify"
    description = (
        "Show a desktop notification with a title and a short message. Use when "
        "information should stay visible on screen or the user asked for a notification."
    )
    Input = NotifyInput
    category = ToolCategory.SYSTEM

    async def run(self, params: NotifyInput, context: ToolContext) -> ToolResult:
        """Show it; the spoken line is minimal because the content is on screen."""
        platform = require_platform(context, self.name)
        await platform.notify(params.title, params.message)
        return ToolResult.success(
            "I've put it on screen.",
            detail=f"Notification shown: {params.title}: {params.message}",
            data={"title": params.title, "message": params.message},
        )
