"""Tool: bring a window to the front by part of its title.

"Switch to Visual Studio Code", "go to my email". Matching is a
case-insensitive substring on the title, front-most match first, which is
how people name windows aloud.
"""

from __future__ import annotations

from typing import Final

from pydantic import Field

from jarvis.agent.tool import Tool, ToolCategory, ToolContext, ToolInput, ToolResult
from jarvis.os_integration.errors import WindowNotFoundError
from jarvis.tools._common import require_platform

MAX_TITLE_CHARS: Final = 200


class SwitchWindowInput(ToolInput):
    """Arguments for :class:`SwitchWindowTool`."""

    title: str = Field(
        min_length=1,
        max_length=MAX_TITLE_CHARS,
        description="Part of the window title, e.g. 'Visual Studio Code' or 'Inbox'.",
    )


class SwitchWindowTool(Tool[SwitchWindowInput]):
    """Focuses the window the user named."""

    name = "switch_window"
    description = (
        "Focus (bring to front and restore) the front-most window whose title contains "
        "the given text, case-insensitively. Use for 'switch to X' or 'go to X'. "
        "If it fails, call list_windows to find the real title."
    )
    Input = SwitchWindowInput
    category = ToolCategory.SYSTEM

    async def run(self, params: SwitchWindowInput, context: ToolContext) -> ToolResult:
        """Focus by title; a miss is a normal failed result the LLM can recover from."""
        platform = require_platform(context, self.name)
        try:
            window = await platform.focus_window_by_title(params.title)
        except WindowNotFoundError as error:
            return ToolResult.failure(
                f"I can't find a window called {params.title}.", detail=str(error)
            )
        return ToolResult.success(
            f"Switched to {window.title}.",
            detail=f"Focused {window.title!r} ({window.process_name}, handle {window.handle}).",
            data={"handle": window.handle, "title": window.title, "pid": window.pid},
        )
