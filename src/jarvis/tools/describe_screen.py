"""Tool: summarise what is on the user's screen.

"What am I looking at?", "summarise this page", "what's on my second
monitor?". A screenshot goes to the configured vision model together with a
prompt that asks for the application, the activity and anything that needs
attention, in a form short enough to be read aloud.
"""

from __future__ import annotations

from typing import Final

from pydantic import Field

from jarvis.agent.tool import Tool, ToolCategory, ToolContext, ToolResult
from jarvis.tools._screen import ScreenInput, answer_from_screen

DESCRIBE_PROMPT: Final = (
    "Summarise what is on this screen for someone who cannot see it right now: which "
    "application or page it is, what the user appears to be doing, and anything that "
    "needs their attention. Three sentences at most."
)


class DescribeScreenInput(ScreenInput):
    """Arguments for :class:`DescribeScreenTool`."""

    focus: str | None = Field(
        default=None,
        max_length=300,
        description="Optional aspect to concentrate on, e.g. 'the chart' or 'the open emails'.",
    )


class DescribeScreenTool(Tool[DescribeScreenInput]):
    """Summarises the screen through the vision service."""

    name = "describe_screen"
    description = (
        "Take a screenshot and summarise what is on the user's screen. Use when the user "
        "asks what they are looking at or wants the screen, a page or a window summarised. "
        "The screenshot is sent to the vision model."
    )
    Input = DescribeScreenInput
    category = ToolCategory.INFORMATION
    keywords = ("skärm", "skärmen", "tittar")

    async def run(self, params: DescribeScreenInput, context: ToolContext) -> ToolResult:
        """Build the summary prompt and ask the vision service."""
        return await answer_from_screen(context, describe_prompt(params.focus), params.target)


def describe_prompt(focus: str | None) -> str:
    """The summary prompt, narrowed to ``focus`` when the user named one."""
    if focus is None or not focus.strip():
        return DESCRIBE_PROMPT
    return f"{DESCRIBE_PROMPT} Concentrate on: {focus.strip()}."
