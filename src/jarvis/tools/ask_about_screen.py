"""Tool: answer a free-form question about the user's screen.

Everything that is neither a summary nor an error: "which tab has the
invoice?", "what's the total in this spreadsheet?", "is the build green?".
The user's question goes to the vision model as it is.
"""

from __future__ import annotations

from pydantic import Field

from jarvis.agent.tool import Tool, ToolCategory, ToolContext, ToolResult
from jarvis.tools._screen import ScreenInput, answer_from_screen


class AskAboutScreenInput(ScreenInput):
    """Arguments for :class:`AskAboutScreenTool`."""

    question: str = Field(
        min_length=1,
        max_length=1000,
        description="The user's question about what is on the screen, in their words.",
    )


class AskAboutScreenTool(Tool[AskAboutScreenInput]):
    """Answers a specific question about the screen."""

    name = "ask_about_screen"
    description = (
        "Take a screenshot and answer a specific question about what is visible on the "
        "user's screen, such as a value, a name, a status or where something is. Prefer "
        "describe_screen for summaries and explain_screen_error for errors. The "
        "screenshot is sent to the vision model."
    )
    Input = AskAboutScreenInput
    category = ToolCategory.INFORMATION

    async def run(self, params: AskAboutScreenInput, context: ToolContext) -> ToolResult:
        """Pass the question straight to the vision service."""
        return await answer_from_screen(context, params.question, params.target)
