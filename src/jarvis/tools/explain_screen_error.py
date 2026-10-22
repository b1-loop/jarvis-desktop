"""Tool: explain the error message or stack trace on the user's screen.

"What does this error mean?" while a red stack trace sits in the terminal
or a dialog box blocks the IDE. The prompt makes the model find the error
first, quote its key line, then give the likely cause and a concrete fix,
which is the order a developer needs it in.
"""

from __future__ import annotations

from typing import Final

from pydantic import Field

from jarvis.agent.tool import Tool, ToolCategory, ToolContext, ToolResult
from jarvis.tools._screen import ScreenInput, answer_from_screen

ERROR_PROMPT: Final = (
    "Find the error messages, warnings, failed tests or stack traces on this screen. "
    "Quote the single most important error line exactly. Then explain in plain words "
    "the most likely cause, and give the most likely fix as one to three concrete steps. "
    "If several errors are shown, focus on the first one, since later ones often follow "
    "from it. If no error is visible, say so plainly and do not invent one."
)


class ExplainScreenErrorInput(ScreenInput):
    """Arguments for :class:`ExplainScreenErrorTool`."""

    details: str | None = Field(
        default=None,
        max_length=500,
        description="What the user said about the error, e.g. 'after I ran npm install'.",
    )


class ExplainScreenErrorTool(Tool[ExplainScreenErrorInput]):
    """Reads the error on screen and explains cause and fix."""

    name = "explain_screen_error"
    description = (
        "Take a screenshot and explain the error message, failed test or stack trace "
        "visible on the user's screen: what it says, the likely cause and how to fix it. "
        "Use for 'what does this error mean' or 'why did this fail'. The screenshot is "
        "sent to the vision model."
    )
    Input = ExplainScreenErrorInput
    category = ToolCategory.DEVELOPER

    async def run(self, params: ExplainScreenErrorInput, context: ToolContext) -> ToolResult:
        """Build the error-focused prompt and ask the vision service."""
        return await answer_from_screen(context, error_prompt(params.details), params.target)


def error_prompt(user_context: str | None) -> str:
    """The error prompt, with the user's own description appended when given."""
    if user_context is None or not user_context.strip():
        return ERROR_PROMPT
    return f"{ERROR_PROMPT} The user adds: {user_context.strip()}"
