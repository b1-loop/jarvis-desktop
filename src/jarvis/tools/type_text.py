"""Tool: type text at the cursor in whatever window has focus.

"Type my address", "write 'see you at five' and press enter". Typing goes
to the focused control, which may not be the one the user is thinking of
(a chat box, a terminal), so every call is confirmed with a short preview
of the text.
"""

from __future__ import annotations

from typing import Final

from pydantic import Field

from jarvis.agent.tool import Tool, ToolCategory, ToolContext, ToolInput, ToolResult
from jarvis.tools._common import require_platform, truncate

MAX_TEXT_CHARS: Final = 5000
PREVIEW_CHARS: Final = 60


class TypeTextInput(ToolInput):
    """Arguments for :class:`TypeTextTool`."""

    text: str = Field(
        min_length=1,
        max_length=MAX_TEXT_CHARS,
        description="Exactly what to type. Newlines become the Enter key.",
    )
    press_enter: bool = Field(
        default=False, description="Press Enter after typing, e.g. to send a chat message."
    )


class TypeTextTool(Tool[TypeTextInput]):
    """Injects keystrokes into the focused window after confirmation."""

    name = "type_text"
    description = (
        "Type text at the current cursor position in the focused window, optionally "
        "pressing Enter afterwards. Use switch_window first if the right window may not "
        "have focus. The user is asked to confirm with a preview."
    )
    Input = TypeTextInput
    category = ToolCategory.SYSTEM
    requires_confirmation = True

    def confirmation_prompt(self, params: TypeTextInput) -> str:
        """Quote a preview so the user confirms the words, not just the action."""
        preview = truncate(" ".join(params.text.split()), PREVIEW_CHARS)
        suffix = " and press Enter" if params.press_enter else ""
        return f"Do you want me to type '{preview}'{suffix} into the focused window?"

    async def run(self, params: TypeTextInput, context: ToolContext) -> ToolResult:
        """Name the target window in the result, since it is what the user will check."""
        platform = require_platform(context, self.name)
        window = await platform.active_window()
        await platform.type_text(params.text, press_enter=params.press_enter)
        where = f" into {window.title}" if window is not None else ""
        return ToolResult.success(
            f"Typed it{where}.",
            detail=f"Typed {len(params.text)} characters{where}; enter={params.press_enter}.",
            data={
                "characters": len(params.text),
                "press_enter": params.press_enter,
                "window": window.title if window is not None else None,
            },
        )


# Design notes:
# - The preview collapses whitespace so a multi-line text is still one
#   speakable sentence; the full text is in the arguments the dashboard
#   shows.
# - The text itself is never logged by the tool (it may be a password the
#   user dictated); only its length goes into detail.
