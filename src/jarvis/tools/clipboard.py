"""Tools: read the clipboard text and replace it.

Two tiny siblings over ``Platform.get_clipboard_text`` and
``set_clipboard_text``, kept in one module because each is a handful of
lines and they share the size limits. "What's on my clipboard", "summarise
what I copied", "copy that to the clipboard".
"""

from __future__ import annotations

from typing import Final

from pydantic import Field

from jarvis.agent.tool import Tool, ToolCategory, ToolContext, ToolInput, ToolResult
from jarvis.tools._common import require_platform, truncate

MAX_CLIPBOARD_CHARS: Final = 50_000
"""Cap on text handed to the LLM or written to the clipboard."""
_SPOKEN_PREVIEW_CHARS: Final = 80


class ClipboardGetInput(ToolInput):
    """No arguments."""


class ClipboardGetTool(Tool[ClipboardGetInput]):
    """Returns the clipboard text for the LLM to read, summarise or reuse."""

    name = "clipboard_get"
    description = (
        "Read the text currently on the clipboard. Returns the full text (capped) for "
        "you to summarise, translate or reuse; empty if the clipboard holds no text."
    )
    Input = ClipboardGetInput
    category = ToolCategory.SYSTEM

    async def run(self, params: ClipboardGetInput, context: ToolContext) -> ToolResult:
        """Read and cap; the spoken line is only a preview."""
        del params
        platform = require_platform(context, self.name)
        text = await platform.get_clipboard_text()
        if not text:
            return ToolResult.success("The clipboard has no text on it.", data={"text": ""})
        capped = truncate(text, MAX_CLIPBOARD_CHARS)
        preview = truncate(" ".join(text.split()), _SPOKEN_PREVIEW_CHARS)
        return ToolResult.success(
            f"The clipboard says: {preview}",
            detail=capped,
            data={"text": capped, "characters": len(text), "truncated": capped != text},
        )


class ClipboardSetInput(ToolInput):
    """Arguments for :class:`ClipboardSetTool`."""

    text: str = Field(
        min_length=1, max_length=MAX_CLIPBOARD_CHARS, description="Text to put on the clipboard."
    )


class ClipboardSetTool(Tool[ClipboardSetInput]):
    """Puts text on the clipboard so the user can paste it."""

    name = "clipboard_set"
    description = (
        "Replace the clipboard contents with the given text so the user can paste it. "
        "Use for 'copy that', 'put this on my clipboard'."
    )
    Input = ClipboardSetInput
    category = ToolCategory.SYSTEM

    async def run(self, params: ClipboardSetInput, context: ToolContext) -> ToolResult:
        """Write the text."""
        platform = require_platform(context, self.name)
        await platform.set_clipboard_text(params.text)
        return ToolResult.success("Copied to the clipboard.", data={"characters": len(params.text)})


# Design notes:
# - clipboard_set needs no confirmation: the user asked for it, and the
#   clipboard is a scratch buffer they overwrite with every Ctrl+C.
