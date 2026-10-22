"""Tool: save a dictated note.

"Note that the parking code is 4711", "make a note: call the plumber about
the leak". Notes go to long-term memory, where they are full-text
searchable (search_notes) and also recalled semantically in later
conversations.
"""

from __future__ import annotations

from typing import Final

from pydantic import Field

from jarvis.agent.tool import Tool, ToolCategory, ToolContext, ToolInput, ToolResult
from jarvis.tools._common import require_memory

MAX_NOTE_CHARS: Final = 10_000
MAX_TITLE_CHARS: Final = 200


class AddNoteInput(ToolInput):
    """Arguments for :class:`AddNoteTool`."""

    content: str = Field(
        min_length=1,
        max_length=MAX_NOTE_CHARS,
        description="The note text, as the user dictated it (fix obvious transcription slips).",
    )
    title: str = Field(
        default="",
        max_length=MAX_TITLE_CHARS,
        description="Optional short heading, e.g. 'Parking'. Leave empty if none was given.",
    )


class AddNoteTool(Tool[AddNoteInput]):
    """Stores a note in memory."""

    name = "add_note"
    description = (
        "Save a note the user dictates (optionally with a short title) to long-term memory. "
        "Use for 'note that...', 'remember to write down...'. For tasks to tick off use "
        "add_todo; for something at a specific time use add_reminder."
    )
    Input = AddNoteInput
    category = ToolCategory.PRODUCTIVITY
    keywords = ("anteckna", "anteckning", "notera")

    async def run(self, params: AddNoteInput, context: ToolContext) -> ToolResult:
        """Store and confirm briefly."""
        memory = require_memory(context, self.name)
        note = await memory.add_note(params.content, params.title)
        return ToolResult.success(
            "Noted.",
            detail=f"Saved note {note.id}" + (f" '{note.title}'" if note.title else ""),
            data={"id": note.id, "title": note.title},
        )


# Design notes:
# - No confirmation for any notes/todo/reminder tool: they only write to
#   Jarvis's own store, exactly what the user just dictated, and nothing
#   leaves the machine.
