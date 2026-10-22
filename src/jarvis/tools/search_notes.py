"""Tool: full-text search over the user's notes.

"What was the parking code?", "find my notes about the plumber". Returns the
matching notes in full so the LLM can answer from them.
"""

from __future__ import annotations

from typing import Final

from pydantic import Field, JsonValue

from jarvis.agent.tool import Tool, ToolCategory, ToolContext, ToolInput, ToolResult
from jarvis.tools._common import require_memory, truncate

MAX_QUERY_CHARS: Final = 200
DEFAULT_LIMIT: Final = 5
MAX_LIMIT: Final = 20
_SPOKEN_NOTE_CHARS: Final = 100


class SearchNotesInput(ToolInput):
    """Arguments for :class:`SearchNotesTool`."""

    query: str = Field(
        min_length=1,
        max_length=MAX_QUERY_CHARS,
        description="Key words to look for, e.g. 'parking code'.",
    )
    limit: int = Field(default=DEFAULT_LIMIT, ge=1, le=MAX_LIMIT, description="Maximum notes.")


class SearchNotesTool(Tool[SearchNotesInput]):
    """Finds notes by their words."""

    name = "search_notes"
    description = (
        "Search the user's saved notes by key words and return the matching notes with "
        "their dates, best match first. Use to answer questions about something the user "
        "noted earlier."
    )
    Input = SearchNotesInput
    category = ToolCategory.PRODUCTIVITY

    async def run(self, params: SearchNotesInput, context: ToolContext) -> ToolResult:
        """Search and return the notes; speak the best one."""
        memory = require_memory(context, self.name)
        notes = await memory.search_notes(params.query, params.limit)
        if not notes:
            return ToolResult.success(
                f"I don't have any notes about {params.query}.", data={"notes": []}
            )
        rows: list[JsonValue] = [
            {
                "id": note.id,
                "title": note.title,
                "content": note.content,
                "created_at": note.created_at.isoformat(timespec="seconds"),
            }
            for note in notes
        ]
        best = truncate(notes[0].content, _SPOKEN_NOTE_CHARS)
        more = f" I found {len(notes)} notes in total." if len(notes) > 1 else ""
        return ToolResult.success(
            f"Your note says: {best}.{more}",
            detail="\n".join(
                f"[{n.id}] {n.created_at:%Y-%m-%d} {n.title + ': ' if n.title else ''}{n.content}"
                for n in notes
            ),
            data={"notes": rows},
        )
