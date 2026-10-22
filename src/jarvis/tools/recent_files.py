"""Tool: list the most recently changed documents, and optionally open the newest.

Backs "open the last thing I worked on": one call finds the newest document
under the configured roots and hands it to the default application through
the Platform launcher.
"""

from __future__ import annotations

from typing import Final

from pydantic import Field, JsonValue

from jarvis.agent.tool import Tool, ToolCategory, ToolContext, ToolInput, ToolResult
from jarvis.integrations.documents import DocumentMatch, recent_documents
from jarvis.tools._common import document_row, require_platform

DEFAULT_LIMIT: Final = 5
MAX_LIMIT: Final = 20


class RecentFilesInput(ToolInput):
    """Arguments for :class:`RecentFilesTool`."""

    limit: int = Field(
        default=DEFAULT_LIMIT, ge=1, le=MAX_LIMIT, description="How many recent files to list."
    )
    open_most_recent: bool = Field(
        default=False,
        description=(
            "Open the newest file in its default application. Set for 'open the last "
            "thing I worked on'."
        ),
    )


class RecentFilesTool(Tool[RecentFilesInput]):
    """Answers "what was I working on" and reopens it."""

    name = "recent_files"
    description = (
        "List the most recently modified documents in the user's Documents, Desktop and "
        "Downloads folders, newest first, and optionally open the newest one. Use for "
        "'open the last thing I worked on' or 'what did I edit recently'."
    )
    Input = RecentFilesInput
    category = ToolCategory.FILES
    keywords = ("senaste", "fil", "filer", "dokument")

    async def run(self, params: RecentFilesInput, context: ToolContext) -> ToolResult:
        """Find recent documents and open the newest if asked."""
        matches = await recent_documents(context.settings.document_roots, limit=params.limit)
        if not matches:
            return ToolResult.success(
                "I couldn't find any recent documents.", data={"matches": [], "opened": None}
            )
        newest = matches[0]
        if params.open_most_recent:
            platform = require_platform(context, self.name)
            await platform.launch(str(newest.path))
        return _result(matches, opened=newest if params.open_most_recent else None)


def _result(matches: list[DocumentMatch], *, opened: DocumentMatch | None) -> ToolResult:
    newest = matches[0]
    if opened is not None:
        spoken = f"Opening {opened.path.name}."
    else:
        spoken = f"The most recent is {newest.path.name}, changed {newest.modified:%d %B}."
    rows: list[JsonValue] = [document_row(match) for match in matches]
    detail = "\n".join(f"- {m.path} (modified {m.modified:%Y-%m-%d %H:%M} UTC)" for m in matches)
    return ToolResult.success(
        spoken,
        detail=detail,
        data={"matches": rows, "opened": str(opened.path) if opened else None},
    )


# Design notes:
# - Opening needs no confirmation: it only shows a document the user owns in
#   its usual application. The extension filter of recent_documents (PDF,
#   Word, text, Markdown, CSV) means this can never launch an executable.
# - The platform is required only when opening, so listing still works in
#   text-only mode without an OS backend.
