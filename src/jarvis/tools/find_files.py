"""Tool: find documents by name, and optionally by content, in the user's folders.

"Find my notes on the hiring plan", "where's the invoice from March". The
search walks ``Settings.document_roots`` (Documents, Desktop and Downloads by
default) through ``jarvis.integrations.documents`` and ranks by
modification time, because the file meant is almost always a recent one.
"""

from __future__ import annotations

from datetime import timedelta
from typing import TYPE_CHECKING, Final

from pydantic import Field, JsonValue

from jarvis.agent.tool import Tool, ToolCategory, ToolContext, ToolInput, ToolResult
from jarvis.integrations.documents import find_documents
from jarvis.tools._common import document_row

if TYPE_CHECKING:
    from jarvis.integrations.documents import DocumentMatch

MAX_QUERY_CHARS: Final = 200
DEFAULT_LIMIT: Final = 5
MAX_LIMIT: Final = 25
MAX_DAYS: Final = 3650
_SPOKEN_NAMES: Final = 3


class FindFilesInput(ToolInput):
    """Arguments for :class:`FindFilesTool`."""

    query: str = Field(
        min_length=1,
        max_length=MAX_QUERY_CHARS,
        description=(
            "Words that must all appear in the file name (any case), e.g. 'hiring plan'. "
            "Keep it to the distinctive words; drop filler like 'my' or 'the file'."
        ),
    )
    search_content: bool = Field(
        default=False,
        description=(
            "Also look inside the most recent documents (PDF, Word, text, Markdown, CSV) "
            "for the words. Slower; use when the user describes what the file says."
        ),
    )
    modified_within_days: int | None = Field(
        default=None,
        ge=1,
        le=MAX_DAYS,
        description="Only files changed in the last N days, e.g. 7 for 'from last week'.",
    )
    limit: int = Field(default=DEFAULT_LIMIT, ge=1, le=MAX_LIMIT, description="Maximum results.")


class FindFilesTool(Tool[FindFilesInput]):
    """Searches the configured document folders."""

    name = "find_files"
    description = (
        "Find documents (PDF, Word, text, Markdown, CSV) in the user's Documents, Desktop "
        "and Downloads folders whose name contains every query word, optionally also "
        "searching inside recent files. Returns full paths, newest first. Follow up with "
        "open_file or summarize_document using a returned path."
    )
    Input = FindFilesInput
    category = ToolCategory.FILES

    async def run(self, params: FindFilesInput, context: ToolContext) -> ToolResult:
        """Search the roots and phrase the matches."""
        days = params.modified_within_days
        matches = await find_documents(
            context.settings.document_roots,
            params.query,
            limit=params.limit,
            modified_within=timedelta(days=days) if days else None,
            search_content=params.search_content,
        )
        if not matches:
            return ToolResult.success(
                f"I couldn't find any documents matching {params.query}.",
                detail=f"No matches under {_roots(context)}.",
                data={"matches": []},
            )
        return _found(matches)


def _found(matches: list[DocumentMatch]) -> ToolResult:
    names = ", ".join(match.path.name for match in matches[:_SPOKEN_NAMES])
    count = "one document" if len(matches) == 1 else f"{len(matches)} documents"
    rows: list[JsonValue] = [document_row(match) for match in matches]
    detail = "\n".join(
        f"- {match.path} (modified {match.modified:%Y-%m-%d %H:%M} UTC, {match.size_bytes} bytes)"
        for match in matches
    )
    return ToolResult.success(f"I found {count}: {names}.", detail=detail, data={"matches": rows})


def _roots(context: ToolContext) -> str:
    return ", ".join(str(root) for root in context.settings.document_roots)


# Design notes:
# - No confirmation: searching only reads metadata (and, for content search,
#   a bounded number of recent files) inside folders the user configured.
# - The query words are AND-ed by find_documents; the field description
#   tells the LLM to drop filler words, because "my notes on hiring" would
#   otherwise require "my" in the file name.
