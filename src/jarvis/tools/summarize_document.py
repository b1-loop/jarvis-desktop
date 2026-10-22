"""Tool: read a document's text so the LLM can summarise or answer questions about it.

"Summarise that PDF", "what does the contract say about notice periods".
The tool does the extraction (PDF, Word, Markdown, text, CSV) and trims the
text to a budget that keeps every heading; the summarising itself is the
LLM's next turn, so the tool never calls a model on its own.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

from pydantic import Field

from jarvis.agent.tool import Tool, ToolCategory, ToolContext, ToolInput, ToolResult
from jarvis.integrations.documents import DocumentError, extract_text, summarise_for_llm

MAX_PATH_CHARS: Final = 1024
DEFAULT_MAX_CHARS: Final = 8_000
MIN_MAX_CHARS: Final = 1_000
MAX_MAX_CHARS: Final = 40_000


class SummarizeDocumentInput(ToolInput):
    """Arguments for :class:`SummarizeDocumentTool`."""

    path: str = Field(
        min_length=1,
        max_length=MAX_PATH_CHARS,
        description="Full path of a PDF, .docx, .md, .txt or .csv file.",
    )
    max_chars: int = Field(
        default=DEFAULT_MAX_CHARS,
        ge=MIN_MAX_CHARS,
        le=MAX_MAX_CHARS,
        description=(
            "How much of the text to return. Raise it for a detailed question about a "
            "long document; the default suits a spoken summary."
        ),
    )


class SummarizeDocumentTool(Tool[SummarizeDocumentInput]):
    """Extracts a document's text, budgeted for the LLM to summarise."""

    name = "summarize_document"
    description = (
        "Extract the text of a PDF, Word, Markdown, text or CSV document, trimmed to a "
        "budget but keeping every heading, and return it so you can summarise it or "
        "answer questions about it. Get the path from find_files or recent_files first."
    )
    Input = SummarizeDocumentInput
    category = ToolCategory.FILES

    async def run(self, params: SummarizeDocumentInput, context: ToolContext) -> ToolResult:
        """Extract, budget and return the text as ``detail`` for the next LLM turn."""
        del context
        path = Path(params.path)
        try:
            document = await extract_text(path)
        except DocumentError as error:
            return ToolResult.failure(f"I couldn't read {path.name}.", detail=str(error))
        text = summarise_for_llm(document, params.max_chars)
        return ToolResult.success(
            f"I've read {path.name}.",
            detail=text,
            data={
                "path": str(path),
                "sections": len(document.sections),
                "characters": len(text),
                "truncated": document.truncated or len(text) < len(document.text),
            },
        )


# Design notes:
# - The text goes into ``detail`` (what the LLM sees next) and not into
#   ``data``: it can be tens of kilobytes, and the dashboard needs only the
#   counts.
# - The LLM, not this tool, writes the summary. That keeps the tool usable
#   in LOCAL mode, avoids a hidden second model call with its own cost, and
#   lets the same output answer a specific question instead of a summary.
