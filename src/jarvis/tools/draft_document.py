"""Tool: write a Markdown or Word document from content the LLM composed.

"Write up the meeting notes as a Word document", "turn Maya's email into a
project brief". The model writes the content (Markdown); this tool turns it
into a file in the configured output folder (``JARVIS_DOCUMENTS_OUTPUT_DIR``,
by default ``~/Documents/Jarvis``), optionally citing a source email. A new
file needs no confirmation; replacing an existing one does.

Google Docs output is out of scope: the Google sign-in (ADR-0009) has no
Drive or Docs scope, and adding one would widen the consent screen for
every Gmail user. A ``.docx`` uploads to Docs losslessly when wanted.
"""

from __future__ import annotations

import asyncio
import os
import re
import tempfile
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Final

from pydantic import Field

from jarvis.agent.tool import Tool, ToolCategory, ToolContext, ToolInput, ToolResult
from jarvis.integrations.base import EmailMessage
from jarvis.tools._integration_support import (
    Service,
    email_providers,
    not_configured,
    resolve_reference,
)

if TYPE_CHECKING:
    from docx.document import Document as DocxDocument
    from docx.text.paragraph import Paragraph

MAX_CONTENT_CHARS: Final = 200_000
_STEM_PATTERN: Final = r"^[A-Za-z0-9][A-Za-z0-9 _.\-]{0,99}$"
_SLUG_JUNK: Final = re.compile(r"[^A-Za-z0-9]+")
_HEADING: Final = re.compile(r"^(#{1,6})\s+(.+?)\s*#*$")
_BULLET: Final = re.compile(r"^\s*[-*+]\s+(.*)$")
_NUMBERED: Final = re.compile(r"^\s*\d+[.)]\s+(.*)$")
_BOLD_MARKER: Final = "**"
_FENCE: Final = "```"
_MAX_STEM_CHARS: Final = 80


class DocumentFormat(StrEnum):
    """Output file formats."""

    MARKDOWN = "markdown"
    DOCX = "docx"

    @property
    def suffix(self) -> str:
        """File extension including the dot."""
        return ".md" if self is DocumentFormat.MARKDOWN else ".docx"


class DraftDocumentInput(ToolInput):
    """Arguments for :class:`DraftDocumentTool`."""

    title: str = Field(min_length=1, max_length=200, description="Document title.")
    content: str = Field(
        min_length=1,
        max_length=MAX_CONTENT_CHARS,
        description="The full document body in Markdown: '#' headings, '-' or '1.' lists, "
        "**bold**, blank lines between paragraphs. Write it completely; nothing is added.",
    )
    format: DocumentFormat = Field(
        default=DocumentFormat.MARKDOWN, description="'markdown' (.md) or 'docx' (Word)."
    )
    filename: str | None = Field(
        default=None,
        pattern=_STEM_PATTERN,
        description="File name without extension or folder; omit to derive it from the title.",
    )
    source_email_id: str | None = Field(
        default=None,
        max_length=512,
        description="Id of an email (from list_unread_email / search_email) the document is "
        "based on; it is cited at the top.",
    )


class DraftDocumentTool(Tool[DraftDocumentInput]):
    """Writes LLM-composed content to a .md or .docx file."""

    name = "draft_document"
    description = (
        "Save a document you have written (Markdown content) as a Markdown or Word (.docx) "
        "file in the user's Jarvis documents folder, optionally citing a source email. Use "
        "for 'write this up as a document', 'draft a brief from that email'. Returns the "
        "file path. Asks before overwriting an existing file."
    )
    Input = DraftDocumentInput
    category = ToolCategory.FILES

    def needs_confirmation(self, params: DraftDocumentInput, context: ToolContext) -> bool:
        """Only overwriting needs the user's yes; a new file harms nothing."""
        return target_path(params, context.settings.documents_output_dir).exists()

    def confirmation_prompt(self, params: DraftDocumentInput) -> str:
        """Name the file that would be replaced."""
        name = f"{file_stem(params)}{params.format.suffix}"
        return f"{name} already exists in your documents folder. Do you want me to overwrite it?"

    async def run(self, params: DraftDocumentInput, context: ToolContext) -> ToolResult:
        """Resolve the optional source, render and write atomically."""
        source = None
        if params.source_email_id is not None:
            providers = email_providers(context)
            if not providers:
                return not_configured(Service.EMAIL)
            provider, message_id = resolve_reference(providers, params.source_email_id)
            source = await provider.get(message_id)
        path = target_path(params, context.settings.documents_output_dir)
        existed = await asyncio.to_thread(path.exists)
        size = await asyncio.to_thread(_write, path, params, source)
        return ToolResult.success(
            f"I've saved {path.name} in your documents folder.",
            detail=f"Wrote {size} bytes to {path}"
            + (" (replaced the old file)" if existed else ""),
            data={
                "path": str(path),
                "format": params.format.value,
                "bytes": size,
                "overwritten": existed,
            },
        )


def file_stem(params: DraftDocumentInput) -> str:
    """The explicit filename, or a slug of the title (``"Q3 plan!"`` -> ``"Q3-plan"``)."""
    if params.filename:
        return params.filename.strip()
    slug = _SLUG_JUNK.sub("-", params.title).strip("-")[:_MAX_STEM_CHARS].strip("-")
    return slug or "document"


def target_path(params: DraftDocumentInput, output_dir: Path) -> Path:
    """Where the document goes; the stem pattern keeps it inside ``output_dir``."""
    return output_dir / f"{file_stem(params)}{params.format.suffix}"


def _write(path: Path, params: DraftDocumentInput, source: EmailMessage | None) -> int:
    """Render to a temp file beside ``path`` and rename over it (blocking; worker thread)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temp_name = tempfile.mkstemp(dir=path.parent, prefix=".jarvis-", suffix=path.suffix)
    os.close(handle)
    temp = Path(temp_name)
    try:
        if params.format is DocumentFormat.DOCX:
            _render_docx(params, source).save(str(temp))
        else:
            temp.write_text(render_markdown(params, source), encoding="utf-8")
        temp.replace(path)
    except BaseException:
        temp.unlink(missing_ok=True)
        raise
    return path.stat().st_size


def _source_line(source: EmailMessage | None) -> str:
    if source is None:
        return ""
    return (
        f'Source: email from {source.sender}, "{source.subject}", '
        f"received {source.received_at:%Y-%m-%d %H:%M}"
    )


def render_markdown(params: DraftDocumentInput, source: EmailMessage | None) -> str:
    """The Markdown file: a title heading unless the content has one, the source, the body."""
    body = params.content.strip()
    parts = [] if body.startswith("# ") else [f"# {params.title}"]
    if line := _source_line(source):
        parts.append(f"> {line}")
    parts.append(body)
    return "\n\n".join(parts) + "\n"


def _render_docx(params: DraftDocumentInput, source: EmailMessage | None) -> DocxDocument:
    # Lazy: python-docx is in the optional ``documents`` extra.
    from docx import Document

    document = Document()
    document.core_properties.title = params.title
    document.core_properties.comments = _source_line(source)
    if not params.content.lstrip().startswith("# "):
        document.add_heading(params.title, level=1)
    if line := _source_line(source):
        document.add_paragraph().add_run(line).italic = True
    _add_markdown(document, params.content)
    return document


def _add_markdown(document: DocxDocument, markdown: str) -> None:
    """Map the Markdown subset the model is told to use onto Word styles."""
    paragraph: list[str] = []
    in_code = False
    for raw in [*markdown.splitlines(), ""]:
        line = raw.rstrip()
        if line.strip().startswith(_FENCE):
            in_code = not in_code
            continue
        block = None if in_code else _block_style(line)
        if in_code or block is not None or not line.strip():
            _flush(document, paragraph)
        if in_code:
            document.add_paragraph(line, style="No Spacing")
        elif block is not None:
            _add_block(document, *block)
        elif line.strip():
            paragraph.append(line.strip())


def _block_style(line: str) -> tuple[str, str] | None:
    """``(style, text)`` for headings and list items, ``None`` for body text."""
    if heading := _HEADING.match(line):
        return f"Heading {len(heading.group(1))}", heading.group(2)
    if bullet := _BULLET.match(line):
        return "List Bullet", bullet.group(1)
    if numbered := _NUMBERED.match(line):
        return "List Number", numbered.group(1)
    return None


def _add_block(document: DocxDocument, style: str, text: str) -> None:
    _add_runs(document.add_paragraph(style=style), text)


def _flush(document: DocxDocument, lines: list[str]) -> None:
    if lines:
        _add_runs(document.add_paragraph(), " ".join(lines))
        lines.clear()


def _add_runs(paragraph: Paragraph, text: str) -> None:
    """Add ``text`` with ``**bold**`` spans as bold runs."""
    for index, piece in enumerate(text.split(_BOLD_MARKER)):
        if piece:
            paragraph.add_run(piece).bold = index % 2 == 1


# Design notes:
# - Confirmation is per call: needs_confirmation stats the target (a single
#   metadata lookup, cheap enough for the event loop) so a fresh document is
#   written straight away and only a replacement is asked about. A file
#   that appears between the check and the write is replaced without asking;
#   for a folder only Jarvis writes to, that race is accepted.
# - Writing goes through a temp file and an atomic rename so an existing
#   document is never left half-written by a crash or a full disk.
# - The filename pattern (no separators, must start alphanumeric) is what
#   keeps every write inside the output folder; there is no path argument.
# - Only a Markdown subset is mapped to Word (headings, lists, bold, code
#   fences, paragraphs). The model is told exactly that subset, and anything
#   else survives as literal text rather than being silently dropped.
