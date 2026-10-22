"""Tool: open a file with its default application.

"Open the quarterly report" after find_files returned its path. The file is
handed to the OS launcher (the same as double-clicking it), so anything
that would *run* rather than *show* is refused: opening ``setup.exe`` or a
``.bat`` with its default handler is executing it, and executing arbitrary
paths is shell_command's job, under the sandbox policy.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Final

from pydantic import Field

from jarvis.agent.tool import Tool, ToolCategory, ToolContext, ToolInput, ToolResult
from jarvis.tools._common import require_platform

MAX_PATH_CHARS: Final = 1024

EXECUTABLE_SUFFIXES: Final = frozenset(
    {
        ".exe", ".com", ".bat", ".cmd", ".ps1", ".psm1", ".vbs", ".vbe", ".js", ".jse",
        ".wsf", ".wsh", ".msi", ".msp", ".scr", ".pif", ".lnk", ".url", ".reg", ".hta",
        ".cpl", ".jar", ".sh", ".appimage",
    }
)  # fmt: skip
"""Extensions whose default "open" action executes code; refused outright."""


class OpenFileInput(ToolInput):
    """Arguments for :class:`OpenFileTool`."""

    path: str = Field(
        min_length=1,
        max_length=MAX_PATH_CHARS,
        description="Full path of the file, as returned by find_files or recent_files.",
    )


class OpenFileTool(Tool[OpenFileInput]):
    """Shows a document in the application the user normally uses for it."""

    name = "open_file"
    description = (
        "Open an existing file (document, image, spreadsheet, ...) in its default "
        "application. Takes a full path; use find_files or recent_files to get one. "
        "Programs and scripts are refused; use open_app to start an application."
    )
    Input = OpenFileInput
    category = ToolCategory.FILES

    async def run(self, params: OpenFileInput, context: ToolContext) -> ToolResult:
        """Check the path, refuse executables, then launch."""
        path = Path(params.path)
        if path.suffix.lower() in EXECUTABLE_SUFFIXES:
            return ToolResult.failure(
                "I won't open that: it's a program or script, not a document.",
                detail=f"refused executable file type {path.suffix!r}: {path}",
            )
        if not await asyncio.to_thread(path.is_file):
            return ToolResult.failure(
                f"I can't find the file {path.name}.", detail=f"not a file: {path}"
            )
        platform = require_platform(context, self.name)
        await platform.launch(str(path))
        return ToolResult.success(
            f"Opening {path.name}.", detail=f"Opened {path}", data={"path": str(path)}
        )


# Design notes:
# - A denylist of executable extensions rather than an allowlist of document
#   types: users open many harmless formats (images, .xlsx, .pptx, .drawio),
#   while the set of extensions Windows *executes* on open is small and
#   well known. Shortcuts (.lnk, .url) are included because they can point
#   at an executable.
# - No confirmation for documents: showing a file changes nothing.
