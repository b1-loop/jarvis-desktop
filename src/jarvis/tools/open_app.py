"""Tool: launch an application, document or URL by name or path.

"Open Spotify", "start notepad", "open github.com". The OS resolves the
target the way the Run dialog does (App Paths, ``PATH``, file associations),
so the tool passes the name through unchanged instead of guessing install
locations.
"""

from __future__ import annotations

from typing import Final

from pydantic import Field

from jarvis.agent.tool import Tool, ToolCategory, ToolContext, ToolInput, ToolResult
from jarvis.tools._common import require_platform

MAX_TARGET_CHARS: Final = 500
MAX_ARGUMENT_CHARS: Final = 1000


class OpenAppInput(ToolInput):
    """Arguments for :class:`OpenAppTool`."""

    target: str = Field(
        min_length=1,
        max_length=MAX_TARGET_CHARS,
        description=(
            "What to open: an executable name ('notepad', 'code', 'spotify'), "
            "a full path to a program, or a URL."
        ),
    )
    arguments: str = Field(
        default="",
        max_length=MAX_ARGUMENT_CHARS,
        description="Optional command-line arguments for the program. Usually empty.",
    )


class OpenAppTool(Tool[OpenAppInput]):
    """Starts a program the user named."""

    name = "open_app"
    description = (
        "Launch an application, open a URL in the default browser or open a program by "
        "path. Use for 'open X', 'start X', 'launch X'. Returns the new process id "
        "(0 when an already-running app took the request). For documents use open_file."
    )
    Input = OpenAppInput
    category = ToolCategory.SYSTEM

    async def run(self, params: OpenAppInput, context: ToolContext) -> ToolResult:
        """Hand the target to the OS launcher."""
        platform = require_platform(context, self.name)
        pid = await platform.launch(params.target, params.arguments)
        return ToolResult.success(
            f"Opening {params.target}.",
            detail=f"Launched {params.target!r} {params.arguments!r} (pid {pid}).",
            data={"target": params.target, "arguments": params.arguments, "pid": pid},
        )


# Design notes:
# - No confirmation: starting a program the user just named is the request
#   itself and is undone by closing it. Closing, killing and typing are the
#   confirmable actions (close_app, window_action, type_text).
