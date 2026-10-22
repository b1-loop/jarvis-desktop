"""Tool: close every running instance of an application, gracefully or by force.

A graceful close asks each window to close, so the application can still
prompt to save. Force terminates the process and loses unsaved work, which
is why it is a separate, explicit flag with a sharper confirmation prompt.
"""

from __future__ import annotations

from typing import Final

from pydantic import Field

from jarvis.agent.tool import Tool, ToolCategory, ToolContext, ToolInput, ToolResult
from jarvis.os_integration.errors import PlatformError, ProcessNotFoundError
from jarvis.tools._common import require_platform

MAX_NAME_CHARS: Final = 260


class CloseAppInput(ToolInput):
    """Arguments for :class:`CloseAppTool`."""

    name: str = Field(
        min_length=1,
        max_length=MAX_NAME_CHARS,
        description=(
            "Executable name of the application as list_windows reports it, "
            "for example 'notepad.exe' or 'Spotify.exe'."
        ),
    )
    force: bool = Field(
        default=False,
        description=(
            "Terminate immediately, discarding unsaved work. Only set this when the "
            "user explicitly asks to force-quit or kill the app."
        ),
    )


class CloseAppTool(Tool[CloseAppInput]):
    """Closes an application after the user confirmed."""

    name = "close_app"
    description = (
        "Close all windows of a running application by its executable name "
        "(e.g. 'notepad.exe'). Graceful by default so the app can ask to save; "
        "set force only when the user says force-quit or kill. Call list_windows "
        "first if you do not know the exact executable name."
    )
    Input = CloseAppInput
    category = ToolCategory.SYSTEM
    requires_confirmation = True
    keywords = ("quit", "stäng", "stänga", "avsluta")

    def confirmation_prompt(self, params: CloseAppInput) -> str:
        """Make the cost of forcing explicit; a plain close gets a plain question."""
        if params.force:
            return (
                f"Force-quitting {params.name} will lose any unsaved work. "
                "Do you really want me to kill it?"
            )
        return f"Do you want me to close {params.name}?"

    async def run(self, params: CloseAppInput, context: ToolContext) -> ToolResult:
        """Close by name and phrase "not running" and "refused" precisely."""
        platform = require_platform(context, self.name)
        try:
            count = await platform.close_process_by_name(params.name, force=params.force)
        except ProcessNotFoundError as error:
            return ToolResult.failure(f"{params.name} isn't running.", detail=str(error))
        except PlatformError as error:
            if params.force:
                raise
            return ToolResult.failure(
                f"{params.name} wouldn't close normally. I can force it if you want.",
                detail=f"graceful close failed: {error}",
            )
        verb = "Killed" if params.force else "Closed"
        return ToolResult.success(
            f"{verb} {params.name}.",
            detail=f"{verb} {count} process(es) named {params.name!r}.",
            data={"name": params.name, "force": params.force, "count": count},
        )


# Design notes:
# - A failed graceful close is reported with an offer to force, not
#   escalated automatically: forcing is a second decision the user makes
#   and confirms again through the stronger prompt.
# - A failed forced close is re-raised so Tool.execute reports the platform
#   error verbatim; there is nothing further to offer.
