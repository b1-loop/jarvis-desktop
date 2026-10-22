"""Tool: list the visible top-level windows, front-most first.

The LLM calls this before switch_window, window_action or close_app when it
needs the exact title or executable name the user only paraphrased ("the
Chrome window with my email").
"""

from __future__ import annotations

from typing import Final

from pydantic import Field, JsonValue

from jarvis.agent.tool import Tool, ToolCategory, ToolContext, ToolInput, ToolResult
from jarvis.os_integration.base import WindowInfo
from jarvis.tools._common import require_platform

MAX_FILTER_CHARS: Final = 200
DEFAULT_LIMIT: Final = 20
MAX_LIMIT: Final = 50
_SPOKEN_TITLES: Final = 3


class ListWindowsInput(ToolInput):
    """Arguments for :class:`ListWindowsTool`."""

    contains: str | None = Field(
        default=None,
        max_length=MAX_FILTER_CHARS,
        description="Only windows whose title or executable contains this text (any case).",
    )
    limit: int = Field(
        default=DEFAULT_LIMIT, ge=1, le=MAX_LIMIT, description="Maximum windows to return."
    )


class ListWindowsTool(Tool[ListWindowsInput]):
    """Answers "what's open" and resolves paraphrased window names."""

    name = "list_windows"
    description = (
        "List open windows (title, executable name, process id), front-most first. "
        "Use to answer 'what is open' or to find the exact title or executable name "
        "before switch_window, window_action or close_app."
    )
    Input = ListWindowsInput
    category = ToolCategory.SYSTEM

    async def run(self, params: ListWindowsInput, context: ToolContext) -> ToolResult:
        """Query, filter and render the window list."""
        platform = require_platform(context, self.name)
        windows = [w for w in await platform.list_windows() if _matches(w, params.contains)]
        shown = windows[: params.limit]
        if not shown:
            return ToolResult.success("I don't see any matching windows.", data={"windows": []})
        rows: list[JsonValue] = [_row(window) for window in shown]
        titles = ", ".join(window.title for window in shown[:_SPOKEN_TITLES])
        more = len(windows) - min(len(windows), _SPOKEN_TITLES)
        spoken = f"{len(windows)} windows: {titles}" + (f", and {more} more." if more else ".")
        detail = "\n".join(
            f"- {w.title} [{w.process_name}, pid {w.pid}, handle {w.handle}]" for w in shown
        )
        return ToolResult.success(spoken, detail=detail, data={"windows": rows})


def _matches(window: WindowInfo, needle: str | None) -> bool:
    if not needle:
        return True
    lowered = needle.lower()
    return lowered in window.title.lower() or lowered in window.process_name.lower()


def _row(window: WindowInfo) -> dict[str, JsonValue]:
    return {
        "handle": window.handle,
        "title": window.title,
        "process_name": window.process_name,
        "pid": window.pid,
    }
