"""Tool: minimise, maximise, restore or close one window.

The window is the focused one unless a title fragment is given, so "minimise
this" and "close the Notepad window" are both one call. Closing asks the
user first; resizing never does.
"""

from __future__ import annotations

from typing import Final, Literal

from pydantic import Field

from jarvis.agent.tool import Tool, ToolCategory, ToolContext, ToolInput, ToolResult
from jarvis.os_integration.base import Platform, WindowInfo
from jarvis.os_integration.errors import WindowNotFoundError
from jarvis.tools._common import require_platform

WindowActionName = Literal["minimize", "maximize", "restore", "close"]

MAX_TITLE_CHARS: Final = 200
_PAST_TENSE: Final[dict[str, str]] = {
    "minimize": "Minimised",
    "maximize": "Maximised",
    "restore": "Restored",
    "close": "Closed",
}


class WindowActionInput(ToolInput):
    """Arguments for :class:`WindowActionTool`."""

    action: WindowActionName = Field(
        description="What to do with the window: minimize, maximize, restore or close."
    )
    title: str | None = Field(
        default=None,
        min_length=1,
        max_length=MAX_TITLE_CHARS,
        description="Part of the window title. Omit to act on the window that has focus.",
    )


class WindowActionTool(Tool[WindowActionInput]):
    """Changes one window's state; closing is confirmed first."""

    name = "window_action"
    description = (
        "Minimize, maximize, restore or close a single window, either the focused one "
        "or the front-most one whose title contains the given text. Closing lets the "
        "app ask to save and needs the user's confirmation. To close every window of an "
        "application use close_app instead."
    )
    Input = WindowActionInput
    category = ToolCategory.SYSTEM
    keywords = ("minimera", "maximera", "fönster", "fönstret")

    def needs_confirmation(self, params: WindowActionInput, context: ToolContext) -> bool:
        """Only closing can lose anything; the size changes are trivially undone."""
        del context
        return params.action == "close"

    def confirmation_prompt(self, params: WindowActionInput) -> str:
        """Name the window that would be closed."""
        target = f"the {params.title} window" if params.title else "the current window"
        return f"Do you want me to close {target}?"

    async def run(self, params: WindowActionInput, context: ToolContext) -> ToolResult:
        """Resolve the window, apply the action and confirm what happened."""
        platform = require_platform(context, self.name)
        window = await _resolve(platform, params.title)
        if window is None:
            what = f"a window called {params.title}" if params.title else "a focused window"
            return ToolResult.failure(f"I can't find {what}.", detail="no matching window")
        try:
            await _apply(platform, params.action, window.handle)
        except WindowNotFoundError as error:
            return ToolResult.failure("That window has just closed.", detail=str(error))
        return ToolResult.success(
            f"{_PAST_TENSE[params.action]} {window.title}.",
            detail=f"{params.action} applied to {window.title!r} (handle {window.handle}).",
            data={"action": params.action, "handle": window.handle, "title": window.title},
        )


async def _resolve(platform: Platform, title: str | None) -> WindowInfo | None:
    """The focused window, or the front-most whose title contains ``title``."""
    if title is None:
        return await platform.active_window()
    needle = title.lower()
    for window in await platform.list_windows():
        if needle in window.title.lower():
            return window
    return None


async def _apply(platform: Platform, action: WindowActionName, handle: int) -> None:
    if action == "minimize":
        await platform.minimize_window(handle)
    elif action == "maximize":
        await platform.maximize_window(handle)
    elif action == "restore":
        await platform.restore_window(handle)
    else:
        await platform.close_window(handle)


# Design notes:
# - Confirmation is per call (needs_confirmation) rather than class-level,
#   so "minimise this" is instant while "close it" asks, which matches the
#   risk: only close can discard work.
# - The window is resolved from list_windows without focusing it, so acting
#   on a background window does not steal focus first.
