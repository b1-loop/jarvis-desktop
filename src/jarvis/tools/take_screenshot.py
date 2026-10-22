"""Tool: capture the screen (or one window) to a PNG file and return its path.

Screenshots are saved under ``<data_dir>/screenshots`` with a timestamped
name, so "take a screenshot" leaves something the user can find later and
the vision tool (a separate plugin) can be pointed at the path.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, Final

from pydantic import Field

from jarvis.agent.tool import Tool, ToolCategory, ToolContext, ToolInput, ToolResult
from jarvis.os_integration.errors import WindowNotFoundError
from jarvis.tools._common import require_platform

if TYPE_CHECKING:
    from datetime import datetime

    from jarvis.os_integration.base import Platform

SCREENSHOT_DIRECTORY: Final = "screenshots"
_FILENAME_FORMAT: Final = "screenshot-%Y%m%d-%H%M%S"


class TakeScreenshotInput(ToolInput):
    """Arguments for :class:`TakeScreenshotTool`."""

    all_monitors: bool = Field(
        default=False,
        description="Capture the whole desktop across all monitors instead of the primary one.",
    )
    active_window_only: bool = Field(
        default=False,
        description="Capture only the focused window. Takes precedence over all_monitors.",
    )


class TakeScreenshotTool(Tool[TakeScreenshotInput]):
    """Saves a screenshot and reports where it went."""

    name = "take_screenshot"
    description = (
        "Take a screenshot of the primary monitor, all monitors, or only the focused "
        "window, and save it as a PNG. Returns the saved file path. This does not "
        "describe the image; use the screen-understanding tool for that."
    )
    Input = TakeScreenshotInput
    category = ToolCategory.SYSTEM

    async def run(self, params: TakeScreenshotInput, context: ToolContext) -> ToolResult:
        """Capture, then write the PNG off the event loop."""
        platform = require_platform(context, self.name)
        try:
            png = await _capture(platform, params)
        except WindowNotFoundError as error:
            return ToolResult.failure("There's no focused window to capture.", detail=str(error))
        directory = context.settings.data_dir / SCREENSHOT_DIRECTORY
        path = await asyncio.to_thread(_save, directory, context.clock.now(), png)
        return ToolResult.success(
            f"Screenshot saved as {path.name}.",
            detail=f"Saved {len(png)} bytes to {path}",
            data={"path": str(path), "bytes": len(png)},
        )


async def _capture(platform: Platform, params: TakeScreenshotInput) -> bytes:
    if not params.active_window_only:
        return await platform.screenshot(all_monitors=params.all_monitors)
    window = await platform.active_window()
    if window is None:
        raise WindowNotFoundError("no window has focus")
    return await platform.screenshot_window(window.handle)


def _save(directory: Path, taken_at: datetime, png: bytes) -> Path:
    """Write ``png`` under a unique timestamped name; never overwrites."""
    directory.mkdir(parents=True, exist_ok=True)
    stem = taken_at.strftime(_FILENAME_FORMAT)
    path = directory / f"{stem}.png"
    counter = 1
    while path.exists():
        path = directory / f"{stem}-{counter}.png"
        counter += 1
    path.write_bytes(png)
    return path.resolve()


# Design notes:
# - No confirmation: capturing writes a new file into Jarvis's own data
#   directory and never overwrites one; it does not change anything the
#   user already has.
# - Two screenshots within one second get a numeric suffix rather than
#   silently replacing the first.
