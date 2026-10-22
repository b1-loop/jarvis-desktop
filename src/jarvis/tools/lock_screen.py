"""Tool: lock the workstation.

Locking interrupts whatever the user is doing and needs their password to
undo, and a misheard "lock" mid-dictation would be unwelcome, so the tool
asks before acting.
"""

from __future__ import annotations

from jarvis.agent.tool import Tool, ToolCategory, ToolContext, ToolInput, ToolResult
from jarvis.tools._common import require_platform


class LockScreenInput(ToolInput):
    """No arguments."""


class LockScreenTool(Tool[LockScreenInput]):
    """Locks the computer after confirmation."""

    name = "lock_screen"
    description = "Lock the computer (the user must sign in again). Use for 'lock my PC'."
    Input = LockScreenInput
    category = ToolCategory.SYSTEM
    keywords = ("computer", "pc", "laptop")
    requires_confirmation = True

    def confirmation_prompt(self, params: LockScreenInput) -> str:
        """A plain question; there are no arguments to quote."""
        del params
        return "Do you want me to lock the computer?"

    async def run(self, params: LockScreenInput, context: ToolContext) -> ToolResult:
        """Lock; the spoken line is short because the user is about to leave."""
        del params
        platform = require_platform(context, self.name)
        await platform.lock_screen()
        return ToolResult.success("Locking now.", data={"locked": True})
