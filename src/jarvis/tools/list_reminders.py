"""Tool: list upcoming reminders.

"What reminders do I have?". Pending reminders by due time; already
delivered ones on request.
"""

from __future__ import annotations

from typing import Final

from pydantic import Field, JsonValue

from jarvis.agent.tool import Tool, ToolCategory, ToolContext, ToolInput, ToolResult
from jarvis.tools._common import require_memory

_SPOKEN_ITEMS: Final = 3


class ListRemindersInput(ToolInput):
    """Arguments for :class:`ListRemindersTool`."""

    include_fired: bool = Field(
        default=False, description="Also list reminders that were already delivered."
    )


class ListRemindersTool(Tool[ListRemindersInput]):
    """Reads out what Jarvis will remind the user of, and when."""

    name = "list_reminders"
    description = (
        "List the user's pending reminders (and past ones with include_fired), soonest "
        "first, with their due times in local time."
    )
    Input = ListRemindersInput
    category = ToolCategory.PRODUCTIVITY

    async def run(self, params: ListRemindersInput, context: ToolContext) -> ToolResult:
        """Fetch, convert to local time and read out the first few."""
        memory = require_memory(context, self.name)
        reminders = await memory.list_reminders(include_fired=params.include_fired)
        if not reminders:
            return ToolResult.success("You have no reminders.", data={"reminders": []})
        zone = context.clock.now().tzinfo
        local = [(r, r.due_at.astimezone(zone)) for r in reminders]
        spoken_items = "; ".join(
            f"{r.text} on {due:%a %d %b} at {due:%H:%M}" for r, due in local[:_SPOKEN_ITEMS]
        )
        more = len(local) - min(len(local), _SPOKEN_ITEMS)
        rows: list[JsonValue] = [
            {"id": r.id, "text": r.text, "due_at": due.isoformat(), "fired": r.fired}
            for r, due in local
        ]
        return ToolResult.success(
            f"{spoken_items}" + (f"; and {more} more." if more else "."),
            detail="\n".join(f"[{r.id}] {due.isoformat()} {r.text}" for r, due in local),
            data={"reminders": rows},
        )
