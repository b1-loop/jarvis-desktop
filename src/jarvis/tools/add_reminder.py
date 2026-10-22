"""Tool: schedule a reminder for a specific time.

"Remind me to call mum at six", "in twenty minutes remind me to take the
pizza out". The LLM turns the natural phrase into an ISO 8601 datetime
(it has current_time for "now"); this tool validates it, interprets a
missing offset as the user's local time zone, and refuses times in the
past. The proactive layer speaks the reminder when it comes due.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Final

from pydantic import Field

from jarvis.agent.tool import Tool, ToolCategory, ToolContext, ToolInput, ToolResult
from jarvis.tools._common import require_memory

MAX_REMINDER_CHARS: Final = 500
MAX_DAYS_AHEAD: Final = 5 * 366
"""Anything further out is almost certainly a mis-parsed year."""
_NOON: Final = 12


class AddReminderInput(ToolInput):
    """Arguments for :class:`AddReminderTool`."""

    text: str = Field(
        min_length=1,
        max_length=MAX_REMINDER_CHARS,
        description="What to remind the user of, phrased to be spoken, e.g. 'Call mum'.",
    )
    due_at: datetime = Field(
        description=(
            "When, as an ISO 8601 datetime such as '2026-10-22T18:00:00+03:00'. Work it out "
            "from the user's phrase and the current time; without an offset it is taken as "
            "the user's local time."
        ),
    )


class AddReminderTool(Tool[AddReminderInput]):
    """Stores a reminder the proactive layer will speak when due."""

    name = "add_reminder"
    description = (
        "Set a reminder at a specific date and time (ISO 8601). Convert phrases like 'at "
        "six' or 'in 20 minutes' using the current time first. Times in the past are "
        "rejected. Jarvis says the reminder aloud when it is due."
    )
    Input = AddReminderInput
    category = ToolCategory.PRODUCTIVITY

    async def run(self, params: AddReminderInput, context: ToolContext) -> ToolResult:
        """Normalise the time, check it is in the future and store it."""
        now = context.clock.now()
        due = params.due_at if params.due_at.tzinfo else params.due_at.replace(tzinfo=now.tzinfo)
        if due <= now:
            return ToolResult.failure(
                "That time has already passed. When should I remind you?",
                detail=f"due_at {due.isoformat()} is not after now ({now.isoformat()})",
            )
        if due - now > timedelta(days=MAX_DAYS_AHEAD):
            return ToolResult.failure(
                "That's more than five years away; did I get the date right?",
                detail=f"due_at {due.isoformat()} is implausibly far ahead",
            )
        memory = require_memory(context, self.name)
        reminder = await memory.add_reminder(params.text, due)
        local = due.astimezone(now.tzinfo)
        return ToolResult.success(
            f"I'll remind you to {_lower_first(params.text)} {_when(local, now)}.",
            detail=f"Reminder {reminder.id} at {local.isoformat()}",
            data={"id": reminder.id, "text": params.text, "due_at": local.isoformat()},
        )


def _when(due: datetime, now: datetime) -> str:
    """Natural phrase for ``due`` relative to ``now`` (both in local time)."""
    clock = f"{due.hour % _NOON or _NOON}:{due:%M} {'AM' if due.hour < _NOON else 'PM'}"
    days = (due.date() - now.date()).days
    if days == 0:
        return f"at {clock}"
    if days == 1:
        return f"tomorrow at {clock}"
    return f"on {due:%A}, {due.day} {due:%B} at {clock}"


def _lower_first(text: str) -> str:
    return text[:1].lower() + text[1:]


# Design notes:
# - The LLM does the language work (it understands "next Friday after
#   lunch"); the tool does the checks the LLM is bad at: a real datetime,
#   not in the past, not decades away.
# - A naive datetime is the user's local time, because that is what they
#   said ("at six"); the clock's zone stands in for the user's.
# - "That time has already passed" usually means the model resolved "at six"
#   to this morning; asking back is better than silently picking tomorrow.
