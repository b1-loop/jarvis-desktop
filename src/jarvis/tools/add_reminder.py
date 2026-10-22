"""Tool: schedule a reminder, either some minutes from now or at a given time.

"Remind me in five minutes to call mum", "remind me at six to take the
pizza out". A relative phrase arrives as ``in_minutes`` and is resolved
here against the injected clock; an absolute one arrives as an ISO 8601
``due_at`` that the LLM worked out (the system prompt states the current
moment). Either way the tool interprets a missing offset as the user's
local time zone and refuses times in the past or implausibly far ahead,
with an error that tells the model how to correct itself. The proactive
layer speaks the reminder when it comes due.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Final, Self

from pydantic import Field, model_validator

from jarvis.agent.tool import Tool, ToolCategory, ToolContext, ToolInput, ToolResult
from jarvis.tools._common import require_memory

MAX_REMINDER_CHARS: Final = 500
MAX_DAYS_AHEAD: Final = 5 * 366
"""Anything further out is almost certainly a mis-parsed year."""
_MINUTES_PER_DAY: Final = 24 * 60
MAX_MINUTES_AHEAD: Final = MAX_DAYS_AHEAD * _MINUTES_PER_DAY
_NOON: Final = 12
ONE_TIME_REQUIRED: Final = (
    "give exactly one of in_minutes (for 'in five minutes') or due_at (for a clock time)"
)


class AddReminderInput(ToolInput):
    """Arguments for :class:`AddReminderTool`; exactly one of the two times is given."""

    text: str = Field(
        min_length=1,
        max_length=MAX_REMINDER_CHARS,
        description="What to remind the user of, phrased to be spoken, e.g. 'Call mum'.",
    )
    in_minutes: int | None = Field(
        default=None,
        ge=1,
        le=MAX_MINUTES_AHEAD,
        description=(
            "For a time relative to now: minutes from now, e.g. 5 for 'in five minutes' or "
            "120 for 'in two hours'. Leave due_at out when you use this."
        ),
    )
    due_at: datetime | None = Field(
        default=None,
        description=(
            "For a clock time or date: ISO 8601 such as '2026-10-22T18:00:00+02:00', worked "
            "out from the current date and time; without an offset it is the user's local "
            "time. Leave in_minutes out when you use this."
        ),
    )

    @model_validator(mode="after")
    def _exactly_one_time(self) -> Self:
        if (self.in_minutes is None) == (self.due_at is None):
            raise ValueError(ONE_TIME_REQUIRED)
        return self


class AddReminderTool(Tool[AddReminderInput]):
    """Stores a reminder the proactive layer will speak when due."""

    name = "add_reminder"
    description = (
        "Set a reminder. For 'in 5 minutes' or 'in two hours' pass in_minutes; for a clock "
        "time or a date pass due_at (ISO 8601, from the current date and time). Times in "
        "the past are rejected. Jarvis says the reminder aloud when it is due."
    )
    Input = AddReminderInput
    category = ToolCategory.PRODUCTIVITY
    keywords = ("remind", "påminn", "påminnelse")

    async def run(self, params: AddReminderInput, context: ToolContext) -> ToolResult:
        """Resolve the time, check it is plausible and store it."""
        now = context.clock.now()
        due = resolve_due(params, now)
        if (problem := implausible(due, now)) is not None:
            return problem
        memory = require_memory(context, self.name)
        reminder = await memory.add_reminder(params.text, due)
        local = due.astimezone(now.tzinfo)
        return ToolResult.success(
            f"I'll remind you to {_lower_first(params.text)} {_when(local, now)}.",
            detail=f"Reminder {reminder.id} at {local.isoformat()}",
            data={"id": reminder.id, "text": params.text, "due_at": local.isoformat()},
        )


def resolve_due(params: AddReminderInput, now: datetime) -> datetime:
    """The absolute, timezone-aware due time the arguments describe."""
    if params.in_minutes is not None:
        return now + timedelta(minutes=params.in_minutes)
    due = params.due_at
    if due is None:  # the validator guarantees one of the two
        raise ValueError(ONE_TIME_REQUIRED)
    return due if due.tzinfo else due.replace(tzinfo=now.tzinfo)


def implausible(due: datetime, now: datetime) -> ToolResult | None:
    """A failed result for a due time in the past or years away; ``None`` when fine.

    The detail (what the model reads) states the current time and the
    relative alternative, so the model can retry instead of giving up.
    """
    current = f"The current local time is {now.isoformat(timespec='seconds')}."
    if due <= now:
        wrong_year = f" The year {due.year} is wrong." if due.year != now.year else ""
        return ToolResult.failure(
            "That time has already passed. When should I remind you?",
            detail=(
                f"due_at {due.isoformat()} is not in the future.{wrong_year} {current} For a "
                "relative time such as 'in five minutes' pass in_minutes instead of due_at."
            ),
        )
    if due - now > timedelta(days=MAX_DAYS_AHEAD):
        return ToolResult.failure(
            "That's more than five years away; did I get the date right?",
            detail=f"due_at {due.isoformat()} is implausibly far ahead. {current}",
        )
    return None


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
# - Two ways to say when, because they fail differently. Small models
#   (qwen2.5:7b, qwen3:8b) turned "om fem minuter" into a 2023 timestamp
#   instead of calling current_time first; "in_minutes": 5 needs no clock
#   arithmetic from the model at all, and the tool adds it to the injected
#   clock. Absolute times still need the model's language work ("next
#   Friday after lunch"), helped by the current moment in the system prompt.
# - Exactly one of the two is accepted (pydantic validator): with both, one
#   of them would silently win and the other may be what the user meant.
# - Implausible times are refused, never stored: a reminder that fires at
#   once (past) or never (decades ahead) is worse than a question back. The
#   detail names the current time and the in_minutes alternative because
#   that is the information the model lacked when it got the time wrong.
# - A naive datetime is the user's local time, because that is what they
#   said ("at six"); the clock's zone stands in for the user's.
# - "That time has already passed" usually means the model resolved "at six"
#   to this morning; asking back is better than silently picking tomorrow.
