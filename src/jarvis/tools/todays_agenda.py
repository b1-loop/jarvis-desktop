"""Tool: the user's agenda for today (or another day close to it).

"What's on today?" is the most common calendar question and feeds the
morning briefing. The day is the user's local day, taken from the injected
clock's zone, so a meeting at 00:30 local time is not filed under
yesterday just because the provider speaks UTC.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Final

from pydantic import Field, JsonValue

from jarvis.agent.tool import Tool, ToolCategory, ToolContext, ToolInput, ToolResult
from jarvis.integrations.base import CalendarEvent, CalendarProvider, TimeRange
from jarvis.tools._integration_support import (
    Service,
    calendar_providers,
    count_phrase,
    event_item,
    event_line,
    flatten,
    gather_each,
    not_configured,
    spoken_clock,
    spoken_day,
    spoken_list,
)

MAX_DAYS_AHEAD: Final = 14
_SPOKEN_EVENTS: Final = 4


class TodaysAgendaInput(ToolInput):
    """Arguments for :class:`TodaysAgendaTool`."""

    days_from_today: int = Field(
        default=0,
        ge=0,
        le=MAX_DAYS_AHEAD,
        description="0 for today, 1 for tomorrow, and so on up to two weeks ahead.",
    )


class TodaysAgendaTool(Tool[TodaysAgendaInput]):
    """Lists the events of one local calendar day across all calendars."""

    name = "todays_agenda"
    description = (
        "Get the user's calendar for today (or, with days_from_today, tomorrow or another "
        "day up to two weeks ahead) across all connected calendars. Use for 'what's on "
        "today?' or 'am I busy tomorrow?'. Returns events with times, locations and attendees."
    )
    Input = TodaysAgendaInput
    category = ToolCategory.PRODUCTIVITY

    async def run(self, params: TodaysAgendaInput, context: ToolContext) -> ToolResult:
        """Query each calendar for the local day and merge by start time."""
        providers = calendar_providers(context)
        if not providers:
            return not_configured(Service.CALENDAR)
        now = context.clock.now()
        day = _local_day(now, params.days_from_today)
        gathered = await gather_each(providers, lambda p: p.agenda(day))
        pairs = sorted(flatten(gathered), key=lambda pair: pair[1].start)
        items: list[JsonValue] = [event_item(provider, event) for provider, event in pairs]
        lines = [f"Agenda for {day.start:%A %d %B %Y}:"]
        lines.extend(event_line(event, now) for _, event in pairs)
        lines.append(gathered.failure_note())
        return ToolResult.success(
            _spoken(pairs, spoken_day(day.start, now), now),
            detail="\n".join(lines).strip(),
            data={
                "date": day.start.date().isoformat(),
                "events": items,
                "failures": gathered.failure_data(),
            },
        )


def _local_day(now: datetime, days_ahead: int) -> TimeRange:
    start = now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=days_ahead)
    return TimeRange(start, start + timedelta(days=1))


def _spoken(pairs: list[tuple[CalendarProvider, CalendarEvent]], when: str, now: datetime) -> str:
    if not pairs:
        return f"Your calendar is clear {when}."
    mentions = [
        event.title if event.all_day else f"{event.title} at {spoken_clock(event.start, now)}"
        for _, event in pairs[:_SPOKEN_EVENTS]
    ]
    more = len(pairs) - len(mentions)
    tail = f", and {more} more" if more else ""
    return f"You have {count_phrase(len(pairs), 'event')} {when}: {spoken_list(mentions)}{tail}."


# Design notes:
# - ``agenda`` is used directly instead of CalendarProvider.today so the day
#   boundary is the user's local midnight (from the clock), not whatever
#   zone a provider's ``now`` default would pick.
# - All-day events are listed (birthdays matter in a briefing) but spoken
#   without a time.
