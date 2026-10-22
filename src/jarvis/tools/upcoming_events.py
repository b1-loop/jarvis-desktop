"""Tool: calendar events starting soon, plus the same logic for the meeting trigger.

"What's next?" and the proactive "your meeting starts in ten minutes" ask
the same question of the calendar, so the answer is computed once, in
:func:`find_upcoming`. The tool wraps it for the LLM and
:class:`CalendarTriggerSource` adapts it to the
:class:`jarvis.proactive.triggers.CalendarSource` Protocol so the
composition root can hand it to ``UpcomingMeetingTrigger`` directly.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timedelta
from typing import Final

from pydantic import Field, JsonValue

from jarvis.agent.tool import Tool, ToolCategory, ToolContext, ToolInput, ToolResult
from jarvis.integrations.base import CalendarEvent, CalendarProvider
from jarvis.integrations.services import make_reference
from jarvis.proactive import triggers
from jarvis.tools._integration_support import (
    Gathered,
    Service,
    calendar_providers,
    event_item,
    event_line,
    flatten,
    gather_each,
    not_configured,
    spoken_minutes,
    spoken_when,
)
from jarvis.utils.timing import Clock, SystemClock

MAX_WITHIN_MINUTES: Final = 7 * 24 * 60
DEFAULT_WITHIN_MINUTES: Final = 240


async def find_upcoming(
    providers: Sequence[CalendarProvider], within: timedelta, now: datetime
) -> Gathered[CalendarProvider, list[CalendarEvent]]:
    """Events starting in ``[now, now + within)`` on every calendar.

    Shared by the tool and the proactive trigger so both agree on what
    "upcoming" means. Partial failures are kept in the result (see
    ``gather_each``); if every calendar fails, the first error is raised.
    """
    return await gather_each(providers, lambda p: p.upcoming(within, now))


def in_start_order(
    gathered: Gathered[CalendarProvider, list[CalendarEvent]],
) -> list[tuple[CalendarProvider, CalendarEvent]]:
    """Flatten a :func:`find_upcoming` result, earliest start first."""
    return sorted(flatten(gathered), key=lambda pair: pair[1].start)


class CalendarTriggerSource:
    """Adapts :func:`find_upcoming` to ``proactive.triggers.CalendarSource``.

    All-day events are left out: a birthday "starting" at midnight should
    not produce a "starts in ten minutes" warning at 23:50.

    Args:
        providers: The calendars to watch.
        clock: Time source; share it with the scheduler.
    """

    def __init__(self, providers: Sequence[CalendarProvider], clock: Clock | None = None) -> None:
        self._providers = tuple(providers)
        self._clock: Clock = clock or SystemClock()

    async def upcoming_events(self, within: timedelta) -> list[triggers.CalendarEvent]:
        """Timed events starting within ``within`` of now, earliest first."""
        gathered = await find_upcoming(self._providers, within, self._clock.now())
        return [
            triggers.CalendarEvent(
                id=make_reference(provider.provider_name, event.id),
                title=event.title,
                start=event.start,
                location=event.location or None,
            )
            for provider, event in in_start_order(gathered)
            if not event.all_day
        ]


class UpcomingEventsInput(ToolInput):
    """Arguments for :class:`UpcomingEventsTool`."""

    within_minutes: int = Field(
        default=DEFAULT_WITHIN_MINUTES,
        ge=1,
        le=MAX_WITHIN_MINUTES,
        description="How far ahead to look, in minutes (default four hours, at most a week).",
    )


class UpcomingEventsTool(Tool[UpcomingEventsInput]):
    """Lists events starting soon, earliest first."""

    name = "upcoming_events"
    description = (
        "List calendar events that start within the next N minutes (default 4 hours), "
        "earliest first. Use for 'what's my next meeting?' or 'anything in the next hour?'. "
        "Returns titles, start and end times, locations and attendees."
    )
    Input = UpcomingEventsInput
    category = ToolCategory.PRODUCTIVITY
    keywords = ("meeting", "next", "möte", "möten", "nästa")

    async def run(self, params: UpcomingEventsInput, context: ToolContext) -> ToolResult:
        """Run the shared query and describe the next event first."""
        providers = calendar_providers(context)
        if not providers:
            return not_configured(Service.CALENDAR)
        now = context.clock.now()
        within = timedelta(minutes=params.within_minutes)
        gathered = await find_upcoming(providers, within, now)
        pairs = in_start_order(gathered)
        items: list[JsonValue] = [event_item(provider, event) for provider, event in pairs]
        lines = [f"Events starting in the next {params.within_minutes} minutes:"]
        lines.extend(event_line(event, now) for _, event in pairs)
        lines.append(gathered.failure_note())
        return ToolResult.success(
            _spoken(pairs, within, now),
            detail="\n".join(lines).strip(),
            data={"events": items, "failures": gathered.failure_data()},
        )


def _spoken(
    pairs: list[tuple[CalendarProvider, CalendarEvent]], within: timedelta, now: datetime
) -> str:
    if not pairs:
        return f"Nothing on your calendar in the next {spoken_minutes(within)}."
    event = pairs[0][1]
    where = f" in {event.location}" if event.location else ""
    following = f" After that, {len(pairs) - 1} more." if len(pairs) > 1 else ""
    return f"Next up: {event.title} {spoken_when(event.start, now)}{where}.{following}"


# Design notes:
# - The trigger adapter lives with the tool, not in proactive/, because it
#   is the integration-facing half: proactive defines the Protocol it needs,
#   and this module is where calendar providers are already understood.
# - Event ids passed to the trigger are "provider:id" references, so the
#   trigger's dedupe key stays unique when two calendars reuse an id.
