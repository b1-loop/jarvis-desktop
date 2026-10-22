"""Tool: create a calendar event after the user confirms it.

Creating an event writes to the user's account and, with attendees, sends
invitations to other people, so it always requires confirmation. The prompt
reads back title, day and time, length and attendees: everything that would
be embarrassing to get wrong.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Annotated, Final

from pydantic import Field

from jarvis.agent.tool import Tool, ToolCategory, ToolContext, ToolInput, ToolResult
from jarvis.integrations.base import CalendarEvent, CalendarProvider
from jarvis.tools._integration_support import (
    Service,
    calendar_providers,
    event_item,
    not_configured,
    provider_named,
    spoken_list,
    spoken_minutes,
    spoken_when,
    twelve_hour,
)

MAX_TITLE_CHARS: Final = 200
MAX_ATTENDEES: Final = 50
MAX_DURATION_MINUTES: Final = 24 * 60
PAST_TOLERANCE: Final = timedelta(minutes=5)
"""Starts slightly in the past are allowed: "a meeting now" takes a moment to confirm."""

EmailAddress = Annotated[
    str, Field(min_length=3, max_length=320, pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
]


class CreateEventInput(ToolInput):
    """Arguments for :class:`CreateEventTool`."""

    title: str = Field(min_length=1, max_length=MAX_TITLE_CHARS, description="Event title.")
    start: datetime = Field(
        description=(
            "Start as ISO 8601, e.g. '2026-10-23T14:00'. Without a UTC offset it is taken as "
            "the user's local time. Resolve words like 'tomorrow' with current_time first."
        )
    )
    duration_minutes: int = Field(
        default=30, ge=5, le=MAX_DURATION_MINUTES, description="Length in minutes."
    )
    attendees: list[EmailAddress] = Field(
        default_factory=list,
        max_length=MAX_ATTENDEES,
        description="Email addresses to invite; they receive an invitation from the provider.",
    )
    location: str = Field(default="", max_length=200, description="Room, address or link.")
    description: str = Field(default="", max_length=4000, description="Notes for the event.")
    calendar: str | None = Field(
        default=None,
        max_length=64,
        description="Provider name (e.g. 'google-calendar', 'outlook') when the user has "
        "several calendars; omit to use the first one.",
    )


class CreateEventTool(Tool[CreateEventInput]):
    """Creates an event, after the user confirms the read-back."""

    name = "create_event"
    description = (
        "Create a calendar event (and invite attendees, if given). Use when the user asks to "
        "schedule, book or add something to the calendar. The user is asked to confirm title, "
        "time and attendees first. Returns the created event."
    )
    Input = CreateEventInput
    category = ToolCategory.PRODUCTIVITY
    requires_confirmation = True
    keywords = ("meeting", "book", "boka", "möte", "kalender")

    def confirmation_prompt(self, params: CreateEventInput) -> str:
        """Read back title, absolute day and time, length and attendees."""
        start = params.start
        when = f"{start:%A} {start.day} {start:%B} at {twelve_hour(start)}"
        length = spoken_minutes(timedelta(minutes=params.duration_minutes))
        guests = f" with {spoken_list(params.attendees)}" if params.attendees else ""
        return f'Create "{params.title}" on {when} for {length}{guests}?'

    async def run(self, params: CreateEventInput, context: ToolContext) -> ToolResult:
        """Resolve the calendar and zone, refuse past times, create the event."""
        providers = calendar_providers(context)
        if not providers:
            return not_configured(Service.CALENDAR)
        provider = provider_named(providers, params.calendar) if params.calendar else providers[0]
        now = context.clock.now()
        start = params.start if params.start.tzinfo else params.start.replace(tzinfo=now.tzinfo)
        if start < now - PAST_TOLERANCE:
            return ToolResult.failure(
                "That time is already in the past, so I didn't create the event.",
                detail=f"start {start.isoformat()} is before now ({now.isoformat()})",
            )
        event = await provider.create_event(
            params.title,
            start,
            start + timedelta(minutes=params.duration_minutes),
            description=params.description,
            location=params.location,
            attendees=params.attendees,
        )
        return _result(provider, event, now)


def _result(provider: CalendarProvider, event: CalendarEvent, now: datetime) -> ToolResult:
    guests = f", with {spoken_list(list(event.attendees))} invited" if event.attendees else ""
    return ToolResult.success(
        f"Done: {event.title} {spoken_when(event.start, now)}{guests}.",
        detail=f"Created event {event.id} on {provider.provider_name}: {event.title} "
        f"{event.start.isoformat()} - {event.end.isoformat()}",
        data={"event": event_item(provider, event)},
    )


# Design notes:
# - A naive start is the user's local time (the clock's zone), because that
#   is what "two o'clock" means when spoken; the LLM is told so.
# - The prompt uses an absolute weekday and date rather than "tomorrow":
#   confirmation_prompt sees only the arguments, not the clock, and an
#   absolute date is also what catches an off-by-one-day mistake.
