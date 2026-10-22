"""Tool: find free time of a given length inside working hours.

"When am I free for an hour this week?" The gap arithmetic is the tested
``CalendarProvider.find_free_slots``; this tool adds what a single provider
cannot know: with Google *and* Outlook calendars a slot is only free if it
is free in both, so the per-calendar answers are intersected.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, time, timedelta
from typing import Final, Self

from pydantic import Field, JsonValue, model_validator

from jarvis.agent.tool import Tool, ToolCategory, ToolContext, ToolInput, ToolResult
from jarvis.integrations.base import TimeRange, WorkingHours
from jarvis.tools._integration_support import (
    Service,
    calendar_providers,
    gather_each,
    not_configured,
    spoken_list,
    spoken_minutes,
    spoken_when,
)

MIN_DURATION_MINUTES: Final = 5
MAX_DURATION_MINUTES: Final = 8 * 60
MAX_DAYS_AHEAD: Final = 14
MAX_RESULTS: Final = 10
ROUNDING: Final = timedelta(minutes=5)
"""Search starts at the next multiple of five minutes; "free at 15:42:07" reads badly."""


class FindFreeSlotInput(ToolInput):
    """Arguments for :class:`FindFreeSlotTool`."""

    duration_minutes: int = Field(
        default=30,
        ge=MIN_DURATION_MINUTES,
        le=MAX_DURATION_MINUTES,
        description="Length of the free slot needed, in minutes.",
    )
    days_ahead: int = Field(
        default=3,
        ge=1,
        le=MAX_DAYS_AHEAD,
        description="How many days to search, starting now (1 = rest of today).",
    )
    working_hours_start: time = Field(
        default=time(9), description="Earliest local start time, e.g. '09:00'."
    )
    working_hours_end: time = Field(
        default=time(17), description="Latest local end time, e.g. '17:00'."
    )
    include_weekends: bool = Field(default=False, description="Also offer Saturday and Sunday.")
    max_results: int = Field(
        default=3, ge=1, le=MAX_RESULTS, description="How many slots to return, earliest first."
    )

    @model_validator(mode="after")
    def _hours_are_ordered(self) -> Self:
        if self.working_hours_end <= self.working_hours_start:
            raise ValueError("working_hours_end must be after working_hours_start")
        return self


class FindFreeSlotTool(Tool[FindFreeSlotInput]):
    """Finds slots that are free on every connected calendar."""

    name = "find_free_slot"
    description = (
        "Find free time slots of a given length within working hours over the next few days, "
        "free on all of the user's calendars. Use for 'when am I free for an hour this week?' "
        "or before create_event when the user has not chosen a time. Returns slots earliest "
        "first, each with start and end."
    )
    Input = FindFreeSlotInput
    category = ToolCategory.PRODUCTIVITY

    async def run(self, params: FindFreeSlotInput, context: ToolContext) -> ToolResult:
        """Ask every calendar for its free slots and intersect them."""
        providers = calendar_providers(context)
        if not providers:
            return not_configured(Service.CALENDAR)
        now = context.clock.now()
        window = _search_window(now, params.days_ahead)
        duration = timedelta(minutes=params.duration_minutes)
        hours = WorkingHours(
            params.working_hours_start, params.working_hours_end, not params.include_weekends
        )
        gathered = await gather_each(
            providers, lambda p: p.find_free_slots(duration, window, hours)
        )
        free = _common_slots([slots for _, slots in gathered.results], duration)
        slots = free[: params.max_results]
        items: list[JsonValue] = [_slot_item(slot) for slot in slots]
        return ToolResult.success(
            _spoken(slots, duration, now),
            detail=_detail(slots, duration, gathered.failure_note()),
            data={
                "duration_minutes": params.duration_minutes,
                "slots": items,
                "failures": gathered.failure_data(),
            },
        )


def _search_window(now: datetime, days_ahead: int) -> TimeRange:
    start = _round_up(now)
    midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
    return TimeRange(start, midnight + timedelta(days=days_ahead))


def _round_up(moment: datetime) -> datetime:
    floor = moment.replace(second=0, microsecond=0)
    floor -= timedelta(minutes=floor.minute % (ROUNDING.seconds // 60))
    return floor if floor == moment else floor + ROUNDING


def _common_slots(per_calendar: Sequence[list[TimeRange]], duration: timedelta) -> list[TimeRange]:
    """Ranges free in every calendar and still at least ``duration`` long."""
    if not per_calendar:
        return []
    common = per_calendar[0]
    for slots in per_calendar[1:]:
        common = _intersect(common, slots)
    return [slot for slot in common if slot.duration >= duration]


def _intersect(left: list[TimeRange], right: list[TimeRange]) -> list[TimeRange]:
    overlaps = [
        TimeRange(max(a.start, b.start), min(a.end, b.end))
        for a in left
        for b in right
        if a.overlaps(b)
    ]
    return sorted(overlaps, key=lambda slot: slot.start)


def _slot_item(slot: TimeRange) -> dict[str, JsonValue]:
    return {
        "start": slot.start.isoformat(timespec="minutes"),
        "end": slot.end.isoformat(timespec="minutes"),
        "free_minutes": int(slot.duration.total_seconds() // 60),
    }


def _spoken(slots: list[TimeRange], duration: timedelta, now: datetime) -> str:
    if not slots:
        return f"I couldn't find {spoken_minutes(duration)} free in that time."
    starts = [spoken_when(slot.start, now) for slot in slots]
    return f"You're free for {spoken_minutes(duration)} {spoken_list(starts)}."


def _detail(slots: list[TimeRange], duration: timedelta, failure_note: str) -> str:
    lines = [f"Free slots of at least {duration} (start of each free window):"]
    lines.extend(f"- {slot.start:%a %d %b %H:%M} until {slot.end:%H:%M}" for slot in slots)
    if failure_note:
        lines.append(f"{failure_note} (availability on those calendars is unknown)")
    return "\n".join(lines)


# Design notes:
# - Intersection instead of merging busy times across providers keeps the
#   gap logic in one tested place (base.CalendarProvider) and the cross-
#   calendar rule here; for the handful of slots a day this is cheap.
# - A calendar that fails is left out of the intersection and named in the
#   detail, so the LLM can say "I couldn't check your Outlook calendar"
#   instead of offering a slot that may clash.
# - Working hours are interpreted in the zone of the clock's ``now``, which
#   is the user's local zone; WorkingHours days are built in that zone.
