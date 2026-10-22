"""Tool: the current date, time, weekday and time zone.

The LLM has no clock of its own, so almost every scheduling question
("what day is it", "how long until five") starts here. Time comes from the
injected ``Clock`` so the answer is deterministic in tests.
"""

from __future__ import annotations

from datetime import datetime
from typing import Final
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import Field

from jarvis.agent.tool import Tool, ToolCategory, ToolContext, ToolInput, ToolResult

_HOURS_ON_A_CLOCK_FACE: Final = 12


class CurrentTimeInput(ToolInput):
    """Arguments for :class:`CurrentTimeTool`."""

    timezone: str | None = Field(
        default=None,
        description=(
            "IANA time zone name such as 'Europe/Sofia' or 'America/New_York'. "
            "Omit for the computer's local time zone."
        ),
    )


class CurrentTimeTool(Tool[CurrentTimeInput]):
    """Reports the current moment, optionally converted to another zone."""

    name = "current_time"
    description = (
        "Get the current date, time, weekday and time zone. "
        "Optionally convert to another IANA time zone."
    )
    Input = CurrentTimeInput
    category = ToolCategory.INFORMATION
    keywords = (
        "time",
        "clock",
        "date",
        "day",
        "today",
        "hour",
        "klocka",
        "klockan",
        "tid",
        "tiden",
        "datum",
        "dag",
        "idag",
        "veckodag",
    )

    async def run(self, params: CurrentTimeInput, context: ToolContext) -> ToolResult:
        """Read the clock and phrase it for speech and for the model."""
        now = context.clock.now()
        if params.timezone is not None:
            try:
                now = now.astimezone(ZoneInfo(params.timezone))
            except (ZoneInfoNotFoundError, ValueError):
                return ToolResult.failure(
                    f"I don't know a time zone called {params.timezone}.",
                    detail=f"unknown IANA time zone: {params.timezone!r}",
                )
        zone = now.tzname() or "local time"
        weekday = now.strftime("%A")
        spoken = f"It's {spoken_time(now)} on {weekday}, {now.day} {now:%B} {now.year}"
        if params.timezone is not None:
            spoken += f" in {params.timezone}"
        return ToolResult.success(
            spoken + ".",
            detail=f"{now.isoformat(timespec='seconds')} ({weekday}, {zone})",
            data={
                "iso": now.isoformat(timespec="seconds"),
                "weekday": weekday,
                "timezone": zone,
                "utc_offset": now.strftime("%z"),
                "unix": int(now.timestamp()),
            },
        )


def spoken_time(moment: datetime) -> str:
    """``15:05`` -> ``3:05 PM``; avoids the platform-specific ``%-I`` directive."""
    hour = moment.hour % _HOURS_ON_A_CLOCK_FACE or _HOURS_ON_A_CLOCK_FACE
    meridiem = "AM" if moment.hour < _HOURS_ON_A_CLOCK_FACE else "PM"
    return f"{hour}:{moment:%M} {meridiem}"


# Design notes:
# - The spoken form is hand-built rather than strftime'd because TTS reads
#   "15:05:00+03:00" badly and because ``%-I`` is not portable to Windows.
# - An unknown zone is a failed result, not an exception: the LLM should
#   get the chance to ask the user which city they meant.
