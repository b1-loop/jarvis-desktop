"""Tool: battery level, charging state and estimated time remaining.

"How much battery do I have?", "how long will it last?". system_monitor
already reports the level as part of a full machine snapshot; this tool
answers the battery question alone, without sampling CPU for a third of a
second, and adds the time-remaining estimate the snapshot does not carry.
The top-processes question is system_monitor's and is not duplicated here.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Final, Protocol

import psutil

from jarvis.agent.tool import Tool, ToolCategory, ToolContext, ToolInput, ToolResult

_SECONDS_PER_MINUTE: Final = 60
_MINUTES_PER_HOUR: Final = 60


@dataclass(frozen=True, slots=True)
class BatteryReading:
    """One battery measurement.

    Attributes:
        percent: Charge level, 0-100.
        plugged_in: Whether mains power is connected.
        seconds_left: Estimated runtime on battery; ``None`` when charging
            or when the OS cannot estimate it.
    """

    percent: float
    plugged_in: bool
    seconds_left: int | None


class BatteryProbe(Protocol):
    """Source of battery readings; blocking, called from a worker thread."""

    def read(self) -> BatteryReading | None:
        """Current reading, or ``None`` on a machine without a battery."""


class PsutilBatteryProbe:
    """The real probe, backed by ``psutil.sensors_battery``."""

    def read(self) -> BatteryReading | None:
        """Read the battery; unknown and unlimited estimates become ``None``."""
        battery = psutil.sensors_battery()
        if battery is None:
            return None
        seconds = int(battery.secsleft)
        known = seconds not in (psutil.POWER_TIME_UNKNOWN, psutil.POWER_TIME_UNLIMITED)
        return BatteryReading(
            percent=float(battery.percent),
            plugged_in=bool(battery.power_plugged),
            seconds_left=seconds if known and seconds >= 0 else None,
        )


class BatteryStatusInput(ToolInput):
    """No arguments."""


class BatteryStatusTool(Tool[BatteryStatusInput]):
    """Answers battery questions quickly."""

    name = "battery_status"
    description = (
        "Get the battery charge percentage, whether the charger is connected and the "
        "estimated time remaining. For CPU, memory or the heaviest processes use "
        "system_monitor instead."
    )
    Input = BatteryStatusInput
    category = ToolCategory.SYSTEM

    def __init__(self, probe: BatteryProbe | None = None) -> None:
        self._probe = probe or PsutilBatteryProbe()

    async def run(self, params: BatteryStatusInput, context: ToolContext) -> ToolResult:
        """Read in a worker thread and phrase it."""
        del params, context
        reading = await asyncio.to_thread(self._probe.read)
        if reading is None:
            return ToolResult.success(
                "This computer doesn't have a battery.", data={"has_battery": False}
            )
        return ToolResult.success(
            _spoken(reading),
            data={
                "has_battery": True,
                "percent": reading.percent,
                "plugged_in": reading.plugged_in,
                "seconds_left": reading.seconds_left,
            },
        )


def _spoken(reading: BatteryReading) -> str:
    level = f"Battery is at {reading.percent:.0f} percent"
    if reading.plugged_in:
        return f"{level} and charging."
    if reading.seconds_left is None:
        return f"{level}, on battery."
    return f"{level}, about {_duration(reading.seconds_left)} left."


def _duration(seconds: int) -> str:
    hours, minutes = divmod(seconds // _SECONDS_PER_MINUTE, _MINUTES_PER_HOUR)
    if hours == 0:
        return f"{minutes} minutes"
    hour_word = "hour" if hours == 1 else "hours"
    return f"{hours} {hour_word} {minutes} minutes" if minutes else f"{hours} {hour_word}"


# Design notes:
# - Same probe-behind-a-Protocol pattern as system_monitor, so the tool is
#   tested with a fake and psutil (the ``desktop`` extra) is the only real
#   dependency; a missing psutil is a recorded discovery failure.
# - "Charging" is reported for plugged-in regardless of the estimate,
#   because psutil reports an unlimited estimate while on mains power.
