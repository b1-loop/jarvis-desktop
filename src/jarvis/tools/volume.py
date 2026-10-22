"""Tools: read the master volume, set it, and mute or unmute.

Three tiny siblings over the same two Platform calls ("what's the volume",
"volume to 40", "a bit louder", "mute"). They live in one module because
each is a few lines and they share the percent/step vocabulary; the
registry discovers every Tool class in the file.
"""

from __future__ import annotations

from typing import Final, Self

from pydantic import Field, model_validator

from jarvis.agent.tool import Tool, ToolCategory, ToolContext, ToolInput, ToolResult
from jarvis.tools._common import require_platform

PERCENT_MIN: Final = 0
PERCENT_MAX: Final = 100


class GetVolumeInput(ToolInput):
    """No arguments."""


class GetVolumeTool(Tool[GetVolumeInput]):
    """Reports the master volume and mute state."""

    name = "get_volume"
    description = "Get the current master volume (0-100) and whether sound is muted."
    Input = GetVolumeInput
    category = ToolCategory.SYSTEM

    async def run(self, params: GetVolumeInput, context: ToolContext) -> ToolResult:
        """Read both values; mute matters because "it's silent" may not be the level."""
        del params
        platform = require_platform(context, self.name)
        volume = await platform.get_volume()
        muted = await platform.is_muted()
        spoken = f"Volume is at {volume} percent" + (", but sound is muted." if muted else ".")
        return ToolResult.success(spoken, data={"volume": volume, "muted": muted})


class SetVolumeInput(ToolInput):
    """Arguments for :class:`SetVolumeTool`; exactly one of the two fields."""

    percent: int | None = Field(
        default=None,
        ge=PERCENT_MIN,
        le=PERCENT_MAX,
        description="Absolute volume level, 0-100. Use for 'set the volume to 40'.",
    )
    change: int | None = Field(
        default=None,
        ge=-PERCENT_MAX,
        le=PERCENT_MAX,
        description=(
            "Relative change in percentage points, e.g. 10 for 'a bit louder', "
            "-20 for 'much quieter'."
        ),
    )

    @model_validator(mode="after")
    def _exactly_one(self) -> Self:
        if (self.percent is None) == (self.change is None):
            raise ValueError("give exactly one of 'percent' or 'change'")
        return self


class SetVolumeTool(Tool[SetVolumeInput]):
    """Sets the master volume to a level or by a step."""

    name = "set_volume"
    description = (
        "Set the master volume to an absolute percentage (percent) or change it by a "
        "number of points (change, negative for quieter). Setting a volume above 0 "
        "also unmutes."
    )
    Input = SetVolumeInput
    category = ToolCategory.SYSTEM

    async def run(self, params: SetVolumeInput, context: ToolContext) -> ToolResult:
        """Compute the target, clamp, apply and unmute."""
        platform = require_platform(context, self.name)
        target = params.percent
        if target is None:
            target = await platform.get_volume() + (params.change or 0)
        target = max(PERCENT_MIN, min(PERCENT_MAX, target))
        await platform.set_volume(target)
        if target > PERCENT_MIN:
            await platform.set_muted(False)
        return ToolResult.success(f"Volume set to {target} percent.", data={"volume": target})


class MuteInput(ToolInput):
    """Arguments for :class:`MuteTool`."""

    muted: bool = Field(default=True, description="True to mute, false to unmute.")


class MuteTool(Tool[MuteInput]):
    """Mutes or unmutes the master output."""

    name = "mute"
    description = "Mute (muted=true) or unmute (muted=false) all sound output."
    Input = MuteInput
    category = ToolCategory.SYSTEM

    async def run(self, params: MuteInput, context: ToolContext) -> ToolResult:
        """Apply the mute state."""
        platform = require_platform(context, self.name)
        await platform.set_muted(params.muted)
        return ToolResult.success(
            "Muted." if params.muted else "Sound is back on.", data={"muted": params.muted}
        )


# Design notes:
# - No confirmation for any of these: the user asked for exactly this
#   change, it is undone by the next sentence, and asking "are you sure?"
#   before "louder" would make the assistant unusable.
# - "set_volume to 40" implicitly unmutes because a user who asks for a
#   level expects to hear it; a silent 40 % would read as a bug.
