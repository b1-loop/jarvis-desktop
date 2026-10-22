"""Tool: set the built-in display's brightness to a level or by a step.

Software brightness only exists for laptop panels (and a few monitors with
DDC/CI); on a desktop with external screens the platform raises
``UnsupportedFeatureError``, which this tool turns into a clear spoken
answer instead of a generic failure.
"""

from __future__ import annotations

from typing import Final, Self

from pydantic import Field, model_validator

from jarvis.agent.tool import Tool, ToolCategory, ToolContext, ToolInput, ToolResult
from jarvis.os_integration.errors import UnsupportedFeatureError
from jarvis.tools._common import require_platform

PERCENT_MIN: Final = 0
PERCENT_MAX: Final = 100
UNSUPPORTED_SPOKEN: Final = (
    "I can't change the brightness on this screen; it has no software brightness control."
)


class SetBrightnessInput(ToolInput):
    """Arguments for :class:`SetBrightnessTool`; exactly one of the two fields."""

    percent: int | None = Field(
        default=None,
        ge=PERCENT_MIN,
        le=PERCENT_MAX,
        description="Absolute brightness, 0-100. Use for 'brightness to 70'.",
    )
    change: int | None = Field(
        default=None,
        ge=-PERCENT_MAX,
        le=PERCENT_MAX,
        description="Relative change in points, e.g. -20 for 'dimmer', 20 for 'brighter'.",
    )

    @model_validator(mode="after")
    def _exactly_one(self) -> Self:
        if (self.percent is None) == (self.change is None):
            raise ValueError("give exactly one of 'percent' or 'change'")
        return self


class SetBrightnessTool(Tool[SetBrightnessInput]):
    """Adjusts the screen brightness, or says plainly that it cannot."""

    name = "set_brightness"
    description = (
        "Set the built-in display brightness to a percentage (percent) or change it by "
        "points (change, negative for dimmer). Fails with an explanation on screens "
        "without software brightness control, typically external desktop monitors."
    )
    Input = SetBrightnessInput
    category = ToolCategory.SYSTEM
    keywords = (
        "dim",
        "dimmer",
        "brighter",
        "screen",
        "display",
        "ljusstyrka",
        "ljusstyrkan",
        "ljusare",
        "mörkare",
        "dämpa",
        "skärm",
    )

    async def run(self, params: SetBrightnessInput, context: ToolContext) -> ToolResult:
        """Compute the clamped target and apply it."""
        platform = require_platform(context, self.name)
        try:
            target = params.percent
            if target is None:
                target = await platform.get_brightness() + (params.change or 0)
            target = max(PERCENT_MIN, min(PERCENT_MAX, target))
            await platform.set_brightness(target)
        except UnsupportedFeatureError as error:
            return ToolResult.failure(UNSUPPORTED_SPOKEN, detail=str(error))
        return ToolResult.success(
            f"Brightness set to {target} percent.", data={"brightness": target}
        )


# Design notes:
# - No confirmation, for the same reason as volume: a reversible setting
#   the user asked for by name.
