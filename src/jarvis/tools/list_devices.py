"""Tool: the smart-home devices Jarvis can control, with their state.

"Which lights are on?" and the lookup the model needs before ``set_device``:
device ids such as ``light.kitchen`` come from here, so the model never has
to guess one.
"""

from __future__ import annotations

from typing import Final

from pydantic import Field, JsonValue

from jarvis.agent.tool import Tool, ToolCategory, ToolContext, ToolInput, ToolResult
from jarvis.integrations.base import Device, SmartHomeProvider
from jarvis.tools._integration_support import (
    Service,
    count_phrase,
    device_item,
    flatten,
    gather_each,
    not_configured,
    smart_home_providers,
    spoken_list,
)

_SPOKEN_NAMES: Final = 5


class ListDevicesInput(ToolInput):
    """Arguments for :class:`ListDevicesTool`."""

    kind: str | None = Field(
        default=None,
        pattern=r"^[a-z_]{2,32}$",
        description="Only devices of this kind (Home Assistant domain), e.g. 'light', "
        "'switch', 'scene', 'climate'; omit for all.",
    )


class ListDevicesTool(Tool[ListDevicesInput]):
    """Lists controllable devices and their current state."""

    name = "list_devices"
    description = (
        "List the user's smart-home devices (lights, switches, scenes, thermostats ...) with "
        "their id, name, kind, state and brightness. Use for 'which lights are on?' and to "
        "find the device id before calling set_device."
    )
    Input = ListDevicesInput
    category = ToolCategory.SMART_HOME

    async def run(self, params: ListDevicesInput, context: ToolContext) -> ToolResult:
        """Collect devices from every smart-home provider."""
        providers = smart_home_providers(context)
        if not providers:
            return not_configured(Service.SMART_HOME)
        gathered = await gather_each(providers, lambda p: p.list_devices(params.kind))
        pairs = sorted(flatten(gathered), key=lambda pair: (pair[1].kind, pair[1].name))
        items: list[JsonValue] = [device_item(provider, device) for provider, device in pairs]
        lines = [f"{len(pairs)} device(s):"]
        lines.extend(_line(device) for _, device in pairs)
        lines.append(gathered.failure_note())
        return ToolResult.success(
            _spoken(pairs),
            detail="\n".join(lines).strip(),
            data={"devices": items, "failures": gathered.failure_data()},
        )


def _line(device: Device) -> str:
    level = f", {device.brightness}%" if device.brightness is not None and device.is_on else ""
    return f"- {device.id} ({device.name}, {device.kind}): {device.state}{level}"


def _spoken(pairs: list[tuple[SmartHomeProvider, Device]]) -> str:
    if not pairs:
        return "I don't see any devices of that kind."
    on = [device.name for _, device in pairs if device.is_on]
    found = f"I see {count_phrase(len(pairs), 'device')}"
    if not on:
        return f"{found}; none of them is on."
    names = on[:_SPOKEN_NAMES]
    if len(on) > len(names):
        listed = f"{', '.join(names)} and {len(on) - len(names)} more"
    else:
        listed = spoken_list(names)
    return f"{found}; on right now: {listed}."
