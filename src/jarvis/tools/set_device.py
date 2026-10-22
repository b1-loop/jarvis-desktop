"""Tool: switch, dim or activate a smart-home device.

"Turn off the kitchen light", "dim the living room to 30 percent", "movie
night scene". Lights, switches and scenes act immediately; devices whose
domain is a lock, an alarm panel or a heating/cooling control require the
user's spoken confirmation first, because getting those wrong has physical
consequences (an unlocked door, a disarmed alarm, a cold house).
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum
from typing import Final, Self

from pydantic import Field, model_validator

from jarvis.agent.tool import Tool, ToolCategory, ToolContext, ToolInput, ToolResult
from jarvis.integrations.base import Device, IntegrationError, SmartHomeProvider
from jarvis.tools._integration_support import (
    Service,
    device_item,
    not_configured,
    smart_home_providers,
)

SENSITIVE_DOMAINS: Final[dict[str, str]] = {
    "lock": "a lock",
    "alarm_control_panel": "an alarm",
    "climate": "a heating or cooling control",
    "water_heater": "a water heater",
}
"""Domains whose devices are confirmed before every change, with a spoken label."""

_DOMAIN_SEPARATOR: Final = "."


class DeviceAction(StrEnum):
    """What to do to the device."""

    ON = "on"
    OFF = "off"
    BRIGHTNESS = "brightness"
    SCENE = "scene"


class SetDeviceInput(ToolInput):
    """Arguments for :class:`SetDeviceTool`."""

    device_id: str = Field(
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9_.\-]+$",
        description="Device id exactly as list_devices returned it, e.g. 'light.kitchen'.",
    )
    action: DeviceAction = Field(
        description="'on', 'off', 'brightness' (needs brightness) or 'scene' (device_id is "
        "a scene)."
    )
    brightness: int | None = Field(
        default=None, ge=0, le=100, description="Percent 0-100; only with action 'brightness'."
    )

    @model_validator(mode="after")
    def _brightness_matches_action(self) -> Self:
        if (self.action is DeviceAction.BRIGHTNESS) != (self.brightness is not None):
            raise ValueError("brightness is required with action 'brightness' and only then")
        return self


def domain_of(device_id: str) -> str:
    """``"light.kitchen"`` -> ``"light"``; empty when the id carries no domain."""
    domain, separator, _ = device_id.partition(_DOMAIN_SEPARATOR)
    return domain if separator else ""


class SetDeviceTool(Tool[SetDeviceInput]):
    """Changes one device's state; confirms first for locks, alarms and heating."""

    name = "set_device"
    description = (
        "Turn a smart-home device on or off, set a light's brightness, or activate a scene. "
        "Use the device id from list_devices. Locks, alarms and thermostats ask the user to "
        "confirm first; lights and switches act immediately. Returns the device's new state."
    )
    Input = SetDeviceInput
    category = ToolCategory.SMART_HOME

    def needs_confirmation(self, params: SetDeviceInput, context: ToolContext) -> bool:
        """Confirm only for sensitive domains, read from the id (``lock.front_door``)."""
        del context
        return domain_of(params.device_id) in SENSITIVE_DOMAINS

    def confirmation_prompt(self, params: SetDeviceInput) -> str:
        """Say what will happen and why Jarvis is asking."""
        label = SENSITIVE_DOMAINS.get(domain_of(params.device_id), "a sensitive device")
        return f"Do you want me to {_phrase(params)}? It's {label}, so I'm checking first."

    async def run(self, params: SetDeviceInput, context: ToolContext) -> ToolResult:
        """Find the provider that owns the device, then apply the action."""
        providers = smart_home_providers(context)
        if not providers:
            return not_configured(Service.SMART_HOME)
        if params.action is DeviceAction.SCENE:
            await providers[0].set_scene(params.device_id)
            return ToolResult.success(
                "Scene activated.", data={"scene": params.device_id, "action": "scene"}
            )
        provider, device = await _locate(providers, params.device_id)
        if device.kind in SENSITIVE_DOMAINS and not self.needs_confirmation(params, context):
            return ToolResult.failure(
                "That device needs your confirmation, and I couldn't ask for it.",
                detail=f"{device.id} is a {device.kind} but its id carries no domain",
            )
        updated = await _apply(provider, params)
        return ToolResult.success(
            _spoken(updated, params),
            detail=f"{updated.id} on {provider.provider_name} is now {updated.state}.",
            data={"device": device_item(provider, updated), "action": params.action.value},
        )


async def _locate(
    providers: Sequence[SmartHomeProvider], device_id: str
) -> tuple[SmartHomeProvider, Device]:
    """The provider that knows ``device_id``; the last error if none does."""
    last_error: IntegrationError | None = None
    for provider in providers:
        try:
            return provider, await provider.state(device_id)
        except IntegrationError as error:
            last_error = error
    assert last_error is not None  # providers is non-empty
    raise last_error


async def _apply(provider: SmartHomeProvider, params: SetDeviceInput) -> Device:
    if params.action is DeviceAction.ON:
        return await provider.turn_on(params.device_id)
    if params.action is DeviceAction.OFF:
        return await provider.turn_off(params.device_id)
    assert params.brightness is not None  # enforced by the input validator
    return await provider.set_brightness(params.device_id, params.brightness)


def _phrase(params: SetDeviceInput) -> str:
    target = params.device_id
    if params.action is DeviceAction.BRIGHTNESS:
        return f"set {target} to {params.brightness} percent"
    if params.action is DeviceAction.SCENE:
        return f"activate {target}"
    return f"turn {params.action.value} {target}"


def _spoken(device: Device, params: SetDeviceInput) -> str:
    if params.action is DeviceAction.BRIGHTNESS:
        return f"{device.name} is at {device.brightness} percent."
    return f"{device.name} is {device.state}."


# Design notes:
# - Confirmation is decided from the id's domain because needs_confirmation
#   is synchronous and runs before any provider call. Home Assistant ids
#   always carry it. As a second line, run() refuses a device whose
#   reported kind is sensitive when the id did not reveal it, so a provider
#   with domain-less ids can never bypass the check.
# - Lights and switches do not confirm: they are what the user asked for,
#   reversible in one sentence and harmless if misheard. Asking every time
#   would make "lights off" slower than a wall switch. A user who wants
#   stricter behaviour can list ToolCategory.SMART_HOME in the policy's
#   always-confirm categories.
# - Scenes are activated on the first provider: scene ids are not listed
#   per provider by the interface, and multi-hub setups are not a goal.
