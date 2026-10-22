from collections.abc import Callable

import pytest

from jarvis.agent.confirmation import DECLINED_SPOKEN, ConfirmationPolicy
from jarvis.agent.registry import ToolRegistry
from jarvis.agent.tool import ToolContext
from jarvis.config import Settings
from jarvis.integrations.base import Device
from jarvis.integrations.services import IntegrationServices
from jarvis.integrations.stubs import StubSmartHomeProvider
from jarvis.tools.set_device import SetDeviceInput, SetDeviceTool, domain_of
from tests.agent.fakes import FakeConfirmation
from tests.tools.integration_helpers import integration_context


def home(*extra: Device) -> StubSmartHomeProvider:
    provider = StubSmartHomeProvider()
    provider.devices.update({device.id: device for device in extra})
    return provider


@pytest.fixture
def settings(make_settings: Callable[..., Settings]) -> Settings:
    return make_settings()


@pytest.fixture
def provider() -> StubSmartHomeProvider:
    return home(Device("lock.front_door", "Front door", "lock", "locked"))


@pytest.fixture
def context(settings: Settings, provider: StubSmartHomeProvider) -> ToolContext:
    return integration_context(settings, IntegrationServices(smart_home=(provider,)))


def test_is_discovered_and_confirms_per_device() -> None:
    tool = ToolRegistry.discover().get("set_device")

    assert isinstance(tool, SetDeviceTool)
    assert not tool.requires_confirmation  # decided from the device's domain


@pytest.mark.parametrize(
    ("device_id", "expected"),
    [
        ("light.kitchen", False),
        ("switch.desk_lamp", False),
        ("scene.movie_night", False),
        ("lock.front_door", True),
        ("alarm_control_panel.home", True),
        ("climate.living_room", True),
        ("water_heater.boiler", True),
        ("kitchen_light", False),
    ],
)
def test_confirmation_follows_the_domain(
    context: ToolContext, device_id: str, expected: bool
) -> None:
    params = SetDeviceInput(device_id=device_id, action="off")

    assert SetDeviceTool().needs_confirmation(params, context) is expected


def test_domain_of() -> None:
    assert domain_of("light.kitchen") == "light"
    assert domain_of("kitchen") == ""


async def test_light_turns_off_without_asking(
    settings: Settings, provider: StubSmartHomeProvider
) -> None:
    asker = FakeConfirmation(answers=[])
    context = integration_context(
        settings, IntegrationServices(smart_home=(provider,)), confirmation=asker
    )

    result = await ConfirmationPolicy().execute_with_policy(
        SetDeviceTool(), {"device_id": "light.living_room", "action": "off"}, context
    )

    assert result.ok
    assert result.spoken == "Living room is off."
    assert asker.prompts == []
    assert provider.devices["light.living_room"].state == "off"


async def test_brightness(context: ToolContext, provider: StubSmartHomeProvider) -> None:
    result = await SetDeviceTool().call(
        {"device_id": "light.bedroom", "action": "brightness", "brightness": 30}, context
    )

    assert result.spoken == "Bedroom is at 30 percent."
    assert provider.devices["light.bedroom"].brightness == 30
    assert result.data is not None
    assert result.data["device"]["state"] == "on"


async def test_switch_on(context: ToolContext) -> None:
    result = await SetDeviceTool().call({"device_id": "switch.desk_lamp", "action": "on"}, context)

    assert result.spoken == "Desk lamp is on."


async def test_scene(context: ToolContext, provider: StubSmartHomeProvider) -> None:
    result = await SetDeviceTool().call(
        {"device_id": "scene.movie_night", "action": "scene"}, context
    )

    assert result.spoken == "Scene activated."
    assert ("set_scene", {"scene_id": "scene.movie_night"}) in provider.calls


async def test_lock_asks_first_and_respects_no(
    settings: Settings, provider: StubSmartHomeProvider
) -> None:
    asker = FakeConfirmation(answers=[False])
    context = integration_context(
        settings, IntegrationServices(smart_home=(provider,)), confirmation=asker
    )

    result = await ConfirmationPolicy().execute_with_policy(
        SetDeviceTool(), {"device_id": "lock.front_door", "action": "off"}, context
    )

    assert result.spoken == DECLINED_SPOKEN
    assert asker.prompts == [
        "Do you want me to turn off lock.front_door? It's a lock, so I'm checking first."
    ]
    assert provider.devices["lock.front_door"].state == "locked"


def test_prompts_for_brightness_and_scene() -> None:
    tool = SetDeviceTool()

    dim = SetDeviceInput(device_id="climate.hall", action="brightness", brightness=20)
    scene = SetDeviceInput(device_id="alarm_control_panel.x", action="scene")

    assert tool.confirmation_prompt(dim) == (
        "Do you want me to set climate.hall to 20 percent? "
        "It's a heating or cooling control, so I'm checking first."
    )
    assert tool.confirmation_prompt(scene) == (
        "Do you want me to activate alarm_control_panel.x? It's an alarm, so I'm checking first."
    )


async def test_sensitive_kind_behind_a_domainless_id_is_refused(settings: Settings) -> None:
    provider = home(Device("frontdoor", "Front door", "lock", "locked"))
    context = integration_context(settings, IntegrationServices(smart_home=(provider,)))

    result = await SetDeviceTool().call({"device_id": "frontdoor", "action": "off"}, context)

    assert not result.ok
    assert "is a lock" in result.detail
    assert provider.devices["frontdoor"].state == "locked"


async def test_device_is_found_on_the_right_provider(settings: Settings) -> None:
    empty, owner = StubSmartHomeProvider({}), home()
    context = integration_context(settings, IntegrationServices(smart_home=(empty, owner)))

    result = await SetDeviceTool().call({"device_id": "switch.desk_lamp", "action": "off"}, context)
    missing = await SetDeviceTool().call({"device_id": "light.garage", "action": "on"}, context)

    assert result.ok
    assert owner.devices["switch.desk_lamp"].state == "off"
    assert not missing.ok
    assert "unknown device 'light.garage'" in missing.detail


@pytest.mark.parametrize(
    "arguments",
    [
        {"device_id": "light.x", "action": "brightness"},
        {"device_id": "light.x", "action": "on", "brightness": 50},
        {"device_id": "light x; rm", "action": "on"},
    ],
)
async def test_invalid_arguments_are_rejected(
    context: ToolContext, arguments: dict[str, object]
) -> None:
    result = await SetDeviceTool().call(arguments, context)

    assert not result.ok
