from collections.abc import Callable

import pytest

from jarvis.agent.registry import ToolRegistry
from jarvis.config import Settings
from jarvis.integrations.base import Device
from jarvis.integrations.services import IntegrationServices
from jarvis.integrations.stubs import StubSmartHomeProvider
from jarvis.tools.list_devices import ListDevicesTool
from tests.tools.integration_helpers import integration_context


@pytest.fixture
def settings(make_settings: Callable[..., Settings]) -> Settings:
    return make_settings()


def test_is_discovered_and_read_only() -> None:
    tool = ToolRegistry.discover().get("list_devices")

    assert isinstance(tool, ListDevicesTool)
    assert not tool.requires_confirmation


async def test_lists_every_device_and_says_what_is_on(settings: Settings) -> None:
    context = integration_context(
        settings, IntegrationServices(smart_home=(StubSmartHomeProvider(),))
    )

    result = await ListDevicesTool().call({}, context)

    assert result.ok
    assert result.spoken == "I see 4 devices; on right now: Living room and Desk lamp."
    assert "- light.living_room (Living room, light): on, 80%" in result.detail
    assert result.data is not None
    assert [d["id"] for d in result.data["devices"]] == [
        "light.bedroom",
        "light.living_room",
        "scene.movie_night",
        "switch.desk_lamp",
    ]


async def test_filters_by_kind(settings: Settings) -> None:
    provider = StubSmartHomeProvider()
    context = integration_context(settings, IntegrationServices(smart_home=(provider,)))

    result = await ListDevicesTool().call({"kind": "scene"}, context)

    assert result.spoken == "I see 1 device; none of them is on."
    assert provider.calls == [("list_devices", {"kind": "scene"})]


async def test_no_devices(settings: Settings) -> None:
    context = integration_context(
        settings, IntegrationServices(smart_home=(StubSmartHomeProvider({}),))
    )

    result = await ListDevicesTool().call({}, context)

    assert result.spoken == "I don't see any devices of that kind."


async def test_many_devices_on_are_summarised(settings: Settings) -> None:
    devices = {f"light.l{i}": Device(f"light.l{i}", f"Lamp {i}", "light", "on") for i in range(7)}
    context = integration_context(
        settings, IntegrationServices(smart_home=(StubSmartHomeProvider(devices),))
    )

    result = await ListDevicesTool().call({}, context)

    assert result.spoken.endswith("Lamp 3, Lamp 4 and 2 more.")


async def test_kind_must_be_a_domain_name(settings: Settings) -> None:
    context = integration_context(
        settings, IntegrationServices(smart_home=(StubSmartHomeProvider(),))
    )

    result = await ListDevicesTool().call({"kind": "Lights!"}, context)

    assert not result.ok
    assert "kind" in result.detail
