from collections.abc import Callable

import httpx
import pytest

from jarvis.config import Settings
from jarvis.integrations.base import AuthError, NotConfiguredError
from jarvis.integrations.home_assistant import HomeAssistantProvider, parse_state
from jarvis.integrations.tokens import StaticTokenSource
from tests.integrations.conftest import FAKE_TOKEN, Router

STATES = [
    {
        "entity_id": "light.kitchen",
        "state": "on",
        "attributes": {"friendly_name": "Kitchen", "brightness": 128, "color_mode": "xy"},
    },
    {"entity_id": "light.hall", "state": "off", "attributes": {"friendly_name": "Hall"}},
    {"entity_id": "switch.fan", "state": "on", "attributes": {"friendly_name": "Fan"}},
    {"entity_id": "scene.movie", "state": "scening", "attributes": {"friendly_name": "Movie"}},
    {"entity_id": "sensor.temp", "state": "21.5", "attributes": {"unit_of_measurement": "C"}},
]


@pytest.fixture
def provider(router: Router, token_source: StaticTokenSource) -> HomeAssistantProvider:
    router.add("GET", "/api/states", STATES)
    for state in STATES:
        router.add("GET", f"/api/states/{state['entity_id']}", state)
    return HomeAssistantProvider(router.client(), token_source, base_url="http://ha.local:8123/")


async def test_list_devices_hides_sensors_unless_asked(
    router: Router, provider: HomeAssistantProvider
) -> None:
    devices = await provider.list_devices()
    sensors = await provider.list_devices("sensor")

    assert str(router.requests[0].url) == "http://ha.local:8123/api/states"
    assert router.requests[0].headers["Authorization"] == f"Bearer {FAKE_TOKEN}"
    assert [d.id for d in devices] == ["light.kitchen", "light.hall", "switch.fan", "scene.movie"]
    assert [d.id for d in sensors] == ["sensor.temp"]
    kitchen = devices[0]
    assert kitchen.name == "Kitchen" and kitchen.kind == "light" and kitchen.is_on
    assert kitchen.brightness == 50
    assert kitchen.attributes == {"brightness": 128, "color_mode": "xy"}


def test_brightness_mapping() -> None:
    assert parse_state(STATES[1]).brightness == 0
    assert parse_state(STATES[2]).brightness is None
    on_no_brightness = {"entity_id": "light.x", "state": "on", "attributes": {}}
    assert parse_state(on_no_brightness).brightness is None
    assert parse_state({"entity_id": "light.y"}).state == "unknown"


async def test_turn_on_and_off_call_the_domain_service_then_reread(
    router: Router, provider: HomeAssistantProvider
) -> None:
    router.add("POST", "/api/services/light/turn_on", [])
    router.add("POST", "/api/services/switch/turn_off", [])

    hall = await provider.turn_on("light.hall")
    fan = await provider.turn_off("switch.fan")

    calls = [(r.method, r.url.path) for r in router.requests]
    assert calls == [
        ("POST", "/api/services/light/turn_on"),
        ("GET", "/api/states/light.hall"),
        ("POST", "/api/services/switch/turn_off"),
        ("GET", "/api/states/switch.fan"),
    ]
    assert router.sent_json(0) == {"entity_id": "light.hall"}
    assert hall.id == "light.hall" and fan.id == "switch.fan"


async def test_set_brightness_uses_percent_and_zero_turns_off(
    router: Router, provider: HomeAssistantProvider
) -> None:
    router.add("POST", "/api/services/light/turn_on", [])
    router.add("POST", "/api/services/light/turn_off", [])

    await provider.set_brightness("light.kitchen", 40)
    await provider.set_brightness("light.kitchen", 0)

    assert router.sent_json(0) == {"entity_id": "light.kitchen", "brightness_pct": 40}
    assert router.requests[2].url.path == "/api/services/light/turn_off"
    with pytest.raises(ValueError, match="0-100"):
        await provider.set_brightness("light.kitchen", 101)


async def test_set_scene_and_state(router: Router, provider: HomeAssistantProvider) -> None:
    router.add("POST", "/api/services/scene/turn_on", [])

    await provider.set_scene("scene.movie")
    device = await provider.state("scene.movie")

    assert router.sent_json(0) == {"entity_id": "scene.movie"}
    assert device.kind == "scene" and device.name == "Movie"


async def test_invalid_token_is_an_auth_error(
    router: Router, token_source: StaticTokenSource
) -> None:
    router.add("GET", "/api/states", status=401, text="401: Unauthorized")
    provider = HomeAssistantProvider(router.client(), token_source, base_url="http://ha.local")

    with pytest.raises(AuthError):
        await provider.list_devices()

    status = await provider.status()
    assert status.name == "home-assistant" and status.last_error is not None


def test_from_settings_names_every_missing_variable(
    make_settings: Callable[..., Settings],
) -> None:
    http = httpx.AsyncClient(transport=httpx.MockTransport(lambda _r: httpx.Response(599)))

    with pytest.raises(NotConfiguredError) as raised:
        HomeAssistantProvider.from_settings(make_settings(), http)
    assert raised.value.missing == ("HOME_ASSISTANT_URL", "HOME_ASSISTANT_TOKEN")

    with pytest.raises(NotConfiguredError, match="HOME_ASSISTANT_TOKEN"):
        HomeAssistantProvider.from_settings(make_settings(home_assistant_url="http://x"), http)

    provider = HomeAssistantProvider.from_settings(
        make_settings(home_assistant_url="http://x", home_assistant_token="fake"), http
    )
    assert provider.provider_name == "home-assistant"
