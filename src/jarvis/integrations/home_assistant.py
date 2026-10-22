"""Home Assistant provider over its REST API with a long-lived access token.

Philips Hue (and every other brand) appears here as ordinary ``light.*``
entities once the Hue integration is enabled in Home Assistant, so there is
no separate Hue provider: one hub-neutral interface, one implementation,
and the user's bridge credentials stay inside Home Assistant.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import httpx

from jarvis.integrations.base import (
    Device,
    NotConfiguredError,
    ProviderStatus,
    SmartHomeProvider,
)
from jarvis.integrations.rest import RestSession
from jarvis.integrations.tokens import StaticTokenSource, TokenSource

if TYPE_CHECKING:
    from jarvis.config import Settings

PROVIDER = "home-assistant"

CONTROLLABLE_KINDS = frozenset(
    {"light", "switch", "scene", "fan", "cover", "media_player", "climate"}
)
_MAX_BRIGHTNESS = 255
_PERCENT = 100


class HomeAssistantProvider(SmartHomeProvider):
    """Entities and services of one Home Assistant instance."""

    provider_name = PROVIDER

    def __init__(
        self, http: httpx.AsyncClient, token_source: TokenSource, *, base_url: str
    ) -> None:
        self._rest = RestSession(
            http,
            provider=self.provider_name,
            base_url=f"{base_url.rstrip('/')}/api",
            token_source=token_source,
        )

    @classmethod
    def from_settings(cls, settings: Settings, http: httpx.AsyncClient) -> HomeAssistantProvider:
        """Build from the instance URL and long-lived token.

        Raises:
            NotConfiguredError: If either variable is unset.
        """
        missing = [
            name
            for name, value in (
                ("HOME_ASSISTANT_URL", settings.home_assistant_url),
                ("HOME_ASSISTANT_TOKEN", settings.home_assistant_token),
            )
            if value is None
        ]
        if missing or settings.home_assistant_url is None or settings.home_assistant_token is None:
            raise NotConfiguredError(PROVIDER, missing)
        token_source = StaticTokenSource(settings.home_assistant_token)
        return cls(http, token_source, base_url=settings.home_assistant_url)

    async def status(self) -> ProviderStatus:
        """Session-level status; see :meth:`RestSession.status`."""
        return self._rest.status()

    async def list_devices(self, kind: str | None = None) -> list[Device]:
        """Controllable entities, or every entity of ``kind`` (e.g. ``sensor``)."""
        states = await self._rest.request("GET", "/states")
        devices = [parse_state(item) for item in states]
        if kind is not None:
            return [d for d in devices if d.kind == kind]
        return [d for d in devices if d.kind in CONTROLLABLE_KINDS]

    async def state(self, device_id: str) -> Device:
        """Current state of one entity."""
        return parse_state(await self._rest.request("GET", f"/states/{device_id}"))

    async def turn_on(self, device_id: str) -> Device:
        """Call ``<domain>.turn_on`` and read the state back."""
        await self._service(_domain(device_id), "turn_on", device_id)
        return await self.state(device_id)

    async def turn_off(self, device_id: str) -> Device:
        """Call ``<domain>.turn_off`` and read the state back."""
        await self._service(_domain(device_id), "turn_off", device_id)
        return await self.state(device_id)

    async def set_brightness(self, device_id: str, percent: int) -> Device:
        """Dim a light; zero turns it off.

        Raises:
            ValueError: If ``percent`` is outside 0-100.
        """
        if not 0 <= percent <= _PERCENT:
            raise ValueError(f"brightness must be 0-100, got {percent}")
        if percent == 0:
            return await self.turn_off(device_id)
        await self._service("light", "turn_on", device_id, brightness_pct=percent)
        return await self.state(device_id)

    async def set_scene(self, scene_id: str) -> None:
        """Activate ``scene.<name>``."""
        await self._service("scene", "turn_on", scene_id)

    async def _service(self, domain: str, service: str, entity_id: str, **data: Any) -> None:
        await self._rest.request(
            "POST", f"/services/{domain}/{service}", json={"entity_id": entity_id, **data}
        )


def parse_state(data: dict[str, Any]) -> Device:
    """Map a Home Assistant state object to the neutral model."""
    entity_id = data["entity_id"]
    attributes = dict(data.get("attributes", {}))
    kind = _domain(entity_id)
    return Device(
        id=entity_id,
        name=str(attributes.pop("friendly_name", entity_id)),
        kind=kind,
        state=str(data.get("state", "unknown")),
        brightness=_brightness_percent(kind, data.get("state"), attributes.get("brightness")),
        attributes=attributes,
    )


def _brightness_percent(kind: str, state: Any, raw: Any) -> int | None:
    if kind != "light":
        return None
    if state == "off":
        return 0
    return round(int(raw) * _PERCENT / _MAX_BRIGHTNESS) if raw is not None else None


def _domain(entity_id: str) -> str:
    return entity_id.split(".", 1)[0]


# Design notes:
# - Service calls return the list of changed states, but not reliably for
#   every integration, so the entity is re-read afterwards. Two requests to a
#   LAN device are still well under the spoken-confirmation latency.
# - list_devices hides sensors by default: a typical install has hundreds of
#   them and the LLM only needs things it can act on, unless asked for a kind.
