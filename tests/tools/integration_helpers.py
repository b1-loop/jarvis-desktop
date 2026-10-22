"""Contexts and failing providers for the integration tool tests."""

from __future__ import annotations

from datetime import datetime, timedelta

from jarvis.agent.tool import ToolContext
from jarvis.config import Settings
from jarvis.core.events import EventBus
from jarvis.integrations.base import (
    CalendarEvent,
    EmailMessage,
    ProviderError,
    TimeRange,
)
from jarvis.integrations.services import IntegrationServices
from jarvis.integrations.stubs import StubCalendarProvider, StubEmailProvider
from tests.agent.fakes import FakeConfirmation
from tests.fakes import FakeClock


def integration_context(
    settings: Settings,
    services: IntegrationServices | None,
    *,
    clock: FakeClock | None = None,
    confirmation: FakeConfirmation | None = None,
) -> ToolContext:
    """A ``ToolContext`` with fakes everywhere and ``services`` as integrations."""
    return ToolContext(
        settings=settings,
        bus=EventBus(),
        ask_confirmation=confirmation or FakeConfirmation(),
        clock=clock or FakeClock(),
        integrations=services,
    )


class BrokenEmailProvider(StubEmailProvider):
    """An email provider whose every call fails like an outage would."""

    provider_name = "broken-email"

    async def list_unread(self, limit: int = 10) -> list[EmailMessage]:
        raise ProviderError(self.provider_name, "503 service unavailable", status_code=503)

    async def search(self, query: str, limit: int = 10) -> list[EmailMessage]:
        raise ProviderError(self.provider_name, "503 service unavailable", status_code=503)


class BodylessEmailProvider(StubEmailProvider):
    """Lists messages fine but cannot fetch a full body."""

    provider_name = "bodyless-email"

    async def get(self, message_id: str) -> EmailMessage:
        raise ProviderError(self.provider_name, "body unavailable")


class BrokenCalendarProvider(StubCalendarProvider):
    """A calendar whose agenda call always fails."""

    provider_name = "broken-calendar"

    async def agenda(self, window: TimeRange) -> list[CalendarEvent]:
        raise ProviderError(self.provider_name, "timeout")


def at(clock: FakeClock, hour: int, minute: int = 0, days: int = 0) -> datetime:
    """A time on the fake clock's day (plus ``days``) in its zone."""
    base = clock.now().replace(hour=hour, minute=minute, second=0, microsecond=0)
    return base + timedelta(days=days)
