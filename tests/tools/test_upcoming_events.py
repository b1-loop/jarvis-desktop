from collections.abc import Callable
from datetime import timedelta

import pytest

from jarvis.agent.registry import ToolRegistry
from jarvis.config import Settings
from jarvis.integrations.base import CalendarEvent
from jarvis.integrations.services import IntegrationServices
from jarvis.integrations.stubs import StubCalendarProvider
from jarvis.proactive import triggers
from jarvis.proactive.triggers import UpcomingMeetingTrigger
from jarvis.tools.upcoming_events import CalendarTriggerSource, UpcomingEventsTool
from tests.fakes import FakeClock
from tests.tools.integration_helpers import at, integration_context


@pytest.fixture
def settings(make_settings: Callable[..., Settings]) -> Settings:
    return make_settings()


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()  # Wednesday 21 October 2026, 15:42 at UTC+3


def test_is_discovered_and_read_only() -> None:
    tool = ToolRegistry.discover().get("upcoming_events")

    assert isinstance(tool, UpcomingEventsTool)
    assert not tool.requires_confirmation


async def test_nothing_in_the_default_window(settings: Settings, clock: FakeClock) -> None:
    services = IntegrationServices(calendar=(StubCalendarProvider(now=clock.now()),))

    result = await UpcomingEventsTool().call(
        {}, integration_context(settings, services, clock=clock)
    )

    assert result.ok
    assert result.spoken == "Nothing on your calendar in the next 240 minutes."


async def test_describes_the_next_event(settings: Settings, clock: FakeClock) -> None:
    extra = CalendarEvent("x", "Gym", at(clock, 19), at(clock, 20))
    calendar = StubCalendarProvider([*StubCalendarProvider(now=clock.now()).events, extra])
    services = IntegrationServices(calendar=(calendar,))

    result = await UpcomingEventsTool().call(
        {"within_minutes": 24 * 60}, integration_context(settings, services, clock=clock)
    )

    assert result.spoken == "Next up: Gym today at 7:00 PM. After that, 1 more."
    assert result.data is not None
    assert [e["title"] for e in result.data["events"]] == ["Gym", "Dentist"]


async def test_trigger_source_skips_all_day_events_and_uses_references(
    clock: FakeClock,
) -> None:
    events = [
        CalendarEvent("m1", "Standup", at(clock, 15, 50), at(clock, 16), location="Room 1"),
        CalendarEvent("a1", "Holiday", at(clock, 16), at(clock, 0, days=1), all_day=True),
    ]
    source = CalendarTriggerSource([StubCalendarProvider(events)], clock=clock)

    upcoming = await source.upcoming_events(timedelta(minutes=30))

    assert upcoming == [
        triggers.CalendarEvent(
            id="stub-calendar:m1", title="Standup", start=at(clock, 15, 50), location="Room 1"
        )
    ]


async def test_trigger_source_drives_the_meeting_trigger(clock: FakeClock) -> None:
    events = [CalendarEvent("m1", "Standup", at(clock, 15, 50), at(clock, 16))]
    source = CalendarTriggerSource([StubCalendarProvider(events)], clock=clock)

    notification = await UpcomingMeetingTrigger(source, clock=clock).check()

    assert notification is not None
    assert notification.spoken_text == "Heads up: Standup starts in 8 minutes."


def test_trigger_source_defaults_to_the_system_clock() -> None:
    assert CalendarTriggerSource([])._clock is not None
