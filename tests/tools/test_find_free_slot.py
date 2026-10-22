from collections.abc import Callable
from datetime import datetime, timedelta, timezone

import pytest

from jarvis.agent.registry import ToolRegistry
from jarvis.config import Settings
from jarvis.integrations.base import CalendarEvent
from jarvis.integrations.services import IntegrationServices
from jarvis.integrations.stubs import StubCalendarProvider
from jarvis.tools.find_free_slot import FindFreeSlotTool
from tests.fakes import FakeClock
from tests.tools.integration_helpers import BrokenCalendarProvider, at, integration_context


@pytest.fixture
def settings(make_settings: Callable[..., Settings]) -> Settings:
    return make_settings()


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()  # Wednesday 21 October 2026, 15:42:07 at UTC+3


def test_is_discovered_and_read_only() -> None:
    tool = ToolRegistry.discover().get("find_free_slot")

    assert isinstance(tool, FindFreeSlotTool)
    assert not tool.requires_confirmation


async def test_rest_of_today_starts_at_the_next_five_minutes(
    settings: Settings, clock: FakeClock
) -> None:
    services = IntegrationServices(calendar=(StubCalendarProvider(now=clock.now()),))

    result = await FindFreeSlotTool().call(
        {"days_ahead": 1}, integration_context(settings, services, clock=clock)
    )

    assert result.ok
    assert result.spoken == "You're free for 30 minutes today at 3:45 PM."
    assert result.data is not None
    assert result.data["slots"] == [
        {"start": "2026-10-21T15:45+03:00", "end": "2026-10-21T17:00+03:00", "free_minutes": 75}
    ]


async def test_slots_must_be_free_on_every_calendar(settings: Settings, clock: FakeClock) -> None:
    lunch = CalendarEvent("l", "Lunch", at(clock, 12, days=1), at(clock, 13, days=1))
    services = IntegrationServices(
        calendar=(StubCalendarProvider(now=clock.now()), StubCalendarProvider([lunch]))
    )

    result = await FindFreeSlotTool().call(
        {"days_ahead": 2, "duration_minutes": 60, "max_results": 5},
        integration_context(settings, services, clock=clock),
    )

    assert result.data is not None
    assert [slot["start"] for slot in result.data["slots"]] == [
        "2026-10-21T15:45+03:00",
        "2026-10-22T09:45+03:00",  # after the dentist on the first calendar
        "2026-10-22T13:00+03:00",  # after lunch on the second
    ]
    assert result.spoken == (
        "You're free for 60 minutes today at 3:45 PM, tomorrow at 9:45 AM and tomorrow at 1:00 PM."
    )


async def test_weekends_are_skipped_unless_asked(settings: Settings) -> None:
    friday_evening = FakeClock(
        wall=datetime(2026, 10, 23, 18, 0, tzinfo=timezone(timedelta(hours=3)))
    )
    services = IntegrationServices(calendar=(StubCalendarProvider([]),))
    context = integration_context(settings, services, clock=friday_evening)

    weekdays = await FindFreeSlotTool().call({"days_ahead": 3}, context)
    any_day = await FindFreeSlotTool().call({"days_ahead": 3, "include_weekends": True}, context)

    assert weekdays.spoken == "I couldn't find 30 minutes free in that time."
    assert any_day.spoken.startswith("You're free for 30 minutes tomorrow at 9:00 AM")


async def test_custom_working_hours(settings: Settings, clock: FakeClock) -> None:
    services = IntegrationServices(calendar=(StubCalendarProvider([]),))

    result = await FindFreeSlotTool().call(
        {"days_ahead": 2, "working_hours_start": "07:30", "working_hours_end": "08:30"},
        integration_context(settings, services, clock=clock),
    )

    assert result.spoken == "You're free for 30 minutes tomorrow at 7:30 AM."


async def test_a_failing_calendar_is_named(settings: Settings, clock: FakeClock) -> None:
    services = IntegrationServices(calendar=(StubCalendarProvider([]), BrokenCalendarProvider([])))

    result = await FindFreeSlotTool().call({}, integration_context(settings, services, clock=clock))

    assert result.ok
    assert "broken-calendar" in result.detail
    assert "availability on those calendars is unknown" in result.detail


async def test_working_hours_must_be_ordered(settings: Settings, clock: FakeClock) -> None:
    services = IntegrationServices(calendar=(StubCalendarProvider([]),))

    result = await FindFreeSlotTool().call(
        {"working_hours_start": "17:00", "working_hours_end": "09:00"},
        integration_context(settings, services, clock=clock),
    )

    assert not result.ok
    assert "working_hours_end must be after working_hours_start" in result.detail
