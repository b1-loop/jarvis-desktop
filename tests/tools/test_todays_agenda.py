from collections.abc import Callable

import pytest

from jarvis.agent.registry import ToolRegistry
from jarvis.config import Settings
from jarvis.integrations.base import CalendarEvent
from jarvis.integrations.services import IntegrationServices
from jarvis.integrations.stubs import StubCalendarProvider
from jarvis.tools.todays_agenda import TodaysAgendaTool
from tests.fakes import FakeClock
from tests.tools.integration_helpers import BrokenCalendarProvider, at, integration_context


@pytest.fixture
def settings(make_settings: Callable[..., Settings]) -> Settings:
    return make_settings()


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()  # Wednesday 21 October 2026, 15:42 at UTC+3


def test_is_discovered_and_read_only() -> None:
    tool = ToolRegistry.discover().get("todays_agenda")

    assert isinstance(tool, TodaysAgendaTool)
    assert not tool.requires_confirmation


async def test_reads_todays_events_in_order(settings: Settings, clock: FakeClock) -> None:
    services = IntegrationServices(calendar=(StubCalendarProvider(now=clock.now()),))

    result = await TodaysAgendaTool().call({}, integration_context(settings, services, clock=clock))

    assert result.ok
    assert result.spoken == (
        "You have 2 events today: Sprint planning at 10:00 AM and 1:1 with Maya at 2:00 PM."
    )
    assert "Agenda for Wednesday 21 October 2026:" in result.detail
    assert "- Wed 21 Oct 10:00-11:00: Sprint planning @ Room 3B with maya@example.com" in (
        result.detail
    )
    assert result.data is not None
    assert result.data["date"] == "2026-10-21"
    assert [e["id"] for e in result.data["events"]] == [
        "stub-calendar:stub-event-1",
        "stub-calendar:stub-event-2",
    ]


async def test_tomorrow_and_a_clear_day(settings: Settings, clock: FakeClock) -> None:
    services = IntegrationServices(calendar=(StubCalendarProvider(now=clock.now()),))
    context = integration_context(settings, services, clock=clock)

    tomorrow = await TodaysAgendaTool().call({"days_from_today": 1}, context)
    monday = await TodaysAgendaTool().call({"days_from_today": 5}, context)

    assert tomorrow.spoken == "You have 1 event tomorrow: Dentist at 9:00 AM."
    assert monday.spoken == "Your calendar is clear on Monday."


async def test_merges_calendars_and_speaks_all_day_events_without_a_time(
    settings: Settings, clock: FakeClock
) -> None:
    birthday = CalendarEvent(
        "b1", "Tom's birthday", at(clock, 0), at(clock, 0, days=1), all_day=True
    )
    services = IntegrationServices(
        calendar=(
            StubCalendarProvider(now=clock.now()),
            StubCalendarProvider([birthday]),
            BrokenCalendarProvider([]),
        )
    )

    result = await TodaysAgendaTool().call({}, integration_context(settings, services, clock=clock))

    assert result.spoken.startswith("You have 3 events today: Tom's birthday, Sprint planning")
    assert "Wed 21 Oct all day: Tom's birthday" in result.detail
    assert "broken-calendar: timeout" in result.detail


async def test_more_than_four_events_are_summarised(settings: Settings, clock: FakeClock) -> None:
    events = [
        CalendarEvent(f"e{h}", f"Block {h}", at(clock, h), at(clock, h, 30)) for h in range(8, 14)
    ]
    services = IntegrationServices(calendar=(StubCalendarProvider(events),))

    result = await TodaysAgendaTool().call({}, integration_context(settings, services, clock=clock))

    assert result.spoken.endswith("Block 11 at 11:00 AM, and 2 more.")
