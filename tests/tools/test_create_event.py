from collections.abc import Callable

import pytest

from jarvis.agent.confirmation import DECLINED_SPOKEN, ConfirmationPolicy
from jarvis.agent.registry import ToolRegistry
from jarvis.config import Settings
from jarvis.integrations.services import IntegrationServices
from jarvis.integrations.stubs import StubCalendarProvider
from jarvis.tools.create_event import CreateEventInput, CreateEventTool
from tests.agent.fakes import FakeConfirmation
from tests.fakes import FakeClock
from tests.tools.integration_helpers import at, integration_context

REQUEST = {
    "title": "Design review",
    "start": "2026-10-22T14:00",
    "duration_minutes": 60,
    "attendees": ["maya@example.com", "tom@example.com"],
    "location": "Room 3B",
}


@pytest.fixture
def settings(make_settings: Callable[..., Settings]) -> Settings:
    return make_settings()


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()  # Wednesday 21 October 2026, 15:42 at UTC+3


@pytest.fixture
def calendar() -> StubCalendarProvider:
    return StubCalendarProvider([])


def test_is_discovered_and_always_confirms() -> None:
    tool = ToolRegistry.discover().get("create_event")

    assert isinstance(tool, CreateEventTool)
    assert tool.requires_confirmation


def test_prompt_reads_back_title_time_and_attendees() -> None:
    params = CreateEventInput.model_validate(REQUEST)

    assert CreateEventTool().confirmation_prompt(params) == (
        'Create "Design review" on Thursday 22 October at 2:00 PM for 60 minutes '
        "with maya@example.com and tom@example.com?"
    )


async def test_confirmed_event_is_created_in_local_time(
    settings: Settings, clock: FakeClock, calendar: StubCalendarProvider
) -> None:
    asker = FakeConfirmation(answers=[True])
    context = integration_context(
        settings, IntegrationServices(calendar=(calendar,)), clock=clock, confirmation=asker
    )

    result = await ConfirmationPolicy().execute_with_policy(CreateEventTool(), REQUEST, context)

    assert result.ok
    assert result.spoken == (
        "Done: Design review tomorrow at 2:00 PM, "
        "with maya@example.com and tom@example.com invited."
    )
    (event,) = calendar.events
    assert event.start == at(clock, 14, days=1)
    assert event.end == at(clock, 15, days=1)
    assert event.attendees == ("maya@example.com", "tom@example.com")
    assert event.location == "Room 3B"
    assert result.data is not None
    assert result.data["event"]["id"] == "stub-calendar:stub-event-1"


async def test_declined_event_is_not_created(
    settings: Settings, clock: FakeClock, calendar: StubCalendarProvider
) -> None:
    context = integration_context(
        settings,
        IntegrationServices(calendar=(calendar,)),
        clock=clock,
        confirmation=FakeConfirmation(answers=[False]),
    )

    result = await ConfirmationPolicy().execute_with_policy(CreateEventTool(), REQUEST, context)

    assert result.spoken == DECLINED_SPOKEN
    assert calendar.events == []


async def test_past_start_is_refused(
    settings: Settings, clock: FakeClock, calendar: StubCalendarProvider
) -> None:
    context = integration_context(settings, IntegrationServices(calendar=(calendar,)), clock=clock)

    result = await CreateEventTool().call(REQUEST | {"start": "2026-10-21T09:00+03:00"}, context)

    assert not result.ok
    assert "in the past" in result.spoken
    assert calendar.events == []


async def test_a_named_calendar_is_used(settings: Settings, clock: FakeClock) -> None:
    first, second = StubCalendarProvider([]), StubCalendarProvider([])
    second.provider_name = "outlook"
    context = integration_context(
        settings, IntegrationServices(calendar=(first, second)), clock=clock
    )

    created = await CreateEventTool().call(REQUEST | {"calendar": "outlook"}, context)
    unknown = await CreateEventTool().call(REQUEST | {"calendar": "icloud"}, context)

    assert created.ok
    assert first.events == []
    assert len(second.events) == 1
    assert not unknown.ok
    assert "available: stub-calendar, outlook" in unknown.detail


async def test_attendees_must_be_email_addresses(
    settings: Settings, calendar: StubCalendarProvider
) -> None:
    context = integration_context(settings, IntegrationServices(calendar=(calendar,)))

    result = await CreateEventTool().call(REQUEST | {"attendees": ["Maya"]}, context)

    assert not result.ok
    assert "attendees.0" in result.detail
