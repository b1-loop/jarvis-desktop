from datetime import UTC, datetime, timedelta

import pytest

from jarvis.integrations.base import TimeRange
from jarvis.integrations.google_calendar import CALENDAR_SCOPES, GoogleCalendarProvider
from jarvis.integrations.tokens import StaticTokenSource
from tests.integrations.conftest import Router

EVENTS = "/calendar/v3/calendars/primary/events"
NOW = datetime(2026, 10, 21, 8, 0, tzinfo=UTC)


@pytest.fixture
def provider(router: Router, token_source: StaticTokenSource) -> GoogleCalendarProvider:
    return GoogleCalendarProvider(router.client(), token_source)


def test_scopes_are_read_plus_events_only() -> None:
    assert GoogleCalendarProvider.scopes == CALENDAR_SCOPES
    assert all(s.endswith(("calendar.readonly", "calendar.events")) for s in CALENDAR_SCOPES)


async def test_agenda_requests_expanded_instances_and_parses_both_kinds(
    router: Router, provider: GoogleCalendarProvider
) -> None:
    router.add(
        "GET",
        EVENTS,
        {
            "items": [
                {
                    "id": "e1",
                    "summary": "Sprint planning",
                    "start": {"dateTime": "2026-10-21T10:00:00+02:00"},
                    "end": {"dateTime": "2026-10-21T11:00:00+02:00"},
                    "location": "Room 3B",
                    "description": "Bring estimates",
                    "attendees": [{"email": "maya@example.com"}, {"displayName": "no email"}],
                },
                {
                    "id": "e2",
                    "start": {"date": "2026-10-22"},
                    "end": {"date": "2026-10-23"},
                },
            ]
        },
    )

    events = await provider.agenda(TimeRange(NOW, NOW + timedelta(days=2)))

    params = router.last().url.params
    assert params["timeMin"] == NOW.isoformat()
    assert params["singleEvents"] == "true"
    assert params["orderBy"] == "startTime"
    timed, all_day = events
    assert timed.title == "Sprint planning"
    assert timed.start == datetime.fromisoformat("2026-10-21T10:00:00+02:00")
    assert timed.attendees == ("maya@example.com",)
    assert timed.location == "Room 3B" and timed.description == "Bring estimates"
    assert not timed.all_day
    assert all_day.all_day and all_day.title == "(no title)"
    assert all_day.start == datetime(2026, 10, 22, tzinfo=UTC)
    assert all_day.end == datetime(2026, 10, 23, tzinfo=UTC)


async def test_create_event_posts_the_expected_body(
    router: Router, provider: GoogleCalendarProvider
) -> None:
    router.add(
        "POST",
        EVENTS,
        {
            "id": "new",
            "summary": "Coffee",
            "start": {"dateTime": "2026-10-21T15:00:00+00:00"},
            "end": {"dateTime": "2026-10-21T15:30:00+00:00"},
        },
    )

    event = await provider.create_event(
        "Coffee",
        NOW.replace(hour=15),
        NOW.replace(hour=15, minute=30),
        location="Cafe",
        attendees=["tom@example.com"],
    )

    assert router.sent_json() == {
        "summary": "Coffee",
        "start": {"dateTime": "2026-10-21T15:00:00+00:00"},
        "end": {"dateTime": "2026-10-21T15:30:00+00:00"},
        "description": "",
        "location": "Cafe",
        "attendees": [{"email": "tom@example.com"}],
    }
    assert event.id == "new" and event.title == "Coffee"
    assert (await provider.status()).authenticated
