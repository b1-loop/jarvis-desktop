from datetime import UTC, datetime, time, timedelta

import pytest

from jarvis.integrations.base import (
    AuthError,
    CalendarEvent,
    Device,
    IntegrationError,
    NotConfiguredError,
    ProviderError,
    RateLimitedError,
    TimeRange,
    WorkingHours,
)
from jarvis.integrations.stubs import StubCalendarProvider
from jarvis.utils.errors import JarvisError

# Wednesday
NOW = datetime(2026, 10, 21, 8, 30, tzinfo=UTC)
DAY = NOW.replace(hour=0, minute=0)


def test_errors_derive_from_jarvis_error_and_carry_provider() -> None:
    error = ProviderError("gmail", "boom", status_code=500)

    assert isinstance(error, IntegrationError)
    assert isinstance(error, JarvisError)
    assert error.provider == "gmail"
    assert error.status_code == 500
    assert str(error) == "gmail: boom"


def test_not_configured_error_names_the_variables() -> None:
    error = NotConfiguredError("spotify", ["SPOTIFY_CLIENT_ID", "SPOTIFY_CLIENT_SECRET"])

    assert error.missing == ("SPOTIFY_CLIENT_ID", "SPOTIFY_CLIENT_SECRET")
    assert "SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET" in str(error)
    assert ".env" in str(error)


def test_rate_limited_and_auth_errors() -> None:
    assert RateLimitedError("slack", "slow down", retry_after=2.5).retry_after == 2.5
    assert RateLimitedError("slack", "slow down").retry_after is None
    assert isinstance(AuthError("x", "y"), IntegrationError)


def test_time_range_rejects_empty_or_inverted_ranges() -> None:
    with pytest.raises(ValueError, match="end must be after start"):
        TimeRange(NOW, NOW)
    with pytest.raises(ValueError, match="end must be after start"):
        TimeRange(NOW, NOW - timedelta(minutes=1))


def test_time_range_overlap_is_half_open() -> None:
    first = TimeRange(NOW, NOW + timedelta(hours=1))
    adjacent = TimeRange(NOW + timedelta(hours=1), NOW + timedelta(hours=2))
    inside = TimeRange(NOW + timedelta(minutes=30), NOW + timedelta(minutes=40))

    assert not first.overlaps(adjacent)
    assert first.overlaps(inside)
    assert inside.overlaps(first)
    assert first.duration == timedelta(hours=1)


def test_device_is_on() -> None:
    assert Device("light.a", "A", "light", "on").is_on
    assert not Device("light.a", "A", "light", "off").is_on


async def test_today_uses_the_calendar_day_of_now() -> None:
    provider = StubCalendarProvider(now=NOW)

    titles = [event.title for event in await provider.today(NOW)]

    assert titles == ["Sprint planning", "1:1 with Maya"]
    window = provider.calls[0][1]["window"]
    assert window == TimeRange(DAY, DAY + timedelta(days=1))


async def test_upcoming_only_returns_events_that_have_not_started() -> None:
    provider = StubCalendarProvider(now=NOW)

    soon = await provider.upcoming(timedelta(hours=2), now=DAY + timedelta(hours=10, minutes=30))

    assert [e.title for e in soon] == []
    later = await provider.upcoming(timedelta(hours=6), now=DAY + timedelta(hours=9))
    assert [e.title for e in later] == ["Sprint planning", "1:1 with Maya"]


async def test_free_slots_skip_busy_blocks_and_respect_working_hours() -> None:
    provider = StubCalendarProvider(now=NOW)
    window = TimeRange(DAY, DAY + timedelta(days=1))

    slots = await provider.find_free_slots(timedelta(hours=1), window)

    assert slots == [
        TimeRange(DAY + timedelta(hours=9), DAY + timedelta(hours=10)),
        TimeRange(DAY + timedelta(hours=11), DAY + timedelta(hours=14)),
        TimeRange(DAY + timedelta(hours=14, minutes=30), DAY + timedelta(hours=17)),
    ]


async def test_free_slots_ignore_all_day_events_and_weekends() -> None:
    saturday = DAY + timedelta(days=3)
    events = [
        CalendarEvent("all", "Birthday", DAY, DAY + timedelta(days=1), all_day=True),
        CalendarEvent(
            "sat", "Hack day", saturday + timedelta(hours=10), saturday + timedelta(hours=12)
        ),
    ]
    provider = StubCalendarProvider(events)
    hours = WorkingHours(start=time(10), end=time(12))

    weekday_slots = await provider.find_free_slots(
        timedelta(hours=2), TimeRange(DAY, DAY + timedelta(days=1)), hours
    )
    weekend_slots = await provider.find_free_slots(
        timedelta(hours=1), TimeRange(saturday, saturday + timedelta(days=1)), hours
    )
    weekend_allowed = await provider.find_free_slots(
        timedelta(hours=1),
        TimeRange(saturday, saturday + timedelta(days=1)),
        WorkingHours(start=time(8), end=time(14), weekdays_only=False),
    )

    assert weekday_slots == [TimeRange(DAY + timedelta(hours=10), DAY + timedelta(hours=12))]
    assert weekend_slots == []
    assert weekend_allowed == [
        TimeRange(saturday + timedelta(hours=8), saturday + timedelta(hours=10)),
        TimeRange(saturday + timedelta(hours=12), saturday + timedelta(hours=14)),
    ]


async def test_free_slots_clip_to_the_window() -> None:
    provider = StubCalendarProvider([])
    window = TimeRange(DAY + timedelta(hours=16), DAY + timedelta(hours=20))

    slots = await provider.find_free_slots(timedelta(minutes=30), window)

    assert slots == [TimeRange(DAY + timedelta(hours=16), DAY + timedelta(hours=17))]
