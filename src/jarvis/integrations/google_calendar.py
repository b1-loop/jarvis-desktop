"""Google Calendar provider over the Calendar REST API.

Shares the :class:`GoogleAuth` token source with Gmail and, like it, calls
the API through httpx. Only the primary calendar is read and written; that
is where a personal assistant's meetings live, and it keeps the interface
free of a calendar-id parameter the LLM would have to guess.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, date, datetime, time
from typing import Any

import httpx

from jarvis.integrations.base import CalendarEvent, CalendarProvider, ProviderStatus, TimeRange
from jarvis.integrations.rest import RestSession
from jarvis.integrations.tokens import TokenSource

CALENDAR_SCOPES = (
    "https://www.googleapis.com/auth/calendar.readonly",
    "https://www.googleapis.com/auth/calendar.events",
)
BASE_URL = "https://www.googleapis.com/calendar/v3/calendars/primary"

_MAX_RESULTS = 50


class GoogleCalendarProvider(CalendarProvider):
    """Calendar over Google Calendar's primary calendar."""

    provider_name = "google-calendar"
    scopes = CALENDAR_SCOPES

    def __init__(self, http: httpx.AsyncClient, token_source: TokenSource) -> None:
        self._rest = RestSession(
            http, provider=self.provider_name, base_url=BASE_URL, token_source=token_source
        )

    async def status(self) -> ProviderStatus:
        """Session-level status; see :meth:`RestSession.status`."""
        return self._rest.status()

    async def agenda(self, window: TimeRange) -> list[CalendarEvent]:
        """Expanded (non-recurring) instances overlapping ``window``."""
        data = await self._rest.request(
            "GET",
            "/events",
            params={
                "timeMin": window.start.isoformat(),
                "timeMax": window.end.isoformat(),
                "singleEvents": "true",
                "orderBy": "startTime",
                "maxResults": _MAX_RESULTS,
            },
        )
        return [parse_event(item) for item in data.get("items", [])]

    async def create_event(
        self,
        title: str,
        start: datetime,
        end: datetime,
        *,
        description: str = "",
        location: str = "",
        attendees: Sequence[str] = (),
    ) -> CalendarEvent:
        """Insert a timed event on the primary calendar."""
        body = {
            "summary": title,
            "start": {"dateTime": start.isoformat()},
            "end": {"dateTime": end.isoformat()},
            "description": description,
            "location": location,
            "attendees": [{"email": address} for address in attendees],
        }
        data = await self._rest.request("POST", "/events", json=body)
        return parse_event(data)


def parse_event(data: dict[str, Any]) -> CalendarEvent:
    """Map a Calendar ``Event`` resource to the neutral model."""
    all_day = "date" in data.get("start", {})
    return CalendarEvent(
        id=data["id"],
        title=data.get("summary", "(no title)"),
        start=_parse_boundary(data.get("start", {})),
        end=_parse_boundary(data.get("end", {})),
        location=data.get("location", ""),
        description=data.get("description", ""),
        attendees=tuple(a["email"] for a in data.get("attendees", []) if "email" in a),
        all_day=all_day,
    )


def _parse_boundary(boundary: dict[str, Any]) -> datetime:
    """``start``/``end`` objects carry ``dateTime`` (timed) or ``date`` (all-day)."""
    if "dateTime" in boundary:
        return datetime.fromisoformat(boundary["dateTime"])
    return datetime.combine(date.fromisoformat(boundary["date"]), time.min, tzinfo=UTC)


# Design notes:
# - singleEvents=true expands recurring series into instances, which is what
#   "what's on today" and free-slot search need; the series master would show
#   up once with no concrete time.
# - All-day events get midnight UTC boundaries. They are flagged all_day so
#   the free-slot algorithm ignores them, and the exact instant is never shown.
