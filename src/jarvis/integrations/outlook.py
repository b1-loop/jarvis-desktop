"""Outlook mail and calendar over Microsoft Graph.

One class implements both :class:`EmailProvider` and :class:`CalendarProvider`
because they share a token, a base URL and an error mapping; splitting them
would only duplicate the plumbing. Scopes are the smallest set that covers
the feature list: ``Mail.ReadWrite``, ``Mail.Send`` and ``Calendars.ReadWrite``.
``Mail.ReadWrite`` rather than ``Mail.Read`` because Graph's ``createReply``,
which creates the reply draft the user approves, requires it.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Any

import httpx

from jarvis.integrations.base import (
    CalendarEvent,
    CalendarProvider,
    EmailDraft,
    EmailMessage,
    EmailProvider,
    ProviderStatus,
    TimeRange,
)
from jarvis.integrations.microsoft_graph import GRAPH_BASE_URL, parse_graph_datetime
from jarvis.integrations.rest import RestSession
from jarvis.integrations.tokens import TokenSource

OUTLOOK_SCOPES = ("Mail.ReadWrite", "Mail.Send", "Calendars.ReadWrite")

_MESSAGE_FIELDS = "id,conversationId,from,toRecipients,subject,bodyPreview,receivedDateTime,isRead"
_TEXT_BODY = {"Prefer": 'outlook.body-content-type="text"'}
_UTC_TIMES = {"Prefer": 'outlook.timezone="UTC"'}
_MAX_EVENTS = 50


class OutlookProvider(EmailProvider, CalendarProvider):
    """Email and calendar for a Microsoft account."""

    provider_name = "outlook"
    scopes = OUTLOOK_SCOPES

    def __init__(self, http: httpx.AsyncClient, token_source: TokenSource) -> None:
        self._rest = RestSession(
            http, provider=self.provider_name, base_url=GRAPH_BASE_URL, token_source=token_source
        )

    async def status(self) -> ProviderStatus:
        """Session-level status; see :meth:`RestSession.status`."""
        return self._rest.status()

    # -- email --------------------------------------------------------------

    async def list_unread(self, limit: int = 10) -> list[EmailMessage]:
        """Unread inbox messages, newest first."""
        data = await self._rest.request(
            "GET",
            "/me/mailFolders/inbox/messages",
            params={
                "$filter": "isRead eq false",
                "$orderby": "receivedDateTime desc",
                "$top": limit,
                "$select": _MESSAGE_FIELDS,
            },
            headers=_TEXT_BODY,
        )
        return [parse_message(item) for item in data.get("value", [])]

    async def search(self, query: str, limit: int = 10) -> list[EmailMessage]:
        """Graph ``$search`` (KQL), e.g. ``from:maya`` or free text."""
        data = await self._rest.request(
            "GET",
            "/me/messages",
            params={"$search": f'"{query}"', "$top": limit, "$select": _MESSAGE_FIELDS},
            headers=_TEXT_BODY,
        )
        return [parse_message(item) for item in data.get("value", [])]

    async def get(self, message_id: str) -> EmailMessage:
        """Full message with its body as plain text."""
        data = await self._rest.request("GET", f"/me/messages/{message_id}", headers=_TEXT_BODY)
        return parse_message(data)

    async def draft_reply(self, message: EmailMessage, body: str) -> EmailDraft:
        """Ask Graph to create the reply draft; it sets recipients and threading."""
        data = await self._rest.request(
            "POST", f"/me/messages/{message.id}/createReply", json={"comment": body}
        )
        return EmailDraft(
            id=data["id"],
            to=tuple(_addresses(data.get("toRecipients", []))),
            subject=data.get("subject", ""),
            body=body,
            in_reply_to=message.id,
            thread_id=data.get("conversationId", message.thread_id),
        )

    async def send(self, draft: EmailDraft) -> str:
        """Send the draft. Graph returns no body, so the draft id is returned."""
        await self._rest.request("POST", f"/me/messages/{draft.id}/send")
        return draft.id

    # -- calendar -----------------------------------------------------------

    async def agenda(self, window: TimeRange) -> list[CalendarEvent]:
        """Expanded instances in ``window`` via ``calendarView``, in UTC."""
        data = await self._rest.request(
            "GET",
            "/me/calendarView",
            params={
                "startDateTime": window.start.isoformat(),
                "endDateTime": window.end.isoformat(),
                "$orderby": "start/dateTime",
                "$top": _MAX_EVENTS,
            },
            headers=_UTC_TIMES,
        )
        return [parse_event(item) for item in data.get("value", [])]

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
        """Create an event on the default calendar."""
        body = {
            "subject": title,
            "body": {"contentType": "text", "content": description},
            "start": {"dateTime": start.isoformat(), "timeZone": "UTC"},
            "end": {"dateTime": end.isoformat(), "timeZone": "UTC"},
            "location": {"displayName": location},
            "attendees": [
                {"emailAddress": {"address": address}, "type": "required"} for address in attendees
            ],
        }
        data = await self._rest.request("POST", "/me/events", json=body, headers=_UTC_TIMES)
        return parse_event(data)


def parse_message(data: dict[str, Any]) -> EmailMessage:
    """Map a Graph ``message`` resource to the neutral model."""
    sender = data.get("from", {}).get("emailAddress", {})
    body = data.get("body", {})
    return EmailMessage(
        id=data["id"],
        thread_id=data.get("conversationId", data["id"]),
        sender=_format_address(sender),
        recipients=tuple(_addresses(data.get("toRecipients", []))),
        subject=data.get("subject", ""),
        snippet=data.get("bodyPreview", ""),
        received_at=parse_graph_datetime(data["receivedDateTime"]),
        unread=not data.get("isRead", False),
        body=body.get("content", "") if body.get("contentType", "text") == "text" else "",
    )


def parse_event(data: dict[str, Any]) -> CalendarEvent:
    """Map a Graph ``event`` resource to the neutral model."""
    return CalendarEvent(
        id=data["id"],
        title=data.get("subject", "(no title)"),
        start=parse_graph_datetime(data["start"]["dateTime"]),
        end=parse_graph_datetime(data["end"]["dateTime"]),
        location=data.get("location", {}).get("displayName", ""),
        description=data.get("bodyPreview", ""),
        attendees=tuple(
            a["emailAddress"]["address"] for a in data.get("attendees", []) if "emailAddress" in a
        ),
        all_day=bool(data.get("isAllDay", False)),
    )


def _addresses(recipients: list[dict[str, Any]]) -> list[str]:
    return [_format_address(r.get("emailAddress", {})) for r in recipients]


def _format_address(email_address: dict[str, Any]) -> str:
    name, address = email_address.get("name"), email_address.get("address", "")
    return f"{name} <{address}>" if name else address


# Design notes:
# - createReply does the threading (conversationId, In-Reply-To, quoted
#   original) on the server, so unlike Gmail no MIME is built here.
# - Prefer headers pin the representation: text bodies instead of HTML and
#   UTC event times, which keeps the parser free of time-zone names.
# - Graph's send endpoint returns 202 with no body; the draft id is returned
#   because it is the only identifier available without another lookup.
