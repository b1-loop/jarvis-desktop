from datetime import UTC, datetime, timedelta

import pytest

from jarvis.integrations.base import CalendarProvider, EmailProvider, ProviderError, TimeRange
from jarvis.integrations.outlook import OUTLOOK_SCOPES, OutlookProvider, parse_message
from jarvis.integrations.tokens import StaticTokenSource
from tests.integrations.conftest import Router

NOW = datetime(2026, 10, 21, 8, 0, tzinfo=UTC)

MESSAGE = {
    "id": "AAMk1",
    "conversationId": "conv-1",
    "from": {"emailAddress": {"name": "Maya", "address": "maya@example.com"}},
    "toRecipients": [{"emailAddress": {"address": "you@example.com"}}],
    "subject": "Design review",
    "bodyPreview": "Can we move it?",
    "receivedDateTime": "2026-10-21T07:48:00.1234567Z",
    "isRead": False,
    "body": {"contentType": "text", "content": "Can we move it?\n\nMaya"},
}

EVENT = {
    "id": "ev1",
    "subject": "Sprint planning",
    "start": {"dateTime": "2026-10-21T10:00:00.0000000", "timeZone": "UTC"},
    "end": {"dateTime": "2026-10-21T11:00:00.0000000", "timeZone": "UTC"},
    "location": {"displayName": "Room 3B"},
    "bodyPreview": "Bring estimates",
    "attendees": [{"emailAddress": {"address": "maya@example.com"}}],
    "isAllDay": False,
}


@pytest.fixture
def provider(router: Router, token_source: StaticTokenSource) -> OutlookProvider:
    return OutlookProvider(router.client(), token_source)


def test_outlook_is_both_an_email_and_a_calendar_provider(provider: OutlookProvider) -> None:
    assert isinstance(provider, EmailProvider)
    assert isinstance(provider, CalendarProvider)
    assert (
        OutlookProvider.scopes
        == OUTLOOK_SCOPES
        == ("Mail.ReadWrite", "Mail.Send", "Calendars.ReadWrite")
    )


async def test_list_unread_filters_and_orders_on_the_server(
    router: Router, provider: OutlookProvider
) -> None:
    router.add("GET", "/v1.0/me/mailFolders/inbox/messages", {"value": [MESSAGE]})

    messages = await provider.list_unread(limit=5)

    request = router.last()
    assert request.url.params["$filter"] == "isRead eq false"
    assert request.url.params["$orderby"] == "receivedDateTime desc"
    assert request.url.params["$top"] == "5"
    assert request.headers["Prefer"] == 'outlook.body-content-type="text"'
    message = messages[0]
    assert message.id == "AAMk1" and message.thread_id == "conv-1"
    assert message.sender == "Maya <maya@example.com>"
    assert message.recipients == ("you@example.com",)
    assert message.received_at == datetime(2026, 10, 21, 7, 48, 0, 123456, tzinfo=UTC)
    assert message.unread and message.body.startswith("Can we move it?")


async def test_search_quotes_the_query(router: Router, provider: OutlookProvider) -> None:
    router.add("GET", "/v1.0/me/messages", {"value": []})

    assert await provider.search("from:maya", limit=2) == []
    assert router.last().url.params["$search"] == '"from:maya"'


async def test_get_fetches_one_message(router: Router, provider: OutlookProvider) -> None:
    router.add("GET", "/v1.0/me/messages/AAMk1", MESSAGE)

    assert (await provider.get("AAMk1")).subject == "Design review"


def test_html_bodies_are_not_passed_through() -> None:
    html = {**MESSAGE, "body": {"contentType": "html", "content": "<p>x</p>"}}

    assert parse_message(html).body == ""
    assert parse_message({**MESSAGE, "from": {}}).sender == ""


async def test_draft_reply_uses_create_reply_and_send_posts_to_the_draft(
    router: Router, provider: OutlookProvider
) -> None:
    router.add(
        "POST",
        "/v1.0/me/messages/AAMk1/createReply",
        {
            "id": "draft-1",
            "subject": "RE: Design review",
            "conversationId": "conv-1",
            "toRecipients": [{"emailAddress": {"name": "Maya", "address": "maya@example.com"}}],
        },
    )
    router.add("POST", "/v1.0/me/messages/draft-1/send", status=202)
    original = parse_message(MESSAGE)

    draft = await provider.draft_reply(original, "Thursday works.")
    sent_id = await provider.send(draft)

    assert router.sent_json(0) == {"comment": "Thursday works."}
    assert draft.id == "draft-1"
    assert draft.to == ("Maya <maya@example.com>",)
    assert draft.subject == "RE: Design review"
    assert draft.in_reply_to == "AAMk1" and draft.thread_id == "conv-1"
    assert sent_id == "draft-1"


async def test_agenda_uses_calendar_view_in_utc(router: Router, provider: OutlookProvider) -> None:
    router.add("GET", "/v1.0/me/calendarView", {"value": [EVENT]})
    window = TimeRange(NOW, NOW + timedelta(days=1))

    events = await provider.agenda(window)

    request = router.last()
    assert request.url.params["startDateTime"] == NOW.isoformat()
    assert request.url.params["endDateTime"] == (NOW + timedelta(days=1)).isoformat()
    assert request.headers["Prefer"] == 'outlook.timezone="UTC"'
    event = events[0]
    assert event.start == datetime(2026, 10, 21, 10, tzinfo=UTC)
    assert event.end == datetime(2026, 10, 21, 11, tzinfo=UTC)
    assert event.location == "Room 3B" and event.attendees == ("maya@example.com",)
    assert event.description == "Bring estimates" and not event.all_day


async def test_create_event_posts_graph_shaped_body(
    router: Router, provider: OutlookProvider
) -> None:
    router.add("POST", "/v1.0/me/events", {**EVENT, "id": "new", "subject": "Coffee"})

    event = await provider.create_event(
        "Coffee",
        NOW.replace(hour=15),
        NOW.replace(hour=15, minute=30),
        description="Catch up",
        location="Cafe",
        attendees=["tom@example.com"],
    )

    body = router.sent_json()
    assert body["subject"] == "Coffee"
    assert body["body"] == {"contentType": "text", "content": "Catch up"}
    assert body["start"] == {"dateTime": "2026-10-21T15:00:00+00:00", "timeZone": "UTC"}
    assert body["location"] == {"displayName": "Cafe"}
    assert body["attendees"] == [
        {"emailAddress": {"address": "tom@example.com"}, "type": "required"}
    ]
    assert event.id == "new" and event.title == "Coffee"


async def test_server_errors_surface_as_provider_errors(
    router: Router, provider: OutlookProvider
) -> None:
    router.add("GET", "/v1.0/me/calendarView", status=503, text="busy")

    with pytest.raises(ProviderError) as raised:
        await provider.agenda(TimeRange(NOW, NOW + timedelta(hours=1)))

    assert raised.value.status_code == 503
    assert (await provider.status()).last_error is not None
