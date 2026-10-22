import base64
from datetime import UTC, datetime
from email import message_from_bytes

import pytest

from jarvis.integrations.base import AuthError, EmailMessage
from jarvis.integrations.gmail import GMAIL_SCOPES, GmailProvider, parse_message
from jarvis.integrations.tokens import StaticTokenSource
from tests.integrations.conftest import Router

PREFIX = "/gmail/v1/users/me"


def b64url(text: str) -> str:
    return base64.urlsafe_b64encode(text.encode()).decode().rstrip("=")


def metadata(message_id: str, *, unread: bool = True) -> dict[str, object]:
    return {
        "id": message_id,
        "threadId": f"t-{message_id}",
        "snippet": "Can we move it?",
        "internalDate": "1792000000000",
        "labelIds": ["INBOX", "UNREAD"] if unread else ["INBOX"],
        "payload": {
            "headers": [
                {"name": "From", "value": "Maya <maya@example.com>"},
                {"name": "To", "value": "you@example.com, tom@example.com"},
                {"name": "Subject", "value": "Design review"},
                {"name": "Message-ID", "value": "<abc@mail.example.com>"},
            ]
        },
    }


@pytest.fixture
def provider(router: Router, token_source: StaticTokenSource) -> GmailProvider:
    return GmailProvider(router.client(), token_source)


def test_scopes_are_read_and_send_only() -> None:
    assert GmailProvider.scopes == GMAIL_SCOPES
    assert all(s.endswith(("gmail.readonly", "gmail.send")) for s in GMAIL_SCOPES)


async def test_list_unread_queries_then_fetches_metadata(
    router: Router, provider: GmailProvider
) -> None:
    router.add("GET", f"{PREFIX}/messages", {"messages": [{"id": "m1"}, {"id": "m2"}]})
    router.add("GET", f"{PREFIX}/messages/m1", metadata("m1"))
    router.add("GET", f"{PREFIX}/messages/m2", metadata("m2", unread=False))

    messages = await provider.list_unread(limit=2)

    listing = router.requests[0]
    assert listing.url.params["q"] == "is:unread in:inbox"
    assert listing.url.params["maxResults"] == "2"
    fetch = router.requests[1]
    assert fetch.url.params["format"] == "metadata"
    assert fetch.url.params.get_list("metadataHeaders") == ["From", "To", "Subject", "Message-ID"]
    assert [m.id for m in messages] == ["m1", "m2"]
    assert messages[0] == EmailMessage(
        id="m1",
        thread_id="t-m1",
        sender="Maya <maya@example.com>",
        recipients=("you@example.com", "tom@example.com"),
        subject="Design review",
        snippet="Can we move it?",
        received_at=datetime.fromtimestamp(1792000000, tz=UTC),
        unread=True,
        rfc822_id="<abc@mail.example.com>",
    )
    assert messages[1].unread is False


async def test_search_passes_the_gmail_query_through(
    router: Router, provider: GmailProvider
) -> None:
    router.add("GET", f"{PREFIX}/messages", {})

    assert await provider.search("from:maya newer_than:7d", limit=3) == []
    assert router.last().url.params["q"] == "from:maya newer_than:7d"


async def test_get_extracts_the_plain_text_part_of_a_multipart_body(
    router: Router, provider: GmailProvider
) -> None:
    full = metadata("m1")
    full["payload"] = {
        "mimeType": "multipart/alternative",
        "headers": [{"name": "Subject", "value": "Hi"}],
        "parts": [
            {"mimeType": "text/html", "body": {"data": b64url("<p>Hi</p>")}},
            {"mimeType": "text/plain", "body": {"data": b64url("Hi there\n")}},
        ],
    }
    router.add("GET", f"{PREFIX}/messages/m1", full)

    message = await provider.get("m1")

    assert router.last().url.params["format"] == "full"
    assert message.body == "Hi there\n"
    assert message.sender == ""


def test_parse_message_without_text_part_has_empty_body() -> None:
    data = metadata("m1")
    data["payload"] = {"mimeType": "text/html", "body": {"data": b64url("<b>x</b>")}}

    assert parse_message(data).body == ""


async def test_draft_reply_builds_a_threaded_mime_message(
    router: Router, provider: GmailProvider
) -> None:
    router.add("POST", f"{PREFIX}/drafts", {"id": "d1"})
    original = parse_message(metadata("m1"))

    draft = await provider.draft_reply(original, "Thursday works.")

    sent = router.sent_json()
    assert sent["message"]["threadId"] == "t-m1"
    raw = sent["message"]["raw"]
    mime = message_from_bytes(base64.urlsafe_b64decode(raw + "=" * (-len(raw) % 4)))
    assert mime["To"] == "Maya <maya@example.com>"
    assert mime["Subject"] == "Re: Design review"
    assert mime["In-Reply-To"] == "<abc@mail.example.com>"
    assert mime.get_payload(decode=True) == b"Thursday works.\n"
    assert draft.id == "d1"
    assert draft.thread_id == "t-m1"
    assert draft.in_reply_to == "m1"


async def test_send_posts_the_draft_id_and_returns_the_sent_id(
    router: Router, provider: GmailProvider
) -> None:
    router.add("POST", f"{PREFIX}/drafts", {"id": "d1"})
    router.add("POST", f"{PREFIX}/drafts/send", {"id": "sent-1"})
    draft = await provider.draft_reply(parse_message(metadata("m1")), "ok")

    assert await provider.send(draft) == "sent-1"
    assert router.sent_json() == {"id": "d1"}


async def test_errors_and_status_come_from_the_session(
    router: Router, provider: GmailProvider
) -> None:
    router.add("GET", f"{PREFIX}/messages", status=401, text="invalid credentials")

    with pytest.raises(AuthError):
        await provider.list_unread()

    status = await provider.status()
    assert status.name == "gmail" and status.configured and not status.authenticated
    assert status.last_error is not None and "HTTP 401" in status.last_error
