from datetime import UTC, datetime

import pytest

from jarvis.integrations.teams import TEAMS_SCOPES, TeamsProvider
from jarvis.integrations.tokens import StaticTokenSource
from tests.integrations.conftest import Router

CHATS = {
    "value": [
        {
            "id": "chat-1",
            "topic": "Audio team",
            "chatType": "group",
            "viewpoint": {"lastMessageReadDateTime": "2026-10-21T07:00:00Z"},
            "lastMessagePreview": {
                "id": "p1",
                "createdDateTime": "2026-10-21T07:30:00.0000000Z",
                "from": {"user": {"displayName": "Maya"}},
                "body": {"contentType": "html", "content": "<p>Ship it?</p>"},
            },
        },
        {
            "id": "chat-2",
            "topic": None,
            "chatType": "oneOnOne",
            "viewpoint": {"lastMessageReadDateTime": "2026-10-21T08:00:00Z"},
            "lastMessagePreview": {
                "id": "p2",
                "createdDateTime": "2026-10-21T06:00:00Z",
                "from": {"application": {"displayName": "Planner"}},
                "body": {"contentType": "text", "content": "Task due"},
            },
        },
        {"id": "chat-3", "chatType": "meeting", "lastMessagePreview": None},
    ]
}

MESSAGES = {
    "value": [
        {
            "id": "m1",
            "createdDateTime": "2026-10-21T07:30:00Z",
            "from": {"user": {"displayName": "Maya"}},
            "body": {"contentType": "text", "content": "@You review?"},
            "mentions": [{"mentioned": {"user": {"id": "me-id"}}}],
        },
        {
            "id": "m2",
            "createdDateTime": "2026-10-21T07:20:00Z",
            "from": None,
            "body": {"contentType": "text", "content": "hello"},
        },
    ]
}


@pytest.fixture
def provider(router: Router, token_source: StaticTokenSource) -> TeamsProvider:
    return TeamsProvider(router.client(), token_source)


def test_scopes_are_chat_read_and_message_send() -> None:
    assert TeamsProvider.scopes == TEAMS_SCOPES == ("Chat.Read", "ChatMessage.Send")


async def test_overview_returns_one_preview_per_chat_with_unread_state(
    router: Router, provider: TeamsProvider
) -> None:
    router.add("GET", "/v1.0/me/chats", CHATS)

    messages = await provider.list_recent(limit=10)

    assert router.last().url.params["$expand"] == "lastMessagePreview"
    assert [m.id for m in messages] == ["p1", "p2"]
    first, second = messages
    assert first.channel_name == "Audio team" and first.text == "Ship it?" and first.unread
    assert first.author == "Maya"
    assert second.channel_name == "oneOnOne" and second.author == "Planner" and not second.unread
    assert [m.id for m in await provider.list_recent(unread_only=True)] == ["p1"]


async def test_chat_messages_detect_mentions_of_the_current_user(
    router: Router, provider: TeamsProvider
) -> None:
    router.add("GET", "/v1.0/me", {"id": "me-id"})
    router.add("GET", "/v1.0/me/chats/chat-1/messages", MESSAGES)

    everything = await provider.list_recent("chat-1", limit=5)
    mentions = await provider.list_recent("chat-1", mentions_only=True)

    assert [(m.id, m.mentions_me, m.author) for m in everything] == [
        ("m1", True, "Maya"),
        ("m2", False, "unknown"),
    ]
    assert [m.id for m in mentions] == ["m1"]
    assert sum(1 for r in router.requests if r.url.path == "/v1.0/me") == 1  # cached


async def test_send_posts_a_plain_text_body(router: Router, provider: TeamsProvider) -> None:
    router.add(
        "POST",
        "/v1.0/me/chats/chat-1/messages",
        {
            "id": "new",
            "createdDateTime": "2026-10-21T08:00:00Z",
            "from": {"user": {"displayName": "Me"}},
            "body": {"contentType": "text", "content": "On it"},
        },
    )

    message = await provider.send("chat-1", "On it")

    assert router.sent_json() == {"body": {"content": "On it"}}
    assert message.id == "new" and message.channel_id == "chat-1"
    assert message.sent_at == datetime(2026, 10, 21, 8, tzinfo=UTC)
