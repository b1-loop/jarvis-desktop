"""Microsoft Teams chats over Microsoft Graph.

Reuses the Outlook sign-in (same msal app, scopes added to the same consent)
and reads the user's chats through REST rather than a bot framework: there
is no bot to register and nothing listens for events. The trade-off is
polling semantics, which suits "what did I miss?" and "reply to Maya" and
is documented in ADR-0008. Channel posts in teams are out of scope; only
chats (1:1, group, meeting) are reachable with the ``Chat.*`` scopes.
"""

from __future__ import annotations

import re
from typing import Any

import httpx

from jarvis.integrations.base import ChatMessage, ChatProvider, ProviderStatus
from jarvis.integrations.microsoft_graph import GRAPH_BASE_URL, parse_graph_datetime
from jarvis.integrations.rest import RestSession
from jarvis.integrations.tokens import TokenSource

TEAMS_SCOPES = ("Chat.Read", "ChatMessage.Send")

_TAGS = re.compile(r"<[^>]+>")


class TeamsProvider(ChatProvider):
    """Chats for the signed-in Microsoft account."""

    provider_name = "teams"
    scopes = TEAMS_SCOPES

    def __init__(self, http: httpx.AsyncClient, token_source: TokenSource) -> None:
        self._rest = RestSession(
            http, provider=self.provider_name, base_url=GRAPH_BASE_URL, token_source=token_source
        )
        self._me_id: str | None = None

    async def status(self) -> ProviderStatus:
        """Session-level status; see :meth:`RestSession.status`."""
        return self._rest.status()

    async def list_recent(
        self,
        channel: str | None = None,
        limit: int = 20,
        *,
        unread_only: bool = False,
        mentions_only: bool = False,
    ) -> list[ChatMessage]:
        """Latest messages of one chat, or the last message of each chat.

        Without a chat id, one message per chat is returned from
        ``lastMessagePreview`` and mention detection is unavailable (the
        preview carries no mentions), so ``mentions_only`` only applies when
        a chat id is given.
        """
        messages = (
            await self._chat_messages(channel, limit)
            if channel
            else await self._chat_overview(limit)
        )
        return [
            m
            for m in messages
            if (not unread_only or m.unread) and (not mentions_only or m.mentions_me)
        ][:limit]

    async def send(self, channel: str, text: str) -> ChatMessage:
        """Post a plain-text message to a chat."""
        data = await self._rest.request(
            "POST", f"/me/chats/{channel}/messages", json={"body": {"content": text}}
        )
        return _parse_message(data, channel, channel, me_id=None)

    async def _chat_messages(self, chat_id: str, limit: int) -> list[ChatMessage]:
        me_id = await self._current_user_id()
        data = await self._rest.request(
            "GET", f"/me/chats/{chat_id}/messages", params={"$top": limit}
        )
        return [_parse_message(m, chat_id, chat_id, me_id=me_id) for m in data.get("value", [])]

    async def _chat_overview(self, limit: int) -> list[ChatMessage]:
        data = await self._rest.request(
            "GET",
            "/me/chats",
            params={"$expand": "lastMessagePreview", "$top": limit},
        )
        return [
            _parse_preview(chat) for chat in data.get("value", []) if chat.get("lastMessagePreview")
        ]

    async def _current_user_id(self) -> str:
        if self._me_id is None:
            data = await self._rest.request("GET", "/me", params={"$select": "id"})
            self._me_id = str(data["id"])
        return self._me_id


def _parse_message(
    data: dict[str, Any], chat_id: str, chat_name: str, *, me_id: str | None
) -> ChatMessage:
    mentioned = {m.get("mentioned", {}).get("user", {}).get("id") for m in data.get("mentions", [])}
    return ChatMessage(
        id=data["id"],
        channel_id=chat_id,
        channel_name=chat_name,
        author=_author(data),
        text=_plain_text(data.get("body", {})),
        sent_at=parse_graph_datetime(data["createdDateTime"]),
        mentions_me=me_id is not None and me_id in mentioned,
    )


def _parse_preview(chat: dict[str, Any]) -> ChatMessage:
    preview = chat["lastMessagePreview"]
    last_read = chat.get("viewpoint", {}).get("lastMessageReadDateTime")
    sent_at = parse_graph_datetime(preview["createdDateTime"])
    return ChatMessage(
        id=preview["id"],
        channel_id=chat["id"],
        channel_name=chat.get("topic") or chat.get("chatType", chat["id"]),
        author=_author(preview),
        text=_plain_text(preview.get("body", {})),
        sent_at=sent_at,
        unread=last_read is None or sent_at > parse_graph_datetime(last_read),
    )


def _author(data: dict[str, Any]) -> str:
    sender = data.get("from") or {}
    user = sender.get("user") or sender.get("application") or {}
    return str(user.get("displayName", "unknown"))


def _plain_text(body: dict[str, Any]) -> str:
    content = str(body.get("content", ""))
    return _TAGS.sub("", content).strip() if body.get("contentType") == "html" else content


# Design notes:
# - The overview uses $expand=lastMessagePreview so "what did I miss" is one
#   request however many chats exist. Reading full histories of every chat
#   would be N+1 and rate-limit prone for little gain in a spoken summary.
# - Unread is derived from chat.viewpoint.lastMessageReadDateTime, which Graph
#   maintains per user; it is the only read-state signal available.
# - HTML bodies are stripped with a regex rather than an HTML parser: Teams
#   message markup is shallow and the text is only read aloud.
