"""Discord provider over the REST API with a bot token.

A gateway (websocket) client such as ``discord.py``'s ``Client`` would have
to stay connected for the whole session and maintain its own event loop
integration just to answer "what did I miss?". Plain REST calls answer that
on demand and need nothing running in the background. The limitation, stated
here and in ADR-0009: only channels reachable by id or listed in the
configured guild are read, and Discord gives bots no unread state, so
``unread_only`` is accepted but cannot filter anything.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any

import httpx

from jarvis.integrations.base import ChatMessage, ChatProvider, NotConfiguredError, ProviderStatus
from jarvis.integrations.rest import RestSession
from jarvis.integrations.tokens import StaticTokenSource, TokenSource
from jarvis.utils.logging import get_logger

if TYPE_CHECKING:
    from jarvis.config import Settings

log = get_logger(__name__)

PROVIDER = "discord"
BASE_URL = "https://discord.com/api/v10"

_GUILD_TEXT_CHANNEL = 0


class DiscordProvider(ChatProvider):
    """Read and post in guild text channels and DMs the bot can reach."""

    provider_name = PROVIDER

    def __init__(
        self, http: httpx.AsyncClient, token_source: TokenSource, *, guild_id: str | None = None
    ) -> None:
        self._rest = RestSession(
            http,
            provider=self.provider_name,
            base_url=BASE_URL,
            token_source=token_source,
            auth_scheme="Bot",
        )
        self._guild_id = guild_id
        self._me_id: str | None = None
        self._channel_names: dict[str, str] = {}

    @classmethod
    def from_settings(cls, settings: Settings, http: httpx.AsyncClient) -> DiscordProvider:
        """Build from the bot token and optional guild id.

        Raises:
            NotConfiguredError: If ``DISCORD_BOT_TOKEN`` is unset.
        """
        if settings.discord_bot_token is None:
            raise NotConfiguredError(PROVIDER, ["DISCORD_BOT_TOKEN"])
        return cls(
            http, StaticTokenSource(settings.discord_bot_token), guild_id=settings.discord_guild_id
        )

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
        """Newest messages in one channel, or across the guild's text channels.

        Raises:
            NotConfiguredError: If no channel is given and ``DISCORD_GUILD_ID``
                is unset, since bots cannot enumerate channels otherwise.
        """
        if unread_only:
            log.debug("integrations.discord.unread_unsupported")
        channel_ids = [channel] if channel else await self._guild_text_channels()
        messages: list[ChatMessage] = []
        for channel_id in channel_ids:
            messages.extend(await self._channel_messages(channel_id, limit))
        messages.sort(key=lambda m: m.sent_at, reverse=True)
        return [m for m in messages if not mentions_only or m.mentions_me][:limit]

    async def send(self, channel: str, text: str) -> ChatMessage:
        """Post as the bot."""
        data = await self._rest.request(
            "POST", f"/channels/{channel}/messages", json={"content": text}
        )
        return parse_message(data, await self._channel_name(channel), me_id=None)

    async def _guild_text_channels(self) -> list[str]:
        if self._guild_id is None:
            raise NotConfiguredError(PROVIDER, ["DISCORD_GUILD_ID"])
        data = await self._rest.request("GET", f"/guilds/{self._guild_id}/channels")
        text_channels = [c for c in data if c.get("type") == _GUILD_TEXT_CHANNEL]
        for item in text_channels:
            self._channel_names[item["id"]] = item.get("name", item["id"])
        return [item["id"] for item in text_channels]

    async def _channel_messages(self, channel_id: str, limit: int) -> list[ChatMessage]:
        name = await self._channel_name(channel_id)
        me = await self._me()
        data = await self._rest.request(
            "GET", f"/channels/{channel_id}/messages", params={"limit": limit}
        )
        return [parse_message(item, name, me_id=me) for item in data]

    async def _channel_name(self, channel_id: str) -> str:
        if channel_id not in self._channel_names:
            data = await self._rest.request("GET", f"/channels/{channel_id}")
            recipients = ", ".join(r.get("username", "") for r in data.get("recipients", []))
            self._channel_names[channel_id] = data.get("name") or recipients or channel_id
        return self._channel_names[channel_id]

    async def _me(self) -> str:
        if self._me_id is None:
            data = await self._rest.request("GET", "/users/@me")
            self._me_id = str(data["id"])
        return self._me_id


def parse_message(data: dict[str, Any], channel_name: str, *, me_id: str | None) -> ChatMessage:
    """Map a Discord ``message`` object to the neutral model."""
    author = data.get("author", {})
    return ChatMessage(
        id=data["id"],
        channel_id=data["channel_id"],
        channel_name=channel_name,
        author=author.get("global_name") or author.get("username", "unknown"),
        text=data.get("content", ""),
        sent_at=datetime.fromisoformat(data["timestamp"]),
        mentions_me=me_id is not None
        and any(u.get("id") == me_id for u in data.get("mentions", [])),
    )


# Design notes:
# - Reading every text channel of a guild is one request per channel. For a
#   personal server that is a handful; a large community server should set a
#   channel explicitly, and the rate limiter will say so otherwise.
# - The "Bot" authorization scheme is the only difference from the other REST
#   providers, hence the auth_scheme knob on RestSession.
