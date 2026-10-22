"""Slack provider over ``slack-sdk``'s ``AsyncWebClient``.

Only ``api_call`` is used, through a Protocol, so the provider depends on
one method of the SDK and tests inject a scripted fake. A bot token is
required: bots only see the channels they were invited to, which doubles
as the privacy boundary for what Jarvis may read aloud.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Protocol

from slack_sdk.errors import SlackApiError, SlackClientError
from slack_sdk.web.async_client import AsyncWebClient

from jarvis.integrations.base import (
    AuthError,
    ChatMessage,
    ChatProvider,
    IntegrationError,
    NotConfiguredError,
    ProviderError,
    ProviderStatus,
    RateLimitedError,
)
from jarvis.utils.logging import get_logger

if TYPE_CHECKING:
    from jarvis.config import Settings

log = get_logger(__name__)

PROVIDER = "slack"
_CONVERSATION_TYPES = "public_channel,private_channel,im,mpim"
_LIST_PAGE = 200
_AUTH_ERRORS = frozenset(
    {
        "invalid_auth",
        "not_authed",
        "token_revoked",
        "token_expired",
        "account_inactive",
        "missing_scope",
        "not_allowed_token_type",
    }
)


class SlackResponse(Protocol):
    """The slice of ``AsyncSlackResponse`` Jarvis relies on."""

    @property
    def data(self) -> Any:
        """Decoded JSON body (a dict for every method used here)."""


class SlackApi(Protocol):
    """The slice of ``AsyncWebClient`` Jarvis relies on."""

    async def api_call(
        self,
        api_method: str,
        *,
        http_verb: str = "POST",
        params: dict[str, Any] | None = None,
        json: dict[str, Any] | None = None,
    ) -> SlackResponse:
        """Call one Web API method."""


@dataclass(frozen=True, slots=True)
class _Channel:
    id: str
    name: str
    last_read: float


class SlackProvider(ChatProvider):
    """Read and post in the channels the bot belongs to."""

    provider_name = PROVIDER

    def __init__(self, api: SlackApi) -> None:
        self._api = api
        self._last_error: str | None = None
        self._authenticated = False
        self._me_id: str | None = None
        self._user_names: dict[str, str] = {}

    @classmethod
    def from_settings(cls, settings: Settings) -> SlackProvider:
        """Build on the real ``AsyncWebClient``.

        Raises:
            NotConfiguredError: If ``SLACK_BOT_TOKEN`` is unset.
        """
        if settings.slack_bot_token is None:
            raise NotConfiguredError(PROVIDER, ["SLACK_BOT_TOKEN"])
        return cls(AsyncWebClient(token=settings.slack_bot_token.get_secret_value()))

    async def status(self) -> ProviderStatus:
        """Configured by construction; authenticated after the first successful call."""
        return ProviderStatus(
            name=self.provider_name,
            configured=True,
            authenticated=self._authenticated,
            last_error=self._last_error,
        )

    async def list_recent(
        self,
        channel: str | None = None,
        limit: int = 20,
        *,
        unread_only: bool = False,
        mentions_only: bool = False,
    ) -> list[ChatMessage]:
        """Newest messages across one channel or every channel the bot is in."""
        channels = [await self._channel(channel)] if channel else await self._member_channels()
        messages: list[ChatMessage] = []
        for item in channels:
            messages.extend(await self._history(item, limit))
        messages.sort(key=lambda m: m.sent_at, reverse=True)
        return [
            m
            for m in messages
            if (not unread_only or m.unread) and (not mentions_only or m.mentions_me)
        ][:limit]

    async def send(self, channel: str, text: str) -> ChatMessage:
        """Post as the bot."""
        data = await self._call("chat.postMessage", json={"channel": channel, "text": text})
        return ChatMessage(
            id=data["ts"],
            channel_id=data["channel"],
            channel_name=channel,
            author="jarvis",
            text=text,
            sent_at=_from_ts(data["ts"]),
        )

    async def _member_channels(self) -> list[_Channel]:
        data = await self._call(
            "conversations.list",
            http_verb="GET",
            params={"types": _CONVERSATION_TYPES, "exclude_archived": "true", "limit": _LIST_PAGE},
        )
        members = [c for c in data.get("channels", []) if c.get("is_member") or c.get("is_im")]
        return [await self._channel(c["id"]) for c in members]

    async def _channel(self, channel_id: str) -> _Channel:
        data = await self._call(
            "conversations.info", http_verb="GET", params={"channel": channel_id}
        )
        info = data["channel"]
        return _Channel(
            id=info["id"],
            name=info.get("name") or info.get("user") or info["id"],
            last_read=float(info.get("last_read", 0)),
        )

    async def _history(self, channel: _Channel, limit: int) -> list[ChatMessage]:
        me = await self._me()
        data = await self._call(
            "conversations.history",
            http_verb="GET",
            params={"channel": channel.id, "limit": limit},
        )
        return [await self._parse(m, channel, me) for m in data.get("messages", [])]

    async def _parse(self, data: dict[str, Any], channel: _Channel, me: str) -> ChatMessage:
        ts = data["ts"]
        return ChatMessage(
            id=ts,
            channel_id=channel.id,
            channel_name=channel.name,
            author=await self._author(data),
            text=data.get("text", ""),
            sent_at=_from_ts(ts),
            mentions_me=f"<@{me}>" in data.get("text", ""),
            unread=float(ts) > channel.last_read,
        )

    async def _author(self, data: dict[str, Any]) -> str:
        user_id = data.get("user")
        if user_id is None:
            return str(data.get("username") or data.get("bot_id") or "bot")
        if user_id not in self._user_names:
            info = await self._call("users.info", http_verb="GET", params={"user": user_id})
            user = info.get("user", {})
            self._user_names[user_id] = user.get("real_name") or user.get("name") or user_id
        return self._user_names[user_id]

    async def _me(self) -> str:
        if self._me_id is None:
            self._me_id = str((await self._call("auth.test"))["user_id"])
        return self._me_id

    async def _call(
        self,
        method: str,
        *,
        http_verb: str = "POST",
        params: dict[str, Any] | None = None,
        json: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        try:
            response = await self._api.api_call(
                method, http_verb=http_verb, params=params, json=json
            )
        except SlackApiError as error:
            raise self._fail(_map_api_error(error)) from error
        except SlackClientError as error:
            raise self._fail(ProviderError(PROVIDER, f"request failed: {error!r}")) from error
        if not isinstance(response.data, dict):
            raise self._fail(ProviderError(PROVIDER, f"unexpected response to {method}"))
        self._last_error, self._authenticated = None, True
        return response.data

    def _fail(self, error: IntegrationError) -> IntegrationError:
        self._last_error = str(error)
        log.warning("integrations.request.failed", provider=PROVIDER, error=str(error))
        return error


def _map_api_error(error: SlackApiError) -> IntegrationError:
    response = error.response
    data = response.data if isinstance(response.data, dict) else {}
    code = str(data.get("error", "unknown_error"))
    if code == "ratelimited":
        retry_after = response.headers.get("Retry-After")
        return RateLimitedError(
            PROVIDER, "rate limited", retry_after=float(retry_after) if retry_after else None
        )
    if code in _AUTH_ERRORS:
        return AuthError(PROVIDER, f"auth failed: {code}")
    return ProviderError(PROVIDER, f"API error: {code}", status_code=response.status_code)


def _from_ts(ts: str) -> datetime:
    return datetime.fromtimestamp(float(ts), tz=UTC)


# Design notes:
# - conversations.info is called once per member channel to get last_read,
#   which is the only unread signal Slack offers a bot. Bots are in few
#   channels (they must be invited), so the N+1 stays small.
# - The provider depends on the SDK's api_call only; the convenience methods
#   (conversations_history ...) would each need their own Protocol entry.
# - Mentions are detected from the raw "<@Uxxxx>" markup rather than parsed
#   blocks: it is what every client renders and it never misses.
