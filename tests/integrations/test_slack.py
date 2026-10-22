from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import pytest
from slack_sdk.errors import SlackApiError, SlackClientError

from jarvis.config import Settings
from jarvis.integrations.base import (
    AuthError,
    NotConfiguredError,
    ProviderError,
    RateLimitedError,
)
from jarvis.integrations.slack import SlackProvider


@dataclass
class FakeResponse:
    data: Any
    status_code: int = 200
    headers: dict[str, str] = field(default_factory=dict)


@dataclass
class FakeSlackApi:
    """Scripted responses per API method; raises when the script says so."""

    responses: dict[str, Any] = field(default_factory=dict)
    calls: list[tuple[str, str, dict[str, Any] | None, dict[str, Any] | None]] = field(
        default_factory=list
    )

    async def api_call(
        self,
        api_method: str,
        *,
        http_verb: str = "POST",
        params: dict[str, Any] | None = None,
        json: dict[str, Any] | None = None,
    ) -> FakeResponse:
        self.calls.append((api_method, http_verb, params, json))
        scripted = self.responses[api_method]
        if isinstance(scripted, Exception):
            raise scripted
        if callable(scripted):
            return FakeResponse(scripted(params or json or {}))
        return FakeResponse(scripted)


def api_error(code: str, status: int = 200, headers: dict[str, str] | None = None) -> SlackApiError:
    response = FakeResponse({"ok": False, "error": code}, status, headers or {})
    return SlackApiError(f"error: {code}", response)


CHANNEL_INFO = {
    "C1": {"id": "C1", "name": "dev", "last_read": "1792000100.000000"},
    "D1": {"id": "D1", "user": "U2", "last_read": "0"},
}

HISTORY = {
    "C1": [
        {"ts": "1792000200.000100", "user": "U2", "text": "<@UBOT> review please"},
        {"ts": "1792000050.000100", "user": "U2", "text": "older"},
        {"ts": "1792000010.000100", "bot_id": "B9", "username": "ci", "text": "build ok"},
    ],
    "D1": [{"ts": "1792000150.000100", "user": "U3", "text": "lunch?"}],
}


def scripted_api() -> FakeSlackApi:
    return FakeSlackApi(
        {
            "auth.test": {"ok": True, "user_id": "UBOT"},
            "conversations.list": {
                "ok": True,
                "channels": [
                    {"id": "C1", "is_member": True},
                    {"id": "C9", "is_member": False},
                    {"id": "D1", "is_im": True},
                ],
            },
            "conversations.info": lambda p: {"ok": True, "channel": CHANNEL_INFO[p["channel"]]},
            "conversations.history": lambda p: {"ok": True, "messages": HISTORY[p["channel"]]},
            "users.info": lambda p: {"ok": True, "user": {"real_name": f"Name of {p['user']}"}},
            "chat.postMessage": lambda j: {
                "ok": True,
                "ts": "1792000300.000100",
                "channel": j["channel"],
            },
        }
    )


async def test_overview_reads_member_channels_and_dms_newest_first() -> None:
    api = scripted_api()
    provider = SlackProvider(api)

    messages = await provider.list_recent(limit=10)

    assert [(m.channel_name, m.text) for m in messages] == [
        ("dev", "<@UBOT> review please"),
        ("U2", "lunch?"),
        ("dev", "older"),
        ("dev", "build ok"),
    ]
    assert [m.unread for m in messages] == [True, True, False, False]
    assert [m.mentions_me for m in messages] == [True, False, False, False]
    assert [m.author for m in messages] == ["Name of U2", "Name of U3", "Name of U2", "ci"]
    assert sum(1 for c in api.calls if c[0] == "users.info") == 2  # cached per user
    assert sum(1 for c in api.calls if c[0] == "auth.test") == 1
    assert "C9" not in {c[2]["channel"] for c in api.calls if c[0] == "conversations.history"}
    assert (await provider.status()).authenticated


async def test_filters_limit_and_single_channel() -> None:
    provider = SlackProvider(scripted_api())

    mentions = await provider.list_recent(mentions_only=True)
    unread = await provider.list_recent("C1", unread_only=True, limit=1)

    assert [m.id for m in mentions] == ["1792000200.000100"]
    assert [m.id for m in unread] == ["1792000200.000100"]
    assert unread[0].channel_id == "C1"


async def test_send_posts_and_returns_the_message() -> None:
    api = scripted_api()
    provider = SlackProvider(api)

    message = await provider.send("C1", "On my way")

    assert api.calls[-1] == (
        "chat.postMessage",
        "POST",
        None,
        {"channel": "C1", "text": "On my way"},
    )
    assert message.id == "1792000300.000100"
    assert message.channel_id == "C1" and message.text == "On my way"


@pytest.mark.parametrize(
    ("error", "expected", "fragment"),
    [
        (api_error("ratelimited", 429, {"Retry-After": "3"}), RateLimitedError, "rate limited"),
        (api_error("invalid_auth"), AuthError, "invalid_auth"),
        (api_error("missing_scope"), AuthError, "missing_scope"),
        (api_error("channel_not_found", 200), ProviderError, "channel_not_found"),
        (SlackClientError("connection reset"), ProviderError, "request failed"),
    ],
)
async def test_sdk_errors_map_to_the_hierarchy(
    error: Exception, expected: type[Exception], fragment: str
) -> None:
    provider = SlackProvider(FakeSlackApi({"chat.postMessage": error}))

    with pytest.raises(expected, match=fragment) as raised:
        await provider.send("C1", "x")

    if isinstance(raised.value, RateLimitedError):
        assert raised.value.retry_after == 3.0
    status = await provider.status()
    assert status.configured and not status.authenticated
    assert status.last_error == str(raised.value)


async def test_non_dict_payload_is_a_provider_error() -> None:
    provider = SlackProvider(FakeSlackApi({"conversations.info": b"binary"}))

    with pytest.raises(ProviderError, match="unexpected response"):
        await provider.list_recent("C1")


def test_from_settings_requires_the_bot_token(make_settings: Callable[..., Settings]) -> None:
    with pytest.raises(NotConfiguredError, match="SLACK_BOT_TOKEN"):
        SlackProvider.from_settings(make_settings())

    provider = SlackProvider.from_settings(make_settings(slack_bot_token="xoxb-fake"))
    assert provider.provider_name == "slack"
