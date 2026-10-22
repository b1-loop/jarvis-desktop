from collections.abc import Callable
from datetime import UTC, datetime

import pytest

from jarvis.config import Settings
from jarvis.integrations.base import NotConfiguredError
from jarvis.integrations.discord import DiscordProvider
from jarvis.integrations.tokens import StaticTokenSource
from tests.integrations.conftest import FAKE_TOKEN, Router


def message(
    message_id: str, channel_id: str, text: str, *, mention: bool = False
) -> dict[str, object]:
    return {
        "id": message_id,
        "channel_id": channel_id,
        "content": text,
        "timestamp": f"2026-10-21T08:0{message_id[-1]}:00.000000+00:00",
        "author": {"id": "u1", "username": "tom", "global_name": "Tom"},
        "mentions": [{"id": "bot-id"}] if mention else [],
    }


@pytest.fixture
def provider(router: Router, token_source: StaticTokenSource) -> DiscordProvider:
    router.add("GET", "/api/v10/users/@me", {"id": "bot-id"})
    router.add("GET", "/api/v10/channels/c1", {"id": "c1", "name": "general"})
    router.add("GET", "/api/v10/channels/d1", {"id": "d1", "recipients": [{"username": "maya"}]})
    router.add(
        "GET",
        "/api/v10/channels/c1/messages",
        [message("m1", "c1", "hi"), message("m2", "c1", "<@bot-id> ping", mention=True)],
    )
    router.add("GET", "/api/v10/channels/d1/messages", [message("m3", "d1", "lunch?")])
    return DiscordProvider(router.client(), token_source, guild_id="g1")


async def test_channel_messages_use_bot_auth_and_detect_mentions(
    router: Router, provider: DiscordProvider
) -> None:
    messages = await provider.list_recent("c1", limit=5)

    history = next(r for r in router.requests if r.url.path.endswith("/messages"))
    assert history.headers["Authorization"] == f"Bot {FAKE_TOKEN}"
    assert history.url.params["limit"] == "5"
    assert [(m.id, m.mentions_me, m.channel_name, m.author) for m in messages] == [
        ("m2", True, "general", "Tom"),
        ("m1", False, "general", "Tom"),
    ]
    assert messages[0].sent_at == datetime(2026, 10, 21, 8, 2, tzinfo=UTC)
    assert [m.id for m in await provider.list_recent("c1", mentions_only=True)] == ["m2"]


async def test_overview_enumerates_guild_text_channels(
    router: Router, provider: DiscordProvider
) -> None:
    router.add(
        "GET",
        "/api/v10/guilds/g1/channels",
        [
            {"id": "c1", "name": "general", "type": 0},
            {"id": "v1", "name": "voice", "type": 2},
            {"id": "d1", "name": "dm-like", "type": 0},
        ],
    )

    messages = await provider.list_recent(limit=2, unread_only=True)

    assert [m.id for m in messages] == ["m3", "m2"]
    assert messages[0].channel_name == "dm-like"  # name cached from the guild listing
    assert all("/channels/v1/" not in r.url.path for r in router.requests)


async def test_overview_without_guild_id_says_which_variable_to_set(
    router: Router, token_source: StaticTokenSource
) -> None:
    provider = DiscordProvider(router.client(), token_source)

    with pytest.raises(NotConfiguredError, match="DISCORD_GUILD_ID"):
        await provider.list_recent()


async def test_send_posts_content_and_names_dm_channels_by_recipient(
    router: Router, provider: DiscordProvider
) -> None:
    router.add("POST", "/api/v10/channels/d1/messages", message("m9", "d1", "On my way"))

    sent = await provider.send("d1", "On my way")

    post = next(r for r in router.requests if r.method == "POST")
    assert post.headers["Authorization"] == f"Bot {FAKE_TOKEN}"
    assert router.sent_json(router.requests.index(post)) == {"content": "On my way"}
    assert sent.id == "m9" and sent.channel_name == "maya" and not sent.mentions_me


def test_from_settings_requires_the_bot_token(
    make_settings: Callable[..., Settings], router: Router
) -> None:
    with pytest.raises(NotConfiguredError, match="DISCORD_BOT_TOKEN"):
        DiscordProvider.from_settings(make_settings(), router.client())

    provider = DiscordProvider.from_settings(
        make_settings(discord_bot_token="fake-bot-token", discord_guild_id="g1"), router.client()
    )
    assert provider.provider_name == "discord"
