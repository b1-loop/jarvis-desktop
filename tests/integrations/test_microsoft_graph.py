from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta, timezone
from typing import Any

import pytest

from jarvis.config import Settings
from jarvis.integrations.base import AuthError, NotConfiguredError
from jarvis.integrations.microsoft_graph import (
    STORE_KEY,
    MicrosoftAuth,
    TokenCache,
    parse_graph_datetime,
)
from jarvis.integrations.tokens import MemoryTokenStore

SCOPES = ("Mail.Read", "Chat.Read")


@dataclass
class FakeCache:
    state: str | None = None
    has_state_changed: bool = False
    deserialized: list[str | None] = field(default_factory=list)

    def serialize(self) -> str:
        return self.state or "{}"

    def deserialize(self, state: str | None) -> None:
        self.deserialized.append(state)
        self.state = state


@dataclass
class FakeApp:
    accounts: list[dict[str, Any]] = field(default_factory=list)
    silent: dict[str, Any] | None = None
    flow: dict[str, Any] = field(
        default_factory=lambda: {"user_code": "ABC123", "message": "go to ... enter ABC123"}
    )
    device_result: dict[str, Any] = field(default_factory=lambda: {"access_token": "fake-device"})
    silent_calls: list[tuple[list[str], dict[str, Any] | None]] = field(default_factory=list)
    cache: FakeCache | None = None

    def get_accounts(self) -> list[dict[str, Any]]:
        return self.accounts

    def acquire_token_silent(
        self, scopes: list[str], account: dict[str, Any] | None
    ) -> dict[str, Any] | None:
        self.silent_calls.append((scopes, account))
        return self.silent

    def initiate_device_flow(self, scopes: list[str]) -> dict[str, Any]:
        assert scopes == list(SCOPES)
        return self.flow

    def acquire_token_by_device_flow(self, flow: dict[str, Any]) -> dict[str, Any]:
        assert flow is self.flow
        if self.cache is not None:
            self.cache.has_state_changed = True
            self.cache.state = '{"fake": "cache"}'
        return self.device_result


def make_auth(
    app: FakeApp, store: MemoryTokenStore, cache: FakeCache, prompts: list[str]
) -> MicrosoftAuth:
    def factory(given: TokenCache) -> FakeApp:
        assert given is cache
        app.cache = cache
        return app

    return MicrosoftAuth(factory, SCOPES, store, cache_factory=lambda: cache, prompt=prompts.append)


async def test_silent_token_is_used_when_an_account_is_cached() -> None:
    app = FakeApp(accounts=[{"username": "me@example.com"}], silent={"access_token": "fake-silent"})
    prompts: list[str] = []

    token = await make_auth(app, MemoryTokenStore(), FakeCache(), prompts).access_token()

    assert token == "fake-silent"
    assert app.silent_calls == [(list(SCOPES), {"username": "me@example.com"})]
    assert prompts == []


async def test_device_flow_runs_when_nothing_is_cached_and_persists_the_cache() -> None:
    app, store, cache, prompts = FakeApp(), MemoryTokenStore(), FakeCache(), []

    token = await make_auth(app, store, cache, prompts).access_token()

    assert token == "fake-device"
    assert prompts == ["go to ... enter ABC123"]
    assert store.load(STORE_KEY) == {"cache": '{"fake": "cache"}'}


async def test_stored_cache_is_loaded_before_acquisition() -> None:
    store = MemoryTokenStore()
    store.save(STORE_KEY, {"cache": '{"stored": true}'})
    cache = FakeCache()
    app = FakeApp(accounts=[{}], silent={"access_token": "fake"})

    await make_auth(app, store, cache, []).access_token()

    assert cache.deserialized == ['{"stored": true}']


async def test_failures_become_auth_errors() -> None:
    cannot_start = FakeApp(flow={"error": "invalid_client", "error_description": "bad app"})
    with pytest.raises(AuthError, match="could not start device flow: bad app"):
        await make_auth(cannot_start, MemoryTokenStore(), FakeCache(), []).access_token()

    declined = FakeApp(device_result={"error": "authorization_declined"})
    with pytest.raises(AuthError, match="authorization_declined"):
        await make_auth(declined, MemoryTokenStore(), FakeCache(), []).access_token()


async def test_app_is_built_once_and_lazily() -> None:
    builds: list[int] = []

    def factory(_cache: TokenCache) -> FakeApp:
        builds.append(1)
        return FakeApp(accounts=[{}], silent={"access_token": "fake"})

    auth = MicrosoftAuth(factory, SCOPES, MemoryTokenStore(), cache_factory=FakeCache)
    assert builds == []

    await auth.access_token()
    await auth.access_token()

    assert len(builds) == 1


def test_from_settings_requires_client_id(make_settings: Callable[..., Settings]) -> None:
    with pytest.raises(NotConfiguredError, match="MS_CLIENT_ID"):
        MicrosoftAuth.from_settings(make_settings(), SCOPES, MemoryTokenStore())

    auth = MicrosoftAuth.from_settings(
        make_settings(ms_client_id="fake-client-id"), SCOPES, MemoryTokenStore()
    )
    assert auth.scopes == SCOPES


def test_parse_graph_datetime_handles_seven_digit_fractions_and_offsets() -> None:
    assert parse_graph_datetime("2026-10-21T10:00:00.0000000") == datetime(
        2026, 10, 21, 10, tzinfo=UTC
    )
    assert parse_graph_datetime("2026-10-21T10:00:00.1234567Z") == datetime(
        2026, 10, 21, 10, 0, 0, 123456, tzinfo=UTC
    )
    assert parse_graph_datetime("2026-10-21T10:00:00+02:00") == datetime(
        2026, 10, 21, 10, tzinfo=timezone(timedelta(hours=2))
    )
    assert parse_graph_datetime("2026-10-21T10:00:00", assume_utc=False).tzinfo is None
