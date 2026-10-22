import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from google.auth.exceptions import RefreshError

from jarvis.config import Settings
from jarvis.integrations.base import AuthError, NotConfiguredError
from jarvis.integrations.google_auth import (
    STORE_KEY,
    GoogleAuth,
    GoogleCredentials,
    load_credentials,
    refresh_credentials,
    run_installed_app_flow,
)
from jarvis.integrations.tokens import MemoryTokenStore

SCOPES = ("https://www.googleapis.com/auth/gmail.readonly",)


@dataclass
class FakeCredentials:
    token: str | None = "fake-access"
    valid: bool = True
    refresh_token: str | None = "fake-refresh"
    scopes: tuple[str, ...] = SCOPES
    refreshed: int = 0

    def refresh(self, _request: Any) -> None:
        self.refreshed += 1
        self.valid = True
        self.token = "fake-refreshed"

    def to_json(self) -> str:
        return json.dumps(
            {"token": self.token, "refresh_token": self.refresh_token, "scopes": list(self.scopes)}
        )


@dataclass
class Seams:
    stored: FakeCredentials | None = None
    flow_result: FakeCredentials = field(default_factory=lambda: FakeCredentials("fake-consented"))
    flow_runs: list[tuple[Path, tuple[str, ...]]] = field(default_factory=list)
    refresh_error: AuthError | None = None

    def load(self, info: dict[str, Any], _scopes: Sequence[str]) -> GoogleCredentials:
        assert self.stored is not None
        assert info["refresh_token"] == self.stored.refresh_token
        return self.stored

    def refresh(self, credentials: GoogleCredentials) -> None:
        if self.refresh_error is not None:
            raise self.refresh_error
        credentials.refresh(None)

    def run_flow(self, path: Path, scopes: Sequence[str]) -> GoogleCredentials:
        self.flow_runs.append((path, tuple(scopes)))
        return self.flow_result


def make_auth(seams: Seams, store: MemoryTokenStore) -> GoogleAuth:
    if seams.stored is not None:
        store.save(STORE_KEY, json.loads(seams.stored.to_json()))
    return GoogleAuth(
        Path("client.json"),
        SCOPES,
        store,
        load=seams.load,
        refresh=seams.refresh,
        run_flow=seams.run_flow,
    )


async def test_valid_stored_credentials_are_used_without_a_flow() -> None:
    seams, store = Seams(stored=FakeCredentials()), MemoryTokenStore()

    token = await make_auth(seams, store).access_token()

    assert token == "fake-access"
    assert seams.flow_runs == []


async def test_expired_credentials_are_refreshed_and_persisted() -> None:
    seams, store = Seams(stored=FakeCredentials(valid=False)), MemoryTokenStore()

    token = await make_auth(seams, store).access_token()

    assert token == "fake-refreshed"
    assert seams.stored is not None and seams.stored.refreshed == 1
    assert store.load(STORE_KEY) == {
        "token": "fake-refreshed",
        "refresh_token": "fake-refresh",
        "scopes": list(SCOPES),
    }


async def test_rejected_refresh_clears_the_store_and_reconsents() -> None:
    seams = Seams(stored=FakeCredentials(valid=False), refresh_error=AuthError("google", "revoked"))
    store = MemoryTokenStore()

    token = await make_auth(seams, store).access_token()

    assert token == "fake-consented"
    assert seams.flow_runs == [(Path("client.json"), SCOPES)]
    assert store.load(STORE_KEY) is not None  # re-saved after consent


async def test_missing_refresh_token_or_scopes_triggers_the_flow() -> None:
    no_refresh = Seams(stored=FakeCredentials(valid=False, refresh_token=None))
    narrower = Seams(stored=FakeCredentials(scopes=("https://www.googleapis.com/auth/other",)))

    await make_auth(no_refresh, MemoryTokenStore()).access_token()
    await make_auth(narrower, MemoryTokenStore()).access_token()

    assert len(no_refresh.flow_runs) == 1
    assert len(narrower.flow_runs) == 1


async def test_empty_store_runs_the_flow_once_then_reuses() -> None:
    seams, store = Seams(), MemoryTokenStore()
    auth = make_auth(seams, store)

    first = await auth.access_token()
    seams.stored = seams.flow_result  # what load() now returns from the store
    second = await auth.access_token()

    assert first == second == "fake-consented"
    assert len(seams.flow_runs) == 1


async def test_consent_without_token_is_an_auth_error() -> None:
    seams = Seams(flow_result=FakeCredentials(token=None))

    with pytest.raises(AuthError, match="without an access token"):
        await make_auth(seams, MemoryTokenStore()).access_token()


def test_from_settings_requires_the_secrets_file(
    make_settings: Callable[..., Settings], tmp_path: Path
) -> None:
    with pytest.raises(NotConfiguredError, match="GOOGLE_CLIENT_SECRETS_FILE"):
        GoogleAuth.from_settings(make_settings(), SCOPES, MemoryTokenStore())

    auth = GoogleAuth.from_settings(
        make_settings(google_client_secrets_file=tmp_path / "c.json"), SCOPES, MemoryTokenStore()
    )
    assert auth.scopes == SCOPES


def test_default_seams_map_library_errors_without_network(tmp_path: Path) -> None:
    credentials = load_credentials(
        {"refresh_token": "fake-refresh", "client_id": "x", "client_secret": "y"}, SCOPES
    )
    assert credentials.refresh_token == "fake-refresh"

    class Rejecting:
        def refresh(self, _request: Any) -> None:
            raise RefreshError("invalid_grant")

    with pytest.raises(AuthError, match="token refresh rejected"):
        refresh_credentials(Rejecting())  # type: ignore[arg-type]
    with pytest.raises(AuthError, match="OAuth consent failed"):
        run_installed_app_flow(tmp_path / "missing.json", SCOPES)
