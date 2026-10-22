import asyncio
from pathlib import Path

from pydantic import SecretStr

from jarvis.integrations.tokens import (
    FileTokenStore,
    MemoryTokenStore,
    StaticTokenSource,
    ThreadedTokenSource,
    TokenSource,
)


def test_file_store_round_trips_and_clears(tmp_path: Path) -> None:
    store = FileTokenStore(tmp_path / "tokens")

    assert store.load("google") is None
    store.save("google", {"refresh_token": "fake-refresh", "scopes": ["a"]})

    assert store.load("google") == {"refresh_token": "fake-refresh", "scopes": ["a"]}
    assert store.path_for("google") == tmp_path / "tokens" / "google.json"
    assert not (tmp_path / "tokens" / "google.json.part").exists()

    store.clear("google")
    store.clear("google")  # idempotent
    assert store.load("google") is None


def test_file_store_treats_corrupt_file_as_absent(tmp_path: Path) -> None:
    store = FileTokenStore(tmp_path)
    store.path_for("x").write_text("{not json", encoding="utf-8")

    assert store.load("x") is None


def test_memory_store_hands_out_copies() -> None:
    store = MemoryTokenStore()
    original = {"access_token": "fake"}
    store.save("k", original)
    original["access_token"] = "changed"

    loaded = store.load("k")
    assert loaded == {"access_token": "fake"}
    assert loaded is not None
    loaded["access_token"] = "mutated"
    assert store.load("k") == {"access_token": "fake"}

    store.clear("k")
    assert store.load("k") is None


async def test_static_token_source_returns_the_secret() -> None:
    source = StaticTokenSource(SecretStr("fake-static"))

    assert isinstance(source, TokenSource)
    assert await source.access_token() == "fake-static"


class CountingSource(ThreadedTokenSource):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def _acquire_blocking(self) -> str:
        self.calls += 1
        return f"token-{self.calls}"


async def test_threaded_source_serialises_concurrent_acquisition() -> None:
    source = CountingSource()

    tokens = await asyncio.gather(*(source.access_token() for _ in range(5)))

    assert source.calls == 5
    assert tokens == [f"token-{i}" for i in range(1, 6)]
