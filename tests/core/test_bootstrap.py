from collections.abc import Callable
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any

import pytest

from jarvis.agent.registry import ToolRegistry
from jarvis.config import Settings
from jarvis.core import bootstrap
from jarvis.core.bootstrap import (
    MEMORY_DB_FILENAME,
    Overrides,
    build_application,
    build_integrations,
    build_llm,
    build_memory,
    build_platform,
    build_registry,
    build_vector_store,
)
from jarvis.integrations.stubs import StubCalendarProvider, StubEmailProvider
from jarvis.llm.ollama_client import OllamaClient
from jarvis.memory import InMemoryVectorStore
from jarvis.memory.errors import MemoryStoreError
from jarvis.os_integration.stub import StubPlatform
from jarvis.utils.errors import ConfigError, NativeUnavailableError
from tests.agent.llm_fakes import ScriptedLLM, text_round
from tests.core.fakes import RecordingOutput, ScriptedInput

MakeSettings = Callable[..., Settings]


def ollama_settings(make_settings: MakeSettings, tmp_path: Path, **extra: Any) -> Settings:
    return make_settings(llm_provider="ollama", data_dir=tmp_path, **extra)


async def test_platform_falls_back_to_the_stub_without_the_native_build(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing() -> Any:
        raise NativeUnavailableError("jarvis_os", "not built")

    monkeypatch.setattr(bootstrap, "create_windows_platform", missing)
    async with AsyncExitStack() as stack:
        platform = build_platform(stack)

    assert isinstance(platform, StubPlatform)


async def test_platform_uses_the_native_backend_when_it_loads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    native = StubPlatform()
    monkeypatch.setattr(bootstrap, "create_windows_platform", lambda: native)
    async with AsyncExitStack() as stack:
        assert build_platform(stack) is native


async def test_vector_store_without_chromadb_is_in_memory(
    monkeypatch: pytest.MonkeyPatch, make_settings: MakeSettings, tmp_path: Path
) -> None:
    monkeypatch.setattr(bootstrap, "chroma_available", lambda: False)
    async with AsyncExitStack() as stack:
        store = await build_vector_store(make_settings(data_dir=tmp_path), stack)

    assert isinstance(store, InMemoryVectorStore)


class FakeChroma:
    instances: list["FakeChroma"] = []  # noqa: RUF012 - test bookkeeping
    fail = False

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.closed = False
        FakeChroma.instances.append(self)

    async def __aenter__(self) -> "FakeChroma":
        if FakeChroma.fail:
            raise MemoryStoreError("chroma is corrupt")
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        self.closed = True


@pytest.mark.parametrize("fail", [False, True])
async def test_vector_store_prefers_chroma_and_degrades_when_it_fails(
    monkeypatch: pytest.MonkeyPatch, make_settings: MakeSettings, tmp_path: Path, fail: bool
) -> None:
    monkeypatch.setattr(bootstrap, "chroma_available", lambda: True)
    monkeypatch.setattr(bootstrap, "ChromaVectorStore", FakeChroma)
    monkeypatch.setattr(FakeChroma, "fail", fail)
    async with AsyncExitStack() as stack:
        store = await build_vector_store(make_settings(data_dir=tmp_path), stack)

    if fail:
        assert isinstance(store, InMemoryVectorStore)
    else:
        assert isinstance(store, FakeChroma)
        assert store.directory == tmp_path / "chroma"
        assert store.closed


async def test_memory_lives_under_data_dir_and_summarises_on_close(
    make_settings: MakeSettings, tmp_path: Path
) -> None:
    async with AsyncExitStack() as stack:
        memory = await build_memory(
            make_settings(data_dir=tmp_path), stack, vectors=InMemoryVectorStore()
        )
        await memory.record_turn("hello", "Good evening.")
        assert memory.conversation is not None

    assert (tmp_path / MEMORY_DB_FILENAME).exists()
    assert memory.conversation is None


async def test_llm_comes_from_the_factory_and_is_closed(
    make_settings: MakeSettings, tmp_path: Path
) -> None:
    async with AsyncExitStack() as stack:
        client = build_llm(ollama_settings(make_settings, tmp_path), stack)
        assert isinstance(client, OllamaClient)
        http = client._http

    assert http.is_closed


def test_registry_discovers_the_bundled_tools() -> None:
    registry = build_registry()

    assert "current_time" in registry


async def test_integrations_default_to_stubs(make_settings: MakeSettings, tmp_path: Path) -> None:
    async with AsyncExitStack() as stack:
        providers = build_integrations(make_settings(data_dir=tmp_path), stack)

    assert isinstance(providers.email[0], StubEmailProvider)
    assert isinstance(providers.calendar[0], StubCalendarProvider)
    assert len(providers.music) == len(providers.chat) == len(providers.smart_home) == 1


async def test_application_runs_a_conversation_and_releases_resources(
    make_settings: MakeSettings, tmp_path: Path
) -> None:
    output = RecordingOutput()
    platform = StubPlatform()
    overrides = Overrides(
        source=ScriptedInput(["hello", "exit"]),
        sink=output,
        llm=ScriptedLLM(rounds=[text_round("Good evening, sir.")]),
        platform=platform,
        vectors=InMemoryVectorStore(),
        registry=ToolRegistry(),
    )
    application = await build_application(make_settings(data_dir=tmp_path), overrides)

    await application.run()

    assert output.said == ["Good evening, sir.", "Goodbye."]
    assert application.agent.persona.name == "Jarvis"
    assert application.agent.context.integrations is not None
    assert (tmp_path / MEMORY_DB_FILENAME).exists()


async def test_missing_credentials_fail_before_anything_opens(
    make_settings: MakeSettings, tmp_path: Path
) -> None:
    with pytest.raises(ConfigError, match="ANTHROPIC_API_KEY"):
        await build_application(make_settings(data_dir=tmp_path, llm_provider="anthropic"))

    assert not (tmp_path / MEMORY_DB_FILENAME).exists()


async def test_failure_half_way_closes_what_was_opened(
    monkeypatch: pytest.MonkeyPatch, make_settings: MakeSettings, tmp_path: Path
) -> None:
    def broken_registry() -> ToolRegistry:
        raise ConfigError("tools package missing")

    monkeypatch.setattr(bootstrap, "build_registry", broken_registry)
    settings = ollama_settings(make_settings, tmp_path)

    with pytest.raises(ConfigError, match="tools package missing"):
        await build_application(settings, Overrides(vectors=InMemoryVectorStore()))

    # The database was opened and closed again: the file can be removed.
    (tmp_path / MEMORY_DB_FILENAME).unlink()
