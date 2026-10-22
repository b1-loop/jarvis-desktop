import asyncio
from collections.abc import Callable
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any

import pytest

from jarvis.agent.registry import ToolRegistry
from jarvis.agent.tool_embeddings import EmbeddingToolRanker
from jarvis.agent.tool_selection import RelevanceToolSelector
from jarvis.config import Settings
from jarvis.core import bootstrap
from jarvis.core.bootstrap import (
    CONSENT_NOTIFICATION_TITLE,
    MEMORY_DB_FILENAME,
    Overrides,
    build_application,
    build_integrations,
    build_llm,
    build_memory,
    build_platform,
    build_registry,
    build_vector_store,
    consent_notifier,
)
from jarvis.core.events import EventBus
from jarvis.integrations.base import ConsentRequiredError
from jarvis.integrations.stubs import StubCalendarProvider, StubEmailProvider
from jarvis.llm.ollama_client import OllamaClient
from jarvis.memory import InMemoryVectorStore
from jarvis.memory.errors import MemoryStoreError
from jarvis.os_integration.errors import PlatformError
from jarvis.os_integration.stub import StubPlatform
from jarvis.proactive.scheduler import Scheduler
from jarvis.utils.errors import ConfigError, NativeUnavailableError
from tests.agent.llm_fakes import ScriptedLLM, text_round
from tests.core.fakes import RecordingOutput, ScriptedInput

MakeSettings = Callable[..., Settings]


def ollama_settings(make_settings: MakeSettings, tmp_path: Path, **extra: Any) -> Settings:
    return make_settings(llm_provider="ollama", data_dir=tmp_path, **extra)


async def test_tool_selection_is_hybrid_with_an_embedding_model(
    make_settings: MakeSettings, tmp_path: Path
) -> None:
    # A closed port: the background warm-up fails fast and the turn falls back to words.
    settings = ollama_settings(
        make_settings,
        tmp_path,
        tool_embedding_model="bge-m3",
        tool_embedding_timeout_ms=40,
        ollama_base_url="http://127.0.0.1:9",
    )
    async with AsyncExitStack() as stack:
        selector = bootstrap.build_tool_selector(settings, stack, ToolRegistry.discover())

        assert isinstance(selector, RelevanceToolSelector)
        assert isinstance(selector.semantic, EmbeddingToolRanker)
        assert selector.semantic.budget_seconds == pytest.approx(0.04)
        assert selector.semantic.cache is not None
        assert selector.semantic.cache.path == tmp_path / "cache" / "tool_embeddings.json"


@pytest.mark.parametrize(
    "overrides",
    [{"tool_embedding_model": None}, {"llm_provider": "anthropic"}],
    ids=["no-model", "claude-gets-every-tool"],
)
async def test_tool_selection_is_words_only_without_a_model_or_a_cap(
    make_settings: MakeSettings, tmp_path: Path, overrides: dict[str, Any]
) -> None:
    settings = make_settings(**{"data_dir": tmp_path, "tool_embedding_model": "bge-m3"} | overrides)
    async with AsyncExitStack() as stack:
        selector = bootstrap.build_tool_selector(settings, stack, ToolRegistry.discover())

        assert isinstance(selector, RelevanceToolSelector)
        assert selector.semantic is None


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


async def test_integrations_are_empty_without_accounts(
    make_settings: MakeSettings, tmp_path: Path
) -> None:
    async with AsyncExitStack() as stack:
        providers = build_integrations(make_settings(data_dir=tmp_path), stack)

    assert providers.email == providers.calendar == ()
    assert providers.music == providers.chat == providers.smart_home == ()


async def test_voice_and_background_sign_in_is_non_interactive_and_notifies_once(
    make_settings: MakeSettings, tmp_path: Path
) -> None:
    # A Google account configured but never connected: the windowless app
    # must answer "run connect_accounts.py", never open a browser.
    (tmp_path / "client.json").write_text("{}", encoding="utf-8")
    settings = make_settings(data_dir=tmp_path, google_client_secrets_file=tmp_path / "client.json")
    platform = StubPlatform()
    async with AsyncExitStack() as stack:
        services = build_integrations(settings, stack, platform=platform)
        gmail = services.email[0]
        for _ in range(2):
            with pytest.raises(ConsentRequiredError, match=r"connect_accounts\.py google"):
                await gmail.list_unread()

    assert platform.notifications == [
        (
            CONSENT_NOTIFICATION_TITLE,
            "Google isn't connected yet; run "
            "`uv run python scripts/connect_accounts.py google` in a terminal.",
        )
    ]


async def test_a_failing_notification_does_not_hide_the_consent_error() -> None:
    class Refusing(StubPlatform):
        async def notify(self, title: str, message: str) -> None:
            raise PlatformError("no notification area")

    error = ConsentRequiredError("google", reason="no stored token", remedy="connect it")

    await consent_notifier(Refusing())(error)  # logged, not raised


def test_overrides_default_to_non_interactive_sign_in() -> None:
    assert Overrides().interactive_sign_in is False


async def test_integrations_serve_stubs_only_in_demo_mode(
    make_settings: MakeSettings, tmp_path: Path
) -> None:
    async with AsyncExitStack() as stack:
        providers = build_integrations(
            make_settings(data_dir=tmp_path, demo_integrations=True), stack
        )

    assert isinstance(providers.email[0], StubEmailProvider)
    assert isinstance(providers.calendar[0], StubCalendarProvider)


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


# --- tool services, proactive and UI wiring --------------------------------


async def _never(_prompt: str) -> bool:
    return False


def offline_overrides(source: ScriptedInput | None = None) -> Overrides:
    return Overrides(
        source=source or ScriptedInput(["exit"]),
        sink=RecordingOutput(),
        llm=ScriptedLLM(),
        platform=StubPlatform(),
        vectors=InMemoryVectorStore(),
        registry=ToolRegistry(),
    )


def test_vision_degrades_to_none_without_a_usable_provider(
    monkeypatch: pytest.MonkeyPatch, make_settings: MakeSettings
) -> None:
    def no_key(_settings: Settings, _platform: object) -> object:
        raise ConfigError("ANTHROPIC_API_KEY is required")

    monkeypatch.setattr(bootstrap, "create_screen_reader", no_key)

    assert bootstrap.build_vision(make_settings(), StubPlatform()) is None


async def test_tool_services_share_one_jobs_tracker_and_vision(
    make_settings: MakeSettings, tmp_path: Path
) -> None:
    settings = ollama_settings(make_settings, tmp_path)
    async with AsyncExitStack() as stack:
        memory = await build_memory(settings, stack, vectors=InMemoryVectorStore())
        services = bootstrap.build_tool_services(settings, stack, StubPlatform(), memory)
        context = bootstrap.build_tool_context(settings, EventBus(), _never, services)

    assert context.jobs is services.jobs
    assert context.vision is services.vision
    assert services.vision is not None
    assert context.memory is memory


def test_battery_source_is_skipped_without_psutil(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing(name: str) -> object:
        raise ImportError(f"No module named {name!r}")

    monkeypatch.setattr(bootstrap.importlib, "import_module", missing)

    assert bootstrap.build_battery_source() is None


def test_battery_source_uses_the_battery_tool_probe() -> None:
    assert bootstrap.build_battery_source() is not None


async def test_proactive_scheduler_runs_with_the_app_and_stops_after(
    monkeypatch: pytest.MonkeyPatch, make_settings: MakeSettings, tmp_path: Path
) -> None:
    routines = tmp_path / "routines"
    routines.mkdir()
    (routines / "evening.yaml").write_text(
        'name: evening\ntrigger: { kind: time, at: "21:00" }\nsteps:\n  - tool: current_time\n',
        encoding="utf-8",
    )
    settings = ollama_settings(
        make_settings, tmp_path, proactive_enabled=True, routines_dir=routines
    )
    schedulers: list[Scheduler | None] = []
    original = bootstrap.build_proactive

    async def spy(*args: Any, **kwargs: Any) -> Scheduler | None:
        schedulers.append(await original(*args, **kwargs))
        return schedulers[-1]

    monkeypatch.setattr(bootstrap, "build_proactive", spy)
    application = await build_application(settings, offline_overrides())
    scheduler = schedulers[0]
    assert scheduler is not None and scheduler.running
    names = {job.name for job in scheduler.jobs}
    assert {"trigger:due_reminder", "trigger:upcoming_meeting", "routine:evening"} <= names

    await application.run()

    assert not scheduler.running


async def test_proactive_can_be_switched_off(make_settings: MakeSettings, tmp_path: Path) -> None:
    settings = ollama_settings(make_settings, tmp_path)
    async with AsyncExitStack() as stack:
        memory = await build_memory(settings, stack, vectors=InMemoryVectorStore())
        services = bootstrap.build_tool_services(settings, stack, StubPlatform(), memory)
        scheduler = await bootstrap.build_proactive(
            settings,
            stack,
            bus=EventBus(),
            sink=RecordingOutput(),
            services=services,
            routines=bootstrap.build_routines(settings),
            invoker=None,  # type: ignore[arg-type]
        )

    assert scheduler is None


async def test_quit_from_the_tray_ends_the_session_cleanly(
    make_settings: MakeSettings, tmp_path: Path
) -> None:
    class Blocking(ScriptedInput):
        async def read_line(self, prompt: str) -> str | None:
            await asyncio.Event().wait()
            return None

    overrides = offline_overrides(Blocking())
    application = await build_application(ollama_settings(make_settings, tmp_path), overrides)
    running = asyncio.create_task(application.run())
    await asyncio.sleep(0.05)

    application.request_stop()
    await asyncio.wait_for(running, 5)

    assert not running.cancelled()


async def test_ctrl_c_still_propagates(make_settings: MakeSettings, tmp_path: Path) -> None:
    class Blocking(ScriptedInput):
        async def read_line(self, prompt: str) -> str | None:
            await asyncio.Event().wait()
            return None

    application = await build_application(
        ollama_settings(make_settings, tmp_path), offline_overrides(Blocking())
    )
    running = asyncio.create_task(application.run())
    await asyncio.sleep(0.05)
    running.cancel()

    with pytest.raises(asyncio.CancelledError):
        await running
