import asyncio
import json
from collections.abc import Callable, Iterator
from contextlib import AsyncExitStack
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import structlog

from jarvis.app import background
from jarvis.app.background import (
    ALREADY_RUNNING_MESSAGE,
    EXIT_FAILED,
    EXIT_OK,
    BackgroundDeps,
    ErrorTrayControls,
    boot_overlay,
    come_forward,
    hotkey_label,
    listening_message,
    notify,
    run_background,
    run_windowless,
    uses_ollama,
)
from jarvis.app.ollama_launcher import OllamaStartError, OllamaStatus
from jarvis.app.single_instance import HandoverRequest
from jarvis.config import RuntimeMode, Settings
from jarvis.core.bootstrap import Overrides
from jarvis.core.voice_bootstrap import SessionHooks, VoiceUnavailableError
from jarvis.os_integration.errors import PlatformError
from jarvis.os_integration.stub import StubPlatform
from jarvis.utils.logging import close_file_logging

# -- fakes --------------------------------------------------------------------------------


class FakeLock:
    def __init__(self, free: bool = True) -> None:
        self.free = free
        self.released = False

    def acquire(self) -> bool:
        return self.free

    def release(self) -> None:
        self.released = True


class FakeServer:
    def __init__(self, on_request: Callable[[HandoverRequest], None], path: Path) -> None:
        self.on_request = on_request
        self.path = path
        self.started = False
        self.closed = False

    async def start(self) -> None:
        self.started = True

    async def aclose(self) -> None:
        self.closed = True


class FakeTray:
    def __init__(self) -> None:
        self.notices: list[str | None] = []
        self.stopped = False

    def set_notice(self, notice: str | None) -> None:
        self.notices.append(notice)

    async def stop(self) -> None:
        self.stopped = True


class FakeControls:
    def __init__(self) -> None:
        self.tray = FakeTray()
        self.shown = 0
        self.huds = 0

    def show_dashboard(self) -> None:
        self.shown += 1

    def bring_hud_forward(self) -> None:
        self.huds += 1


class FakeApplication:
    """Runs ``during_run`` (a hook for the test), then returns or raises ``error``."""

    def __init__(self, error: Exception | None = None) -> None:
        self.controls = FakeControls()
        self.error = error
        self.during_run: Callable[[], None] = lambda: None

    async def run(self) -> None:
        self.during_run()
        if self.error is not None:
            raise self.error


class FakeOllama:
    def __init__(self, error: OllamaStartError | None = None) -> None:
        self.error = error
        self.calls = 0

    async def ensure_running(self) -> OllamaStatus:
        self.calls += 1
        if self.error is not None:
            raise self.error
        return OllamaStatus.ALREADY_RUNNING


class Harness:
    """Builds ``BackgroundDeps`` from fakes and records what the session did."""

    def __init__(self, application: FakeApplication | None = None) -> None:
        self.lock = FakeLock()
        self.ping_result = False
        self.pinged: list[HandoverRequest] = []
        self.platform = StubPlatform()
        self.application = application or FakeApplication()
        self.build_error: Exception | None = None
        self.built_with: list[tuple[Settings, Overrides]] = []
        self.servers: list[FakeServer] = []
        self.ollama = FakeOllama()
        self.error_tray: FakeTray | None = FakeTray()
        self.tray_callbacks: list[ErrorTrayControls] = []
        self.front_ends: list[tuple[bool, object]] = []
        self.hooks: list[SessionHooks] = []
        self.spawned: list[list[str]] = []
        self.opened: list[Path] = []

    def deps(self) -> BackgroundDeps:
        return BackgroundDeps(
            lock=lambda _path: self.lock,
            ping=self._ping,
            server=self._server,
            platform=self._platform,
            application=self._build,
            front_end=self._front_end,
            ollama=lambda _settings: self.ollama,
            tray=self._tray,
            open_file=self.opened.append,
            spawn=lambda command, _cwd: self.spawned.append(list(command)),
        )

    async def _ping(self, _path: Path, request: HandoverRequest) -> bool:
        self.pinged.append(request)
        return self.ping_result

    def _server(self, on_request: Callable[[HandoverRequest], None], path: Path) -> FakeServer:
        server = FakeServer(on_request, path)
        self.servers.append(server)
        return server

    def _platform(self, _stack: AsyncExitStack) -> StubPlatform:
        return self.platform

    def _front_end(self, wake_word: bool, write: object, hooks: SessionHooks) -> Any:
        self.hooks.append(hooks)
        self.front_ends.append((wake_word, write))
        return "voice-front-end"

    async def _build(self, settings: Settings, overrides: Overrides) -> Any:
        self.built_with.append((settings, overrides))
        if self.build_error is not None:
            raise self.build_error
        return self.application

    async def _tray(self, _bus: object, callbacks: ErrorTrayControls, _settings: object) -> Any:
        self.tray_callbacks.append(callbacks)
        if self.error_tray is not None:
            # The user reads the tooltip, then picks Quit.
            asyncio.get_running_loop().call_soon(callbacks.quit)
        return self.error_tray

    @property
    def messages(self) -> list[str]:
        return [message for _title, message in self.platform.notifications]


@pytest.fixture
def settings(make_settings: Callable[..., Settings], tmp_path: Path) -> Settings:
    return make_settings(data_dir=tmp_path, llm_provider="anthropic")


@pytest.fixture
def harness() -> Harness:
    return Harness()


# -- single instance ---------------------------------------------------------------------


async def test_second_launch_hands_over_and_starts_nothing(
    settings: Settings, harness: Harness
) -> None:
    harness.lock.free = False
    harness.ping_result = True

    assert await run_background(settings, deps=harness.deps()) == EXIT_OK

    assert harness.built_with == []
    assert harness.messages == []
    assert harness.pinged == [HandoverRequest.SHOW]


async def test_a_hud_only_launch_asks_the_running_one_for_the_hud(
    settings: Settings, harness: Harness
) -> None:
    harness.lock.free = False
    harness.ping_result = True

    code = await run_background(settings, request=HandoverRequest.SHOW_HUD, deps=harness.deps())

    assert code == EXIT_OK
    assert harness.pinged == [HandoverRequest.SHOW_HUD]
    assert harness.built_with == []


async def test_second_launch_notifies_when_the_first_does_not_answer(
    settings: Settings, harness: Harness
) -> None:
    harness.lock.free = False

    assert await run_background(settings, deps=harness.deps()) == EXIT_OK

    assert harness.messages == [ALREADY_RUNNING_MESSAGE]
    assert harness.built_with == []


# -- normal start --------------------------------------------------------------------------


async def test_start_builds_voice_mode_with_the_dashboard_hidden_and_notifies(
    settings: Settings, harness: Harness
) -> None:
    assert await run_background(settings, wake_word=False, deps=harness.deps()) == EXIT_OK

    built, overrides = harness.built_with[0]
    assert built.dashboard_start_visible is False
    assert overrides.platform is harness.platform
    assert overrides.front_end == "voice-front-end"
    assert harness.front_ends[0][0] is False
    assert harness.messages == ["Jarvis is listening; say 'Hey Jarvis' or press Ctrl+Alt+J."]
    assert harness.lock.released
    assert harness.servers[0].started and harness.servers[0].closed
    assert harness.spawned == []  # no restart was asked for


async def test_restart_from_the_hud_relaunches_after_the_lock_is_released(
    settings: Settings, harness: Harness
) -> None:
    def ask_for_restart() -> None:
        restart = harness.hooks[0].restart
        assert restart is not None
        restart()
        assert not harness.lock.released  # still running: nothing spawned yet
        assert harness.spawned == []

    harness.application.during_run = ask_for_restart

    assert await run_background(settings, wake_word=False, deps=harness.deps()) == EXIT_OK

    assert harness.lock.released
    assert len(harness.spawned) == 1
    assert harness.spawned[0][-2:] == ["--background", "--no-wakeword"]


async def test_open_log_from_the_hud_opens_the_session_log(
    settings: Settings, harness: Harness
) -> None:
    def open_log() -> None:
        opener = harness.hooks[0].open_log
        assert opener is not None
        opener()

    harness.application.during_run = open_log
    await run_background(settings, wake_word=True, deps=harness.deps())

    assert harness.opened == [settings.data_dir / "logs" / "jarvis.log"]
    assert harness.servers[0].path == settings.data_dir / "jarvis.instance"


async def test_a_later_launch_replays_the_hud_and_opens_the_dashboard(
    settings: Settings, harness: Harness
) -> None:
    application = harness.application

    def second_launch() -> None:  # the server only exists once the session runs
        harness.servers[0].on_request(HandoverRequest.SHOW)

    application.during_run = second_launch

    await run_background(settings, deps=harness.deps())

    assert application.controls.huds == 1
    assert application.controls.shown == 1


async def test_a_later_hud_only_launch_leaves_the_dashboard_closed(
    settings: Settings, harness: Harness
) -> None:
    application = harness.application
    application.during_run = lambda: harness.servers[0].on_request(HandoverRequest.SHOW_HUD)

    await run_background(settings, deps=harness.deps())

    assert application.controls.huds == 1
    assert application.controls.shown == 0


@pytest.mark.parametrize(
    ("request_kind", "dashboards"), [(HandoverRequest.SHOW, 1), (HandoverRequest.SHOW_HUD, 0)]
)
def test_come_forward_always_brings_the_hud(request_kind: HandoverRequest, dashboards: int) -> None:
    controls = FakeControls()

    come_forward(controls, request_kind)

    assert (controls.huds, controls.shown) == (1, dashboards)


async def test_show_request_while_starting_is_harmless(
    settings: Settings, harness: Harness
) -> None:
    async def build(_settings: Settings, _overrides: Overrides) -> Any:
        harness.servers[0].on_request(HandoverRequest.SHOW)  # before the assistant exists
        return harness.application

    deps = replace(harness.deps(), application=build)

    assert await run_background(settings, deps=deps) == EXIT_OK
    assert harness.application.controls.shown == 0
    assert harness.application.controls.huds == 0


# -- failures stay visible -------------------------------------------------------------------


async def test_voice_unavailable_notifies_reason_and_log_then_waits_for_quit(
    settings: Settings, harness: Harness
) -> None:
    harness.build_error = VoiceUnavailableError("no usable microphone: device busy")

    assert await run_background(settings, deps=harness.deps()) == EXIT_FAILED

    (message,) = harness.messages
    assert "no usable microphone: device busy" in message
    assert str(settings.data_dir.resolve() / "logs" / "jarvis.log") in message
    assert harness.error_tray is not None
    assert harness.error_tray.notices == ["Not listening: no usable microphone: device busy"]
    assert harness.error_tray.stopped
    assert harness.lock.released


async def test_error_without_a_tray_exits_after_the_notification(
    settings: Settings, harness: Harness
) -> None:
    harness.build_error = VoiceUnavailableError("jarvis_audio is not built")
    harness.error_tray = None

    assert await run_background(settings, deps=harness.deps()) == EXIT_FAILED
    assert len(harness.messages) == 1


async def test_a_crash_while_running_enters_the_error_tray(settings: Settings) -> None:
    harness = Harness(FakeApplication(error=RuntimeError("audio thread died")))

    assert await run_background(settings, deps=harness.deps()) == EXIT_FAILED

    assert "audio thread died" in harness.messages[-1]
    assert harness.error_tray is not None
    assert harness.error_tray.stopped


async def test_error_tray_menu_explains_and_quits() -> None:
    explained = asyncio.Event()
    quit_requested = asyncio.Event()
    controls = ErrorTrayControls(quit_requested, explain=explained.set)

    controls.show_dashboard()
    controls.toggle_overlay()
    controls.toggle_microphone()
    controls.bring_hud_forward()
    controls.quit()

    assert explained.is_set()
    assert quit_requested.is_set()


async def test_show_from_the_error_tray_notifies_the_reason_again(
    settings: Settings, harness: Harness
) -> None:
    harness.build_error = VoiceUnavailableError("no models")

    async def tray(_bus: object, callbacks: ErrorTrayControls, _settings: object) -> FakeTray:
        loop = asyncio.get_running_loop()
        loop.call_soon(callbacks.show_dashboard)
        loop.call_later(0.05, callbacks.quit)
        return FakeTray()

    assert await run_background(settings, deps=replace(harness.deps(), tray=tray)) == EXIT_FAILED

    assert len(harness.messages) == 2
    assert harness.messages[0] == harness.messages[1]


async def test_a_hud_only_launch_into_the_error_tray_repeats_the_reason(
    settings: Settings, harness: Harness
) -> None:
    harness.build_error = VoiceUnavailableError("no models")

    async def tray(_bus: object, callbacks: ErrorTrayControls, _settings: object) -> FakeTray:
        loop = asyncio.get_running_loop()
        loop.call_soon(harness.servers[0].on_request, HandoverRequest.SHOW_HUD)
        loop.call_later(0.05, callbacks.quit)
        return FakeTray()

    assert await run_background(settings, deps=replace(harness.deps(), tray=tray)) == EXIT_FAILED

    assert len(harness.messages) == 2


async def test_a_failing_notification_is_logged_not_raised() -> None:
    class BrokenPlatform(StubPlatform):
        async def notify(self, title: str, message: str) -> None:
            raise PlatformError("Shell_NotifyIcon failed")

    await notify(BrokenPlatform(), "hello")


# -- Ollama ----------------------------------------------------------------------------------


async def test_ollama_down_is_reported_in_the_notification_and_tooltip(
    make_settings: Callable[..., Settings], tmp_path: Path, harness: Harness
) -> None:
    settings = make_settings(data_dir=tmp_path, llm_provider="ollama")
    harness.ollama = FakeOllama(OllamaStartError("Ollama is not running and is not installed"))

    assert await run_background(settings, deps=harness.deps()) == EXIT_OK

    assert harness.ollama.calls == 1
    assert "Ollama is not running and is not installed" in harness.messages[0]
    assert harness.messages[0].startswith("Jarvis is listening")
    assert harness.application.controls.tray.notices == [
        "LLM offline: Ollama is not running and is not installed"
    ]


async def test_ollama_is_checked_only_when_it_is_the_llm(
    make_settings: Callable[..., Settings], tmp_path: Path, harness: Harness
) -> None:
    await run_background(
        make_settings(data_dir=tmp_path, llm_provider="anthropic"), deps=harness.deps()
    )
    await run_background(
        make_settings(data_dir=tmp_path, llm_provider="ollama", ollama_autostart=False),
        deps=harness.deps(),
    )

    assert harness.ollama.calls == 0


def test_uses_ollama_follows_the_provider_that_build_llm_picks(
    make_settings: Callable[..., Settings],
) -> None:
    assert uses_ollama(make_settings(llm_provider="ollama"))
    assert not uses_ollama(make_settings(llm_provider="anthropic"))
    # LOCAL mode with the inherited default provider runs llama.cpp, not Ollama.
    assert not uses_ollama(make_settings(mode=RuntimeMode.LOCAL))


# -- messages and the overlay hook ------------------------------------------------------------


def test_hotkey_label_uses_windows_capitalisation() -> None:
    assert hotkey_label("ctrl+alt+j") == "Ctrl+Alt+J"
    assert hotkey_label("win + shift + space") == "Win+Shift+Space"


def test_listening_message_names_the_wake_phrase_and_hotkey(
    make_settings: Callable[..., Settings],
) -> None:
    message = listening_message(make_settings(wake_word="hey jarvis", hotkey="ctrl+alt+j"))

    assert message == "Jarvis is listening; say 'Hey Jarvis' or press Ctrl+Alt+J."


class OverlayWithBoot:
    def __init__(self) -> None:
        self.boots = 0

    def play_boot(self) -> None:
        self.boots += 1


@pytest.mark.parametrize(
    ("settings_value", "expected"),
    [
        (SimpleNamespace(), 1),
        (SimpleNamespace(overlay_boot_animation=True), 1),
        (SimpleNamespace(overlay_boot_animation=False), 0),
    ],
)
def test_boot_animation_plays_when_the_overlay_has_one(
    monkeypatch: pytest.MonkeyPatch, settings_value: SimpleNamespace, expected: int
) -> None:
    overlay = OverlayWithBoot()
    monkeypatch.setattr(background, "build_overlay", lambda _settings: overlay)

    assert boot_overlay(settings_value) is overlay  # type: ignore[arg-type]
    assert overlay.boots == expected


def test_overlay_without_a_boot_animation_is_returned_as_is(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    overlay = object()
    monkeypatch.setattr(background, "build_overlay", lambda _settings: overlay)

    assert boot_overlay(SimpleNamespace()) is overlay  # type: ignore[arg-type]


# -- process entry point ------------------------------------------------------------------------


@pytest.fixture
def _file_logging_cleanup() -> Iterator[None]:
    yield
    close_file_logging()
    structlog.reset_defaults()


@pytest.mark.usefixtures("_file_logging_cleanup")
def test_run_windowless_logs_to_the_file_and_returns_the_session_code(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    requests: list[HandoverRequest] = []

    async def session(_settings: Settings, *, wake_word: bool, request: HandoverRequest) -> int:
        background.log.info("inside", wake_word=wake_word)
        requests.append(request)
        return EXIT_OK

    monkeypatch.setattr(background, "run_background", session)

    assert run_windowless(settings, wake_word=False, request=HandoverRequest.SHOW_HUD) == EXIT_OK
    assert requests == [HandoverRequest.SHOW_HUD]

    log_path = settings.data_dir / "logs" / "jarvis.log"
    events = [json.loads(line)["event"] for line in log_path.read_text("utf-8").splitlines()]
    assert events == ["desktop.start", "inside"]


@pytest.mark.usefixtures("_file_logging_cleanup")
def test_run_windowless_logs_an_unexpected_crash_and_fails(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def session(_settings: Settings, **_options: object) -> int:
        raise OSError("data directory is read-only")

    monkeypatch.setattr(background, "run_background", session)

    assert run_windowless(settings) == EXIT_FAILED

    text = (settings.data_dir / "logs" / "jarvis.log").read_text("utf-8")
    assert "desktop.fatal" in text
    assert "data directory is read-only" in text
