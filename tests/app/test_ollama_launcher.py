from collections.abc import Sequence
from pathlib import Path

import httpx
import pytest

from jarvis.app import ollama_launcher
from jarvis.app.ollama_launcher import (
    DetachedProcessLauncher,
    HttpOllamaProbe,
    LaunchTiming,
    OllamaLauncher,
    OllamaStartError,
    OllamaStatus,
    create_ollama_launcher,
    find_ollama_command,
)


class FakeProbe:
    """Answers "down" for the first ``down_for`` checks, then "up"."""

    def __init__(self, down_for: int) -> None:
        self.down_for = down_for
        self.checks: list[float] = []

    async def is_up(self, wait_seconds: float) -> bool:
        self.checks.append(wait_seconds)
        return len(self.checks) > self.down_for


class FakeLauncher:
    def __init__(self, error: OSError | None = None) -> None:
        self.spawned: list[tuple[str, ...]] = []
        self.error = error

    def spawn(self, argv: Sequence[str]) -> None:
        if self.error is not None:
            raise self.error
        self.spawned.append(tuple(argv))


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.now += seconds


COMMAND = ("C:/Ollama/ollama app.exe",)


def make_launcher(
    probe: FakeProbe,
    launcher: FakeLauncher,
    command: tuple[str, ...] | None = COMMAND,
) -> tuple[OllamaLauncher, FakeClock]:
    clock = FakeClock()
    return (
        OllamaLauncher(
            probe,
            launcher,
            lambda: command,
            timing=LaunchTiming(probe_timeout=2.0, start_timeout=20.0, poll_interval=0.5),
            sleep=clock.sleep,
            clock=clock,
        ),
        clock,
    )


async def test_running_server_is_left_alone() -> None:
    launcher = FakeLauncher()
    subject, _ = make_launcher(FakeProbe(down_for=0), launcher)

    assert await subject.ensure_running() is OllamaStatus.ALREADY_RUNNING
    assert launcher.spawned == []


async def test_down_server_is_started_and_waited_for() -> None:
    probe = FakeProbe(down_for=4)
    launcher = FakeLauncher()
    subject, clock = make_launcher(probe, launcher)

    assert await subject.ensure_running() is OllamaStatus.STARTED
    assert launcher.spawned == [COMMAND]
    assert probe.checks == [2.0] * 5
    assert clock.now == pytest.approx(2.0)


async def test_server_that_never_answers_times_out_with_the_reason() -> None:
    subject, clock = make_launcher(FakeProbe(down_for=1_000), FakeLauncher())

    with pytest.raises(OllamaStartError, match="did not answer within 20 s"):
        await subject.ensure_running()
    assert clock.now == pytest.approx(20.0)


async def test_missing_ollama_is_reported_without_spawning() -> None:
    launcher = FakeLauncher()
    subject, _ = make_launcher(FakeProbe(down_for=1), launcher, command=None)

    with pytest.raises(OllamaStartError, match="not installed"):
        await subject.ensure_running()
    assert launcher.spawned == []


async def test_spawn_failure_is_reported() -> None:
    subject, _ = make_launcher(FakeProbe(down_for=1), FakeLauncher(OSError("access denied")))

    with pytest.raises(OllamaStartError, match="access denied"):
        await subject.ensure_running()


# -- finding the command ----------------------------------------------------------------


def test_tray_app_is_preferred_when_installed(tmp_path: Path) -> None:
    app = tmp_path / "Programs" / "Ollama" / "ollama app.exe"
    app.parent.mkdir(parents=True)
    app.write_bytes(b"")

    command = find_ollama_command({"LOCALAPPDATA": str(tmp_path)}, lambda _name: "/bin/ollama")

    assert command == (str(app),)


def test_ollama_serve_on_path_is_the_fallback(tmp_path: Path) -> None:
    command = find_ollama_command({"LOCALAPPDATA": str(tmp_path)}, lambda _name: "/bin/ollama")

    assert command == ("/bin/ollama", "serve")


def test_nothing_installed_is_none() -> None:
    assert find_ollama_command({}, lambda _name: None) is None


# -- real adapters, without real processes or sockets -----------------------------------


async def test_http_probe_reports_up_on_success_and_down_on_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/tags"
        if request.url.host == "down":
            raise httpx.ConnectError("refused", request=request)
        return httpx.Response(200 if request.url.host == "up" else 500, json={})

    real_client = httpx.AsyncClient

    def client(**kwargs: object) -> httpx.AsyncClient:
        return real_client(transport=httpx.MockTransport(handler), **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(ollama_launcher.httpx, "AsyncClient", client)

    assert await HttpOllamaProbe("http://up:11434/").is_up(1.0)
    assert not await HttpOllamaProbe("http://down:11434").is_up(1.0)
    assert not await HttpOllamaProbe("http://broken:11434").is_up(1.0)


def test_detached_launcher_hides_and_retries_without_breakaway(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[int] = []

    def popen(argv: list[str], **kwargs: int) -> None:
        calls.append(kwargs["creationflags"])
        if len(calls) == 1 and ollama_launcher._BREAKAWAY:
            raise PermissionError("job forbids breakaway")

    monkeypatch.setattr(ollama_launcher.subprocess, "Popen", popen)

    DetachedProcessLauncher().spawn(["ollama", "serve"])

    assert calls[-1] == ollama_launcher._HIDDEN_DETACHED
    assert calls[0] == ollama_launcher._HIDDEN_DETACHED | ollama_launcher._BREAKAWAY


def test_factory_builds_a_launcher() -> None:
    assert isinstance(create_ollama_launcher("http://localhost:11434", 5.0), OllamaLauncher)
