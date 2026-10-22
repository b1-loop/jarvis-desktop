from pathlib import Path

import pytest
from typer.testing import CliRunner

from jarvis import __version__, _native
from jarvis.__main__ import (
    EXIT_STARTUP_FAILED,
    LaunchOptions,
    app,
    load_settings,
)
from jarvis.config import RuntimeMode
from jarvis.core import bootstrap
from jarvis.utils.errors import NativeUnavailableError

runner = CliRunner()


@pytest.fixture
def offline_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Ollama provider (no key needed, no request before the first turn), data in tmp."""
    monkeypatch.setenv("JARVIS_LLM_PROVIDER", "ollama")
    monkeypatch.setenv("JARVIS_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("JARVIS_LOG_LEVEL", "WARNING")
    monkeypatch.setattr(bootstrap, "chroma_available", lambda: False)
    return tmp_path


def test_version_flag_prints_version_and_exits_cleanly() -> None:
    result = runner.invoke(app, ["--version"])

    assert result.exit_code == 0
    assert __version__ in result.output


def test_help_lists_every_launch_flag() -> None:
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    for flag in ("--mode", "--text-only", "--no-wakeword", "--version"):
        assert flag in result.output


def test_missing_api_key_stops_startup_with_the_remedy(tmp_path: Path) -> None:
    result = runner.invoke(app, ["--text-only"], env={"JARVIS_DATA_DIR": str(tmp_path)})

    assert result.exit_code == EXIT_STARTUP_FAILED
    assert "mode=cloud" in result.output
    assert "ANTHROPIC_API_KEY" in result.output


def test_flags_are_reflected_in_the_resolved_configuration(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        ["--mode", "LOCAL", "--text-only", "--no-wakeword"],
        env={"JARVIS_DATA_DIR": str(tmp_path)},
    )

    assert "mode=local" in result.output
    assert "text_only=True" in result.output
    assert "wake_word=False" in result.output
    assert "Voice mode is unavailable" not in result.output


@pytest.mark.usefixtures("offline_environment")
def test_text_only_session_runs_until_exit() -> None:
    result = runner.invoke(app, ["--text-only"], input="exit\n")

    assert result.exit_code == 0, result.output
    assert "Jarvis> Goodbye." in result.output
    assert "Voice mode is unavailable" not in result.output


@pytest.mark.usefixtures("offline_environment")
def test_voice_mode_without_the_audio_build_falls_back_to_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def not_built(name: str) -> object:
        raise NativeUnavailableError(name, "not built")

    monkeypatch.setattr(_native, "load", not_built)

    result = runner.invoke(app, [], input="")

    assert result.exit_code == 0, result.output
    assert "Voice mode is unavailable" in result.output
    assert "jarvis_audio" in result.output
    assert "Goodbye." in result.output


def test_invalid_mode_is_rejected_by_the_parser() -> None:
    result = runner.invoke(app, ["--mode", "hybrid"])

    assert result.exit_code != 0


def test_mode_flag_overrides_environment() -> None:
    settings = load_settings(RuntimeMode.LOCAL)

    assert settings.mode is RuntimeMode.LOCAL


def test_mode_flag_absent_keeps_environment_value() -> None:
    settings = load_settings(None)

    assert settings.mode is RuntimeMode.CLOUD


def test_launch_options_are_immutable() -> None:
    options = LaunchOptions(mode=RuntimeMode.CLOUD, text_only=True, wake_word_enabled=False)

    try:
        options.text_only = False  # type: ignore[misc]
    except AttributeError:
        return
    raise AssertionError("LaunchOptions must be frozen")
