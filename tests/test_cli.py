from typer.testing import CliRunner

from jarvis import __version__
from jarvis.__main__ import EXIT_NOT_IMPLEMENTED, LaunchOptions, app, load_settings
from jarvis.config import RuntimeMode

runner = CliRunner()


def test_version_flag_prints_version_and_exits_cleanly() -> None:
    result = runner.invoke(app, ["--version"])

    assert result.exit_code == 0
    assert __version__ in result.output


def test_help_lists_every_launch_flag() -> None:
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    for flag in ("--mode", "--text-only", "--no-wakeword", "--version"):
        assert flag in result.output


def test_default_launch_reports_resolved_configuration() -> None:
    result = runner.invoke(app, [])

    assert result.exit_code == EXIT_NOT_IMPLEMENTED
    assert "mode=cloud" in result.output
    assert "text_only=False" in result.output
    assert "wake_word=True" in result.output


def test_flags_are_reflected_in_the_resolved_configuration() -> None:
    result = runner.invoke(app, ["--mode", "LOCAL", "--text-only", "--no-wakeword"])

    assert result.exit_code == EXIT_NOT_IMPLEMENTED
    assert "mode=local" in result.output
    assert "text_only=True" in result.output
    assert "wake_word=False" in result.output


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
