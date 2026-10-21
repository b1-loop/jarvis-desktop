from collections.abc import Callable
from pathlib import Path

import pytest
from pydantic import ValidationError

from jarvis.config import RuntimeMode, Settings
from jarvis.utils.errors import ConfigError


def test_defaults_are_cloud_mode_with_no_credentials(
    make_settings: Callable[..., Settings],
) -> None:
    settings = make_settings()

    assert settings.mode is RuntimeMode.CLOUD
    assert settings.wake_word == "hey jarvis"
    assert settings.anthropic_api_key is None
    assert settings.data_dir == Path("data")


def test_jarvis_prefixed_variables_are_read(
    monkeypatch: pytest.MonkeyPatch, make_settings: Callable[..., Settings]
) -> None:
    monkeypatch.setenv("JARVIS_MODE", "local")
    monkeypatch.setenv("JARVIS_WAKE_WORD", "computer")
    monkeypatch.setenv("JARVIS_LOG_LEVEL", "debug")

    settings = make_settings()

    assert settings.mode is RuntimeMode.LOCAL
    assert settings.wake_word == "computer"
    assert settings.log_level == "DEBUG"


def test_credentials_keep_their_conventional_names(
    monkeypatch: pytest.MonkeyPatch, make_settings: Callable[..., Settings]
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-test")

    settings = make_settings()

    assert settings.anthropic_api_key is not None
    assert settings.anthropic_api_key.get_secret_value() == "sk-ant-test"
    assert settings.slack_bot_token is not None


def test_secrets_never_appear_in_repr(make_settings: Callable[..., Settings]) -> None:
    settings = make_settings(anthropic_api_key="sk-ant-very-secret")

    assert "sk-ant-very-secret" not in repr(settings)
    assert "sk-ant-very-secret" not in str(settings)


def test_unknown_mode_is_rejected(
    monkeypatch: pytest.MonkeyPatch, make_settings: Callable[..., Settings]
) -> None:
    monkeypatch.setenv("JARVIS_MODE", "hybrid")

    with pytest.raises(ValidationError):
        make_settings()


def test_cloud_mode_requires_anthropic_key(make_settings: Callable[..., Settings]) -> None:
    settings = make_settings(mode=RuntimeMode.CLOUD)

    with pytest.raises(ConfigError, match="ANTHROPIC_API_KEY"):
        settings.validate_for_mode()


def test_local_mode_needs_no_credentials(make_settings: Callable[..., Settings]) -> None:
    make_settings(mode=RuntimeMode.LOCAL).validate_for_mode()


def test_cloud_mode_with_key_validates(make_settings: Callable[..., Settings]) -> None:
    make_settings(mode=RuntimeMode.CLOUD, anthropic_api_key="sk-ant-test").validate_for_mode()


def test_proactive_settings_have_defaults_and_read_env(
    monkeypatch: pytest.MonkeyPatch, make_settings: Callable[..., Settings]
) -> None:
    assert make_settings().routines_dir == Path("routines")
    assert make_settings().proactive_poll_seconds == 60

    monkeypatch.setenv("JARVIS_ROUTINES_DIR", "C:/jarvis/routines")
    monkeypatch.setenv("JARVIS_PROACTIVE_POLL_SECONDS", "15")
    settings = make_settings()

    assert settings.routines_dir == Path("C:/jarvis/routines")
    assert settings.proactive_poll_seconds == 15
    with pytest.raises(ValidationError):
        make_settings(proactive_poll_seconds=0)
