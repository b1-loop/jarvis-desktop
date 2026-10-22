"""user_settings.json: precedence over .env, persistence and validation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from jarvis.config import HudPosition, Settings
from jarvis.user_settings import (
    UserSettingsError,
    UserSettingsStore,
    editable_fields,
    user_settings_path,
)


def write_user_settings(data_dir: Path, values: dict[str, object]) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    user_settings_path(data_dir).write_text(json.dumps(values), encoding="utf-8")


def write_env(path: Path, lines: str) -> Path:
    path.write_text(lines, encoding="utf-8")
    return path


def load(data_dir: Path, env_file: Path | None = None, **kwargs: object) -> Settings:
    return Settings(
        _env_file=env_file,  # type: ignore[call-arg]
        data_dir=data_dir,
        user_settings_enabled=True,
        **kwargs,
    )


def test_user_settings_override_dotenv(tmp_path: Path) -> None:
    env = write_env(tmp_path / ".env", "JARVIS_TTS_VOICE=en-GB-RyanNeural\n")
    write_user_settings(tmp_path, {"tts_voice": "sv-SE-MattiasNeural"})

    assert load(tmp_path, env).tts_voice == "sv-SE-MattiasNeural"


def test_environment_variables_override_user_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_user_settings(tmp_path, {"overlay_theme": "amber"})
    monkeypatch.setenv("JARVIS_OVERLAY_THEME", "cyan")

    assert load(tmp_path).overlay_theme == "cyan"


def test_init_kwargs_override_everything(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    write_user_settings(tmp_path, {"overlay_size": "compact"})
    monkeypatch.setenv("JARVIS_OVERLAY_SIZE", "normal")

    assert load(tmp_path, overlay_size="large").overlay_size == "large"


def test_defaults_apply_when_nobody_sets_a_value(tmp_path: Path) -> None:
    write_user_settings(tmp_path, {})
    assert load(tmp_path).speak_replies is True


def test_data_dir_from_dotenv_locates_the_file(tmp_path: Path) -> None:
    data = tmp_path / "elsewhere"
    write_user_settings(data, {"speak_replies": False})
    env = write_env(tmp_path / ".env", f"JARVIS_DATA_DIR={data}\n")

    settings = Settings(_env_file=env, user_settings_enabled=True)  # type: ignore[call-arg]

    assert settings.speak_replies is False


def test_the_file_is_ignored_when_disabled(tmp_path: Path) -> None:
    write_user_settings(tmp_path, {"overlay_theme": "amber"})
    settings = Settings(_env_file=None, data_dir=tmp_path)  # type: ignore[call-arg]
    # The suite's environment switches user settings off (conftest).
    assert settings.overlay_theme == "cyan"


def test_an_invalid_value_is_skipped_not_fatal(tmp_path: Path) -> None:
    write_user_settings(
        tmp_path,
        {"wake_word_threshold": 7.0, "overlay_theme": "purple", "speak_replies": False},
    )
    settings = load(tmp_path)

    assert settings.wake_word_threshold == 0.8
    assert settings.overlay_theme == "cyan"
    assert settings.speak_replies is False


def test_unknown_keys_secrets_and_locating_fields_are_ignored(tmp_path: Path) -> None:
    write_user_settings(
        tmp_path,
        {"no_such_setting": 1, "anthropic_api_key": "sk-leak", "data_dir": str(tmp_path / "other")},
    )
    settings = load(tmp_path)

    assert settings.anthropic_api_key is None
    assert settings.data_dir == tmp_path
    assert "anthropic_api_key" not in editable_fields(Settings)


def test_a_corrupt_file_falls_back_to_dotenv(tmp_path: Path) -> None:
    tmp_path.mkdir(exist_ok=True)
    user_settings_path(tmp_path).write_text("{not json", encoding="utf-8")
    env = write_env(tmp_path / ".env", "JARVIS_OVERLAY_THEME=amber\n")

    assert load(tmp_path, env).overlay_theme == "amber"


def test_positions_round_trip_through_the_file(tmp_path: Path) -> None:
    position = {"monitor": "\\\\.\\DISPLAY2", "x": 40.0, "y": 60.0}
    write_user_settings(tmp_path, {"overlay_position": position})

    assert load(tmp_path).overlay_position == HudPosition(monitor="\\\\.\\DISPLAY2", x=40, y=60)


def test_store_saves_atomically_and_updates_one_key(tmp_path: Path) -> None:
    store = UserSettingsStore(user_settings_path(tmp_path / "data"))
    assert store.load() == {}

    store.update("overlay_theme", "amber")
    store.update("speak_replies", False)

    assert store.load() == {"overlay_theme": "amber", "speak_replies": False}
    assert not store.path.with_name(store.path.name + ".tmp").exists()


def test_store_rejects_a_file_that_is_not_an_object(tmp_path: Path) -> None:
    path = user_settings_path(tmp_path)
    path.write_text("[1, 2]", encoding="utf-8")

    with pytest.raises(UserSettingsError):
        UserSettingsStore(path).load()
