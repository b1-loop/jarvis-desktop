from collections.abc import Callable

import pytest

from jarvis.config import DEFAULT_STT_HOTWORDS, Settings

MakeSettings = Callable[..., Settings]


def test_defaults_pick_the_safe_barge_in_mode_and_per_device_models(
    make_settings: MakeSettings,
) -> None:
    settings = make_settings()

    assert settings.barge_in_mode == "auto"
    assert settings.whisper_model(gpu=True) == "large-v3-turbo"
    assert settings.whisper_model(gpu=False) == "small"
    assert settings.stt_hotwords == DEFAULT_STT_HOTWORDS
    assert "Discord" in DEFAULT_STT_HOTWORDS


def test_an_explicit_stt_model_pins_both_devices(make_settings: MakeSettings) -> None:
    settings = make_settings(stt_model="medium")

    assert settings.whisper_model(gpu=True) == settings.whisper_model(gpu=False) == "medium"


def test_the_env_file_spelling_is_accepted(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JARVIS_BARGE_IN_MODE", "wake_word")
    monkeypatch.setenv("JARVIS_STT_MODEL_GPU", "medium")
    monkeypatch.setenv("JARVIS_SPEAKING_WAKE_MARGIN", "0.05")

    settings = Settings(_env_file=None)

    assert settings.barge_in_mode == "wake_word"
    assert settings.whisper_model(gpu=True) == "medium"
    assert settings.speaking_wake_margin == pytest.approx(0.05)


def test_unknown_barge_in_modes_are_rejected(make_settings: MakeSettings) -> None:
    with pytest.raises(ValueError, match="barge_in_mode"):
        make_settings(barge_in_mode="sometimes")
