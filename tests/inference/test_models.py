from pathlib import Path

from jarvis.audio.models import AUDIO_MODELS
from jarvis.config import Settings
from jarvis.inference.models import (
    INFERENCE_MODELS,
    PIPER_VOICE_EN_GB,
    PIPER_VOICE_EN_GB_CONFIG,
    QWEN2_5_0_5B_INSTRUCT,
    SWEDISH_VOICE_MODELS,
    WHISPER_TINY,
    InferenceModelPaths,
    piper_config_path,
)


def test_every_model_is_pinned_by_revision_and_full_sha256() -> None:
    for model in INFERENCE_MODELS + SWEDISH_VOICE_MODELS:
        assert len(model.sha256) == 64
        assert model.url.startswith("https://huggingface.co/")
        assert "/resolve/main/" not in model.url  # pinned to a revision, not a branch
        assert model.size_bytes > 0


def test_model_names_do_not_collide_with_the_audio_manifest() -> None:
    names = [model.name for model in AUDIO_MODELS + INFERENCE_MODELS + SWEDISH_VOICE_MODELS]
    assert len(names) == len(set(names))


def test_paths_follow_the_standard_layout(tmp_path: Path) -> None:
    paths = InferenceModelPaths.in_directory(tmp_path)

    assert paths.whisper == tmp_path / WHISPER_TINY.name
    assert paths.llama == tmp_path / QWEN2_5_0_5B_INSTRUCT.name
    assert paths.piper_voice == tmp_path / PIPER_VOICE_EN_GB.name
    assert paths.piper_config == tmp_path / PIPER_VOICE_EN_GB_CONFIG.name


def test_missing_lists_only_absent_files(tmp_path: Path) -> None:
    paths = InferenceModelPaths.in_directory(tmp_path)
    paths.whisper.write_bytes(b"x")
    paths.piper_voice.write_bytes(b"x")

    assert paths.missing() == (paths.llama, paths.piper_config)


def test_piper_config_sits_next_to_the_voice(tmp_path: Path) -> None:
    assert piper_config_path(tmp_path / "v.onnx") == tmp_path / "v.onnx.json"
    assert piper_config_path(PIPER_VOICE_EN_GB.path(tmp_path)).name == PIPER_VOICE_EN_GB_CONFIG.name


def test_settings_defaults_point_at_the_pinned_manifest_files() -> None:
    settings = Settings(_env_file=None)
    expected = InferenceModelPaths.in_directory(settings.models_dir)

    assert settings.local_stt_model == expected.whisper
    assert settings.local_llm_model == expected.llama
    assert settings.local_tts_model == expected.piper_voice


def test_the_swedish_voice_and_its_config_are_one_download_pair(tmp_path: Path) -> None:
    voice, config = SWEDISH_VOICE_MODELS

    assert piper_config_path(voice.path(tmp_path)) == config.path(tmp_path)
