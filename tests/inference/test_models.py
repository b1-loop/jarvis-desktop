from pathlib import Path

from jarvis.audio.models import AUDIO_MODELS
from jarvis.inference.models import (
    INFERENCE_MODELS,
    QWEN2_5_0_5B_INSTRUCT,
    WHISPER_TINY,
    InferenceModelPaths,
)


def test_every_model_is_pinned_by_revision_and_full_sha256() -> None:
    for model in INFERENCE_MODELS:
        assert len(model.sha256) == 64
        assert model.url.startswith("https://huggingface.co/")
        assert "/resolve/main/" not in model.url  # pinned to a revision, not a branch
        assert model.size_bytes > 0


def test_model_names_do_not_collide_with_the_audio_manifest() -> None:
    names = [model.name for model in AUDIO_MODELS + INFERENCE_MODELS]
    assert len(names) == len(set(names))


def test_paths_follow_the_standard_layout(tmp_path: Path) -> None:
    paths = InferenceModelPaths.in_directory(tmp_path)

    assert paths.whisper == tmp_path / WHISPER_TINY.name
    assert paths.llama == tmp_path / QWEN2_5_0_5B_INSTRUCT.name


def test_missing_lists_only_absent_files(tmp_path: Path) -> None:
    paths = InferenceModelPaths.in_directory(tmp_path)
    paths.whisper.write_bytes(b"x")

    assert paths.missing() == (paths.llama,)
