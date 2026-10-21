from pathlib import Path

from jarvis.audio.models import AUDIO_MODELS, HEY_JARVIS, SILERO_VAD, AudioModelPaths


def test_every_model_is_pinned_by_a_full_sha256() -> None:
    for model in AUDIO_MODELS:
        assert len(model.sha256) == 64
        assert model.url.startswith("https://")
        assert model.size_bytes > 0


def test_paths_follow_the_standard_layout(tmp_path: Path) -> None:
    paths = AudioModelPaths.in_directory(tmp_path)

    assert paths.vad == tmp_path / SILERO_VAD.name
    assert paths.wake_word == tmp_path / HEY_JARVIS.name


def test_missing_lists_only_absent_files(tmp_path: Path) -> None:
    paths = AudioModelPaths.in_directory(tmp_path)
    paths.vad.write_bytes(b"x")

    missing = paths.missing()

    assert paths.vad not in missing
    assert set(missing) == {paths.melspectrogram, paths.embedding, paths.wake_word}
