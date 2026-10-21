"""Tests against the compiled ``jarvis_audio`` module.

``native``-marked tests need the build but no audio hardware; ``hardware``
tests open real devices and are only run by hand.
"""

import time
from types import ModuleType

import numpy as np
import pytest

from jarvis import _native
from jarvis.audio.models import AudioModelPaths
from jarvis.config import Settings
from jarvis.utils.errors import NativeUnavailableError

pytestmark = pytest.mark.native


@pytest.fixture
def jarvis_audio() -> ModuleType:
    try:
        return _native.load("jarvis_audio")
    except NativeUnavailableError:
        pytest.skip("jarvis_audio is not built")


@pytest.fixture
def model_paths() -> AudioModelPaths:
    paths = AudioModelPaths.in_directory(Settings(_env_file=None).models_dir)
    if paths.missing():
        pytest.skip("audio models not downloaded")
    return paths


def test_module_exposes_format_constants(jarvis_audio: ModuleType) -> None:
    assert jarvis_audio.SAMPLE_RATE == 16_000
    assert jarvis_audio.VAD_FRAME_SAMPLES == 512
    assert jarvis_audio.WAKE_WORD_CHUNK_SAMPLES == 1280
    assert "step 3" in jarvis_audio.IMPLEMENTATION_STATUS


def test_config_objects_carry_the_documented_defaults(jarvis_audio: ModuleType) -> None:
    config = jarvis_audio.EngineConfig()

    assert config.pipeline.segmenter.start_threshold == pytest.approx(0.5)
    assert config.pipeline.segmenter.silence_end_ms == 700
    assert config.pipeline.wake_word_cooldown_ms == 2000
    assert config.pipeline.pre_roll_ms == 300
    assert config.ring_buffer_samples == 16_000 * 10


def test_exceptions_form_a_hierarchy(jarvis_audio: ModuleType) -> None:
    assert issubclass(jarvis_audio.ModelError, jarvis_audio.NativeError)
    assert issubclass(jarvis_audio.AudioDeviceError, jarvis_audio.NativeError)
    assert issubclass(jarvis_audio.NativeError, RuntimeError)


def test_missing_model_raises_model_error(jarvis_audio: ModuleType, tmp_path: object) -> None:
    with pytest.raises(jarvis_audio.ModelError):
        jarvis_audio.AudioEngine(f"{tmp_path}/nope.onnx", None)


def test_engine_loads_models_without_touching_a_device(
    jarvis_audio: ModuleType, model_paths: AudioModelPaths
) -> None:
    engine = jarvis_audio.AudioEngine(
        model_paths.vad,
        (model_paths.melspectrogram, model_paths.embedding, model_paths.wake_word),
    )

    assert not engine.running()
    assert engine.poll_events() == []
    assert engine.take_captured().dtype == np.float32
    assert engine.take_captured().size == 0


@pytest.mark.hardware
def test_microphone_frames_reach_python(
    jarvis_audio: ModuleType, model_paths: AudioModelPaths
) -> None:
    engine = jarvis_audio.AudioEngine(model_paths.vad, None)
    engine.start()
    try:
        engine.set_listening(True)
        time.sleep(1.0)
        engine.set_listening(False)
        captured = engine.take_captured()
    finally:
        engine.stop()

    assert captured.size >= 12_000  # at least 0.75 s of a 1 s window at 16 kHz
    assert engine.dropped_samples() == 0


@pytest.mark.hardware
def test_playback_opens_and_stops(jarvis_audio: ModuleType) -> None:
    playback = jarvis_audio.Playback(16_000)
    tone = (0.1 * np.sin(np.linspace(0, 2 * np.pi * 440 * 0.2, 3200))).astype(np.float32)

    playback.enqueue(tone)
    playback.mark_end_of_stream()
    deadline = time.monotonic() + 2.0
    while playback.is_playing() and time.monotonic() < deadline:
        time.sleep(0.01)

    assert not playback.is_playing()
    assert playback.stop_count() == 0
