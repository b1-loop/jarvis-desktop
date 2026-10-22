"""Smoke test on the real devices: microphone engine, playback and barge-in arming.

Never runs in CI (``hardware``); run it locally with
``uv run pytest -m hardware tests/core/test_voice_hardware.py``.
"""

import asyncio
from pathlib import Path

import pytest

from jarvis import _native
from jarvis.audio.engine import AudioEngine, Speaker, create_native_engine, create_native_playback
from jarvis.audio.models import AudioModelPaths
from jarvis.core.events import EventBus
from jarvis.core.interruption import BargeInController
from jarvis.core.state_machine import StateMachine
from jarvis.core.voice_io import Earcon, make_earcon
from jarvis.utils.errors import NativeUnavailableError

pytestmark = [pytest.mark.hardware, pytest.mark.native]

MODELS = Path("models")
RUN_SECONDS = 3.0


@pytest.fixture
def paths() -> AudioModelPaths:
    try:
        _native.load("jarvis_audio")
    except NativeUnavailableError:
        pytest.skip("jarvis_audio is not built")
    paths = AudioModelPaths.in_directory(MODELS)
    if paths.missing():
        pytest.skip("audio models not downloaded; run scripts/download_models.py")
    return paths


async def test_engine_and_playback_run_together_for_a_few_seconds(paths: AudioModelPaths) -> None:
    engine = AudioEngine(create_native_engine(paths, min_speech_ms=250))
    speaker = Speaker(create_native_playback(24_000))
    barge_in = BargeInController(
        engine=engine,
        playback=speaker.native,
        state=StateMachine(EventBus()),
        echo_guard_seconds=0.1,
    )
    levels: list[float] = []

    async with engine:
        await Earcon(speaker, make_earcon(speaker.sample_rate))()
        barge_in.playback_started()
        for _ in range(int(RUN_SECONDS / 0.05)):
            levels.append(engine.input_level)
            await asyncio.sleep(0.05)
        assert barge_in.armed
        barge_in.playback_finished()
        engine.discard_pending_events()

    assert not engine.running
    assert all(0.0 <= level <= 1.0 for level in levels)
    assert speaker.stop_count == 0
