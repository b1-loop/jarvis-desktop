import asyncio
from collections.abc import AsyncIterator
from pathlib import Path

import numpy as np
import pytest

from jarvis.audio.engine import (
    AudioEngine,
    AudioEventKind,
    Speaker,
    create_native_engine,
)
from jarvis.audio.models import AudioModelPaths
from jarvis.utils.errors import ConfigError
from tests.audio.fakes import FakeNativeEngine, FakeNativePlayback, native_event

POLL = 0.001
WAIT = 1.0


@pytest.fixture
def native() -> FakeNativeEngine:
    return FakeNativeEngine()


@pytest.fixture
def engine(native: FakeNativeEngine) -> AudioEngine:
    return AudioEngine(native, poll_interval=POLL)


async def test_start_and_stop_drive_the_native_engine_off_the_loop(
    engine: AudioEngine, native: FakeNativeEngine
) -> None:
    assert not engine.running

    await engine.start()
    assert engine.running
    assert native.started == 1

    await engine.stop()
    assert not engine.running
    assert native.stopped == 1


async def test_start_and_stop_are_idempotent(engine: AudioEngine, native: FakeNativeEngine) -> None:
    await engine.start()
    await engine.start()
    await engine.stop()
    await engine.stop()

    assert native.started == 1
    assert native.stopped == 1


async def test_context_manager_starts_and_stops(native: FakeNativeEngine) -> None:
    async with AudioEngine(native, poll_interval=POLL) as engine:
        assert engine.running
    assert native.stopped == 1


async def test_native_events_are_converted_and_delivered(
    engine: AudioEngine, native: FakeNativeEngine
) -> None:
    native.script(native_event("WAKE_WORD_DETECTED", 0.93, 4096))
    async with engine:
        event = await asyncio.wait_for(engine.next_event(), WAIT)

    assert event.kind is AudioEventKind.WAKE_WORD_DETECTED
    assert event.value == pytest.approx(0.93)
    assert event.sample_position == 4096


async def test_events_iterator_yields_in_order(
    engine: AudioEngine, native: FakeNativeEngine
) -> None:
    native.script(native_event("SPEECH_STARTED"), native_event("SPEECH_ENDED"))
    async with engine:
        iterator = engine.events()
        first = await asyncio.wait_for(anext(iterator), WAIT)
        second = await asyncio.wait_for(anext(iterator), WAIT)

    assert [first.kind, second.kind] == [
        AudioEventKind.SPEECH_STARTED,
        AudioEventKind.SPEECH_ENDED,
    ]


async def test_next_event_skips_kinds_the_caller_did_not_ask_for(
    engine: AudioEngine, native: FakeNativeEngine
) -> None:
    native.script(native_event("SAMPLES_DROPPED", 128), native_event("SPEECH_ENDED"))
    async with engine:
        event = await asyncio.wait_for(engine.next_event(AudioEventKind.SPEECH_ENDED), WAIT)

    assert event.kind is AudioEventKind.SPEECH_ENDED


async def test_next_event_can_be_bounded_with_asyncio_timeout(engine: AudioEngine) -> None:
    async with engine:
        with pytest.raises(TimeoutError):
            async with asyncio.timeout(0.02):
                await engine.next_event()


async def test_capture_utterance_listens_until_speech_ends(
    engine: AudioEngine, native: FakeNativeEngine
) -> None:
    native.captured = np.full(1600, 0.25, dtype=np.float32)
    native.batches.append([])  # one empty poll before speech ends
    native.script(native_event("SPEECH_ENDED", 0.1, 8000))

    async with engine:
        audio = await asyncio.wait_for(engine.capture_utterance(), WAIT)

    assert audio.size == 1600
    assert native.listening_calls == [True, False]
    assert not engine.listening


async def test_capture_utterance_returns_partial_audio_on_timeout(
    native: FakeNativeEngine,
) -> None:
    native.captured = np.ones(100, dtype=np.float32)
    engine = AudioEngine(native, poll_interval=POLL, max_utterance_seconds=0.02)

    async with engine:
        audio = await engine.capture_utterance()

    assert audio.size == 100
    assert native.listening_calls == [True, False]


async def test_a_crashing_poller_is_logged_and_surfaces(
    native: FakeNativeEngine,
) -> None:
    native.script(native_event("NOT_A_KIND"))
    engine = AudioEngine(native, poll_interval=POLL)

    async with engine:
        await asyncio.sleep(POLL * 20)
        poller = engine._poller
        assert poller is not None
        with pytest.raises(KeyError):
            await poller


async def test_enable_barge_in_forwards_the_target(
    engine: AudioEngine, native: FakeNativeEngine
) -> None:
    playback = FakeNativePlayback()

    engine.enable_barge_in(playback)
    engine.enable_barge_in(None)

    assert native.barge_in_targets == [playback, None]


def test_input_level_is_read_through(engine: AudioEngine, native: FakeNativeEngine) -> None:
    native.level = 0.42

    assert engine.input_level == pytest.approx(0.42)


async def chunks_of(*values: float) -> AsyncIterator[np.ndarray]:
    for value in values:
        yield np.full(10, value, dtype=np.float32)


async def test_speaker_plays_every_chunk_and_reports_completion() -> None:
    playback = FakeNativePlayback()
    speaker = Speaker(playback, poll_interval=POLL)

    async def drain_soon() -> None:
        await asyncio.sleep(POLL * 5)
        playback.drain()

    drain = asyncio.create_task(drain_soon())
    finished = await asyncio.wait_for(speaker.play(chunks_of(0.1, 0.2, 0.3)), WAIT)
    await drain

    assert finished
    assert len(playback.chunks) == 3
    assert playback.end_marked
    assert speaker.sample_rate == 24_000


async def test_speaker_stop_interrupts_and_skips_remaining_chunks() -> None:
    playback = FakeNativePlayback()
    speaker = Speaker(playback, poll_interval=POLL)

    async def two_then_stop() -> AsyncIterator[np.ndarray]:
        yield np.zeros(10, dtype=np.float32)
        speaker.stop()
        yield np.zeros(10, dtype=np.float32)
        yield np.zeros(10, dtype=np.float32)

    finished = await asyncio.wait_for(speaker.play(two_then_stop()), WAIT)

    assert not finished
    assert playback.stops == 1
    assert playback.chunks == []  # stop cleared the queue and nothing was added after


async def test_speaker_detects_native_barge_in_while_waiting() -> None:
    playback = FakeNativePlayback()
    speaker = Speaker(playback, poll_interval=POLL)

    async def barge_in_soon() -> None:
        await asyncio.sleep(POLL * 5)
        playback.stop()  # as the native engine would on SpeechStarted

    task = asyncio.create_task(barge_in_soon())
    finished = await asyncio.wait_for(speaker.play(chunks_of(0.5)), WAIT)
    await task

    assert not finished
    assert not speaker.playing


def test_create_native_engine_reports_missing_models(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match=r"silero_vad\.onnx"):
        create_native_engine(AudioModelPaths.in_directory(tmp_path))


def test_create_native_engine_without_wake_word_only_needs_the_vad(tmp_path: Path) -> None:
    with pytest.raises(ConfigError) as caught:
        create_native_engine(AudioModelPaths.in_directory(tmp_path), wake_word=False)

    assert "silero_vad.onnx" in str(caught.value)
    assert "hey_jarvis" not in str(caught.value)


async def test_record_utterance_waits_for_start_then_end(
    engine: AudioEngine, native: FakeNativeEngine
) -> None:
    native.captured = np.ones(320, dtype=np.float32)
    native.script(native_event("SPEECH_STARTED"))
    native.script(native_event("SPEECH_ENDED"))

    async with engine:
        utterance = await asyncio.wait_for(engine.record_utterance(start_timeout=WAIT), WAIT)

    assert utterance.speech_detected
    assert not utterance.truncated
    assert utterance.samples.size == 320
    assert native.listening_calls == [True, False]


async def test_record_utterance_gives_up_when_nobody_speaks(
    engine: AudioEngine, native: FakeNativeEngine
) -> None:
    async with engine:
        utterance = await engine.record_utterance(start_timeout=0.02)

    assert not utterance.speech_detected
    assert native.listening_calls == [True, False]


async def test_record_utterance_marks_truncated_speech(native: FakeNativeEngine) -> None:
    native.script(native_event("SPEECH_STARTED"))
    engine = AudioEngine(native, poll_interval=POLL, max_utterance_seconds=0.05)

    async with engine:
        utterance = await engine.record_utterance(start_timeout=WAIT)

    assert utterance.speech_detected
    assert utterance.truncated


async def test_discard_pending_events_drops_stale_events(
    engine: AudioEngine, native: FakeNativeEngine
) -> None:
    native.script(native_event("SPEECH_STARTED"), native_event("SPEECH_ENDED"))
    async with engine:
        await asyncio.sleep(POLL * 20)
        assert engine.discard_pending_events() == 2
        assert engine.discard_pending_events() == 0


def test_speaker_exposes_output_level_and_stop_count() -> None:
    playback = FakeNativePlayback(level=0.3)
    speaker = Speaker(playback)
    speaker.stop()

    assert speaker.output_level == pytest.approx(0.3)
    assert speaker.stop_count == 1
