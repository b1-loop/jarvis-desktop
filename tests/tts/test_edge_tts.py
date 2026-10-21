from collections.abc import Callable

import edge_tts
import numpy as np
import pytest
from edge_tts.exceptions import NoAudioReceived

from jarvis.config import Settings
from jarvis.tts.base import TTS, AudioChunk, SynthesisError
from jarvis.tts.edge_tts import EDGE_TTS_SAMPLE_RATE, EdgeTTS, edge_tts_stream
from jarvis.utils.errors import JarvisError
from tests.tts.fakes import FRAME_SAMPLES, FakeChunkStream, fake_decode, frames


def make_tts(stream: FakeChunkStream, **kwargs: object) -> EdgeTTS:
    return EdgeTTS("en-GB-RyanNeural", stream_factory=stream, decoder=fake_decode, **kwargs)  # type: ignore[arg-type]


async def speak(tts: EdgeTTS, text: str) -> list[AudioChunk]:
    return [chunk async for chunk in tts.synthesize(text)]


def test_properties_and_hierarchy() -> None:
    tts = make_tts(FakeChunkStream([]))

    assert isinstance(tts, TTS)
    assert tts.voice == "en-GB-RyanNeural"
    assert tts.sample_rate == EDGE_TTS_SAMPLE_RATE
    assert issubclass(SynthesisError, JarvisError)


async def test_audio_is_decoded_in_batches_as_chunks_arrive() -> None:
    stream_bytes = frames(7)
    chunks = [stream_bytes[i : i + 100] for i in range(0, len(stream_bytes), 100)]
    stream = FakeChunkStream(chunks)
    tts = make_tts(stream, batch_frames=3, overlap_frames=2)

    audio = await speak(tts, "Hello there.")

    assert stream.calls == [("Hello there.", "en-GB-RyanNeural")]
    assert len(audio) == 3  # frames 0-2, 3-5, then 6 on finish
    assert all(chunk.sample_rate == EDGE_TTS_SAMPLE_RATE for chunk in audio)
    joined = np.concatenate([chunk.samples for chunk in audio])
    np.testing.assert_array_equal(joined, np.repeat(np.arange(7, dtype=np.float32), FRAME_SAMPLES))
    assert audio[0].duration_seconds == pytest.approx(3 * FRAME_SAMPLES / EDGE_TTS_SAMPLE_RATE)


async def test_empty_text_yields_nothing_and_makes_no_request() -> None:
    stream = FakeChunkStream([frames(2)])

    assert await speak(make_tts(stream), "   ") == []
    assert stream.calls == []


async def test_no_audio_at_all_is_an_error() -> None:
    with pytest.raises(SynthesisError, match="no audio"):
        await speak(make_tts(FakeChunkStream([])), "Hello")


async def test_decoder_failures_become_synthesis_errors() -> None:
    def broken(_data: bytes) -> tuple[np.ndarray, int]:
        raise RuntimeError("DecodeError: garbage")

    tts = EdgeTTS("v", stream_factory=FakeChunkStream([frames(3)]), decoder=broken)

    with pytest.raises(SynthesisError, match="garbage"):
        await speak(tts, "Hello")


async def test_unexpected_sample_rate_is_rejected() -> None:
    def wrong_rate(data: bytes) -> tuple[np.ndarray, int]:
        return fake_decode(data)[0], 16_000

    tts = EdgeTTS("v", stream_factory=FakeChunkStream([frames(3)]), decoder=wrong_rate)

    with pytest.raises(SynthesisError, match="24000"):
        await speak(tts, "Hello")


async def test_a_decoder_that_returns_too_little_audio_is_rejected() -> None:
    def too_short(_data: bytes) -> tuple[np.ndarray, int]:
        return np.zeros(1, dtype=np.float32), EDGE_TTS_SAMPLE_RATE

    tts = EdgeTTS(
        "v",
        stream_factory=FakeChunkStream([frames(4)]),
        decoder=too_short,
        batch_frames=1,
        overlap_frames=1,
    )

    with pytest.raises(SynthesisError, match="fewer samples"):
        await speak(tts, "Hello")


async def test_stream_errors_propagate_as_synthesis_errors() -> None:
    async def failing(_text: str, _voice: str):  # type: ignore[no-untyped-def]
        raise SynthesisError("edge-tts failed for voice 'v': no audio received")
        yield  # pragma: no cover - makes this an async generator

    tts = EdgeTTS("v", stream_factory=failing, decoder=fake_decode)

    with pytest.raises(SynthesisError, match="edge-tts failed"):
        await speak(tts, "Hello")


async def test_real_stream_wrapper_translates_edge_tts_exceptions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailingCommunicate:
        def __init__(self, text: str, voice: str) -> None:
            self.text, self.voice = text, voice

        async def stream(self):  # type: ignore[no-untyped-def]
            raise NoAudioReceived("No audio was received.")
            yield  # pragma: no cover

    monkeypatch.setattr(edge_tts, "Communicate", FailingCommunicate)

    with pytest.raises(SynthesisError, match="No audio was received"):
        async for _ in edge_tts_stream("Hello", "en-GB-RyanNeural"):
            pass


def test_from_settings_uses_the_configured_voice_without_needing_the_native_build(
    make_settings: Callable[..., Settings],
) -> None:
    tts = EdgeTTS.from_settings(make_settings(tts_voice="en-US-AriaNeural"))

    assert tts.voice == "en-US-AriaNeural"
