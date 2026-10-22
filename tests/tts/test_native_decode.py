"""Tests against the compiled ``jarvis_audio.decode_mp3`` and the batching pipeline."""

from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from jarvis import _native
from jarvis.tts.edge_tts import EdgeTTS, native_mp3_decoder
from jarvis.tts.mp3_frames import DEFAULT_OVERLAP_FRAMES
from jarvis.utils.errors import NativeUnavailableError
from tests.tts.fakes import FakeChunkStream

pytestmark = pytest.mark.native

FIXTURE = Path(__file__).parent.parent / "fixtures" / "edge_tts_hello.mp3"


@pytest.fixture
def jarvis_audio() -> ModuleType:
    try:
        return _native.load("jarvis_audio")
    except NativeUnavailableError:
        pytest.skip("jarvis_audio is not built")


def test_decode_mp3_returns_mono_float32_at_24k(jarvis_audio: ModuleType) -> None:
    samples, rate = jarvis_audio.decode_mp3(FIXTURE.read_bytes())

    assert rate == 24_000
    assert samples.dtype == np.float32
    assert samples.ndim == 1
    assert samples.size > 12_000
    assert 0.05 < float(np.abs(samples).max()) <= 1.0


def test_decode_mp3_rejects_garbage_with_decode_error(jarvis_audio: ModuleType) -> None:
    assert issubclass(jarvis_audio.DecodeError, jarvis_audio.NativeError)

    with pytest.raises(jarvis_audio.DecodeError):
        jarvis_audio.decode_mp3(b"definitely not an mp3 file")


async def test_batched_decoding_with_overlap_matches_a_whole_file_decode(
    jarvis_audio: ModuleType,
) -> None:
    data = FIXTURE.read_bytes()
    whole, _ = jarvis_audio.decode_mp3(data)
    chunks = [data[i : i + 720] for i in range(0, len(data), 720)]
    tts = EdgeTTS(
        "v", stream_factory=FakeChunkStream(chunks), decoder=native_mp3_decoder(), batch_frames=10
    )

    pieces = [chunk.samples async for chunk in tts.synthesize("Hello from Jarvis.")]
    batched = np.concatenate(pieces)

    assert len(pieces) > 1
    assert batched.size == whole.size
    # Overlap priming must reproduce the continuous decode; a tiny tolerance
    # covers float rounding in the synthesis filterbank.
    np.testing.assert_allclose(batched, whole, atol=1e-3)
    assert DEFAULT_OVERLAP_FRAMES >= 4  # the bit reservoir can reach 3.5 frames back
