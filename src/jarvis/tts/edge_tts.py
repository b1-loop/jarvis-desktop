"""edge-tts backed ``TTS`` (the CLOUD-mode default).

edge-tts streams MP3 from Microsoft's neural voices over a websocket. The
MP3 bytes are decoded natively (``jarvis_audio.decode_mp3``) in batches of
whole frames as they arrive, so the first half second of a sentence is
playing while the rest is still downloading. Both the edge-tts stream and
the decoder are injected, which keeps the batching logic unit-testable
without network or a native build.
"""

from __future__ import annotations

import asyncio
import importlib
import time
from collections.abc import AsyncIterator, Mapping
from typing import TYPE_CHECKING, Any, Final, Protocol

import numpy as np

from jarvis import _native
from jarvis.tts.base import TTS, AudioChunk, Samples, SynthesisError
from jarvis.tts.mp3_frames import (
    DEFAULT_BATCH_FRAMES,
    DEFAULT_OVERLAP_FRAMES,
    DecodeBatch,
    Mp3Batcher,
)
from jarvis.utils.logging import get_logger

if TYPE_CHECKING:
    from jarvis.config import Settings

EDGE_TTS_SAMPLE_RATE: Final = 24_000
"""edge-tts always returns 24 kHz mono MP3 ("audio-24khz-48kbitrate-mono-mp3")."""

log = get_logger(__name__)


class AudioChunkStream(Protocol):
    """Opens an edge-tts style stream of chunk dicts for ``text`` in ``voice``.

    Chunks with ``type == "audio"`` carry MP3 bytes under ``data``; other
    types (word and sentence boundaries) are ignored.
    """

    def __call__(self, text: str, voice: str) -> AsyncIterator[Mapping[str, Any]]:
        """Start streaming."""


class Mp3Decoder(Protocol):
    """Decodes a complete MP3 buffer to ``(mono float32 samples, sample_rate)``."""

    def __call__(self, data: bytes) -> tuple[Samples, int]:
        """Decode; raises on data that holds no decodable frame."""


class EdgeTTS(TTS):
    """``TTS`` over edge-tts with native, batched MP3 decoding."""

    def __init__(
        self,
        voice: str,
        *,
        stream_factory: AudioChunkStream,
        decoder: Mp3Decoder,
        batch_frames: int = DEFAULT_BATCH_FRAMES,
        overlap_frames: int = DEFAULT_OVERLAP_FRAMES,
    ) -> None:
        self._voice = voice
        self._stream_factory = stream_factory
        self._decoder = decoder
        self._batch_frames = batch_frames
        self._overlap_frames = overlap_frames

    @classmethod
    def from_settings(cls, settings: Settings) -> EdgeTTS:
        """Build the real provider: edge-tts over the network, native decoder."""
        return cls(settings.tts_voice, stream_factory=edge_tts_stream, decoder=native_mp3_decoder())

    @property
    def voice(self) -> str:
        """The edge-tts voice id, e.g. ``en-GB-RyanNeural``."""
        return self._voice

    @property
    def sample_rate(self) -> int:
        """Always 24 kHz for edge-tts."""
        return EDGE_TTS_SAMPLE_RATE

    async def synthesize(self, text: str) -> AsyncIterator[AudioChunk]:
        """Stream ``text`` as PCM chunks of roughly half a second each.

        Raises:
            SynthesisError: If edge-tts fails, returns no audio, or the audio
                is not the expected 24 kHz mono MP3.
        """
        if not text.strip():
            return
        batcher = Mp3Batcher(batch_frames=self._batch_frames, overlap_frames=self._overlap_frames)
        started = time.perf_counter()
        first_chunk_ms: int | None = None
        async for raw in self._stream_factory(text, self._voice):
            if raw.get("type") != "audio":
                continue
            batch = batcher.push(raw["data"])
            if batch is not None:
                yield await self._decode(batch)
                first_chunk_ms = first_chunk_ms or round((time.perf_counter() - started) * 1000)
        final = batcher.finish()
        if final is not None:
            yield await self._decode(final)
            first_chunk_ms = first_chunk_ms or round((time.perf_counter() - started) * 1000)
        if batcher.frames_seen == 0:
            raise SynthesisError("edge-tts returned no audio; check the voice id and the network")
        log.info(
            "tts.synthesized",
            voice=self._voice,
            chars=len(text),
            frames=batcher.frames_seen,
            skipped_bytes=batcher.skipped_bytes,
            first_chunk_ms=first_chunk_ms,
            total_ms=round((time.perf_counter() - started) * 1000),
        )

    async def _decode(self, batch: DecodeBatch) -> AudioChunk:
        try:
            samples, rate = await asyncio.to_thread(self._decoder, batch.data)
        except Exception as error:
            raise SynthesisError(f"MP3 decoding failed: {error}") from error
        if rate != EDGE_TTS_SAMPLE_RATE:
            raise SynthesisError(f"expected {EDGE_TTS_SAMPLE_RATE} Hz from edge-tts, got {rate}")
        if batch.keep_samples is None:
            return AudioChunk(samples, rate)
        if batch.keep_samples > samples.size:
            raise SynthesisError("decoder returned fewer samples than the new frames hold")
        return AudioChunk(samples[samples.size - batch.keep_samples :], rate)


async def edge_tts_stream(text: str, voice: str) -> AsyncIterator[Mapping[str, Any]]:
    """The real ``AudioChunkStream``: ``edge_tts.Communicate`` with errors translated.

    Raises:
        SynthesisError: On any edge-tts or network failure, with the cause attached.
    """
    # Imported lazily by name: an optional dependency text-only mode never needs.
    edge_tts = importlib.import_module("edge_tts")
    exceptions = importlib.import_module("edge_tts.exceptions")
    try:
        async for chunk in edge_tts.Communicate(text, voice).stream():
            yield chunk
    except (exceptions.EdgeTTSException, OSError) as error:
        raise SynthesisError(f"edge-tts failed for voice '{voice}': {error}") from error


def native_mp3_decoder() -> Mp3Decoder:
    """The real ``Mp3Decoder``: ``jarvis_audio.decode_mp3`` from the native build.

    The module is loaded on the first decode, not here, so building the
    provider never requires the native build; only speaking does.

    Raises:
        NativeUnavailableError: From the first call, if the extension module
            is not built.
    """

    def decode(data: bytes) -> tuple[Samples, int]:
        samples, rate = _native.load("jarvis_audio").decode_mp3(data)
        return np.asarray(samples, dtype=np.float32), int(rate)

    return decode


# Design notes:
# - edge-tts is imported by name inside the stream function so that
#   text-only mode and the test suite never pay for (or require) the
#   dependency; importlib keeps the lazy import honest under every ruff
#   version the repo runs (the pre-commit hook pins an older one).
# - Chunks are decoded in asyncio.to_thread even though a batch takes well
#   under a millisecond natively: the rule is that nothing CPU-bound runs on
#   the loop, and the thread hop is cheaper than an exception to the rule.
# - The sample-rate check is a guard against a silently changed service
#   format: the playback device is opened at 24 kHz and would otherwise play
#   garbled speech rather than fail loudly.
