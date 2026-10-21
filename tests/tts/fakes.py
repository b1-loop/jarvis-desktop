"""Synthetic MP3 frames and fakes for the TTS tests.

Frames use the header edge-tts produces (MPEG-2 Layer III, 48 kbps, 24 kHz,
mono: 144 bytes, 576 samples) and carry their own index in the first payload
byte so a fake decoder can prove which frames a batch contained.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from jarvis.tts.base import Samples

EDGE_HEADER = bytes([0xFF, 0xF3, 0x64, 0xC0])
EDGE_HEADER_PADDED = bytes([0xFF, 0xF3, 0x66, 0xC0])
FRAME_BYTES = 144
FRAME_SAMPLES = 576
SAMPLE_RATE = 24_000


def frame(index: int, *, padded: bool = False) -> bytes:
    header = EDGE_HEADER_PADDED if padded else EDGE_HEADER
    length = FRAME_BYTES + (1 if padded else 0)
    return header + bytes([index]) + bytes(length - len(header) - 1)


def frames(count: int) -> bytes:
    return b"".join(frame(i) for i in range(count))


def fake_decode(data: bytes) -> tuple[Samples, int]:
    """Decode synthetic frames: each becomes 576 samples equal to its index."""
    if len(data) % FRAME_BYTES or not data:
        raise ValueError("not whole synthetic frames")
    indices = [data[offset + 4] for offset in range(0, len(data), FRAME_BYTES)]
    return np.repeat(np.array(indices, dtype=np.float32), FRAME_SAMPLES), SAMPLE_RATE


@dataclass
class FakeChunkStream:
    """Scripted edge-tts stream: audio chunks plus boundary events."""

    chunks: list[bytes]
    calls: list[tuple[str, str]] = field(default_factory=list)

    def __call__(self, text: str, voice: str) -> AsyncIterator[Mapping[str, Any]]:
        self.calls.append((text, voice))
        return self._stream()

    async def _stream(self) -> AsyncIterator[Mapping[str, Any]]:
        yield {"type": "WordBoundary", "offset": 0, "duration": 1, "text": "Hello"}
        for chunk in self.chunks:
            yield {"type": "audio", "data": chunk}
        yield {"type": "SentenceBoundary", "offset": 0, "duration": 1, "text": "Hello."}
