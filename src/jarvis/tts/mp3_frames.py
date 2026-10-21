"""Splitting an MP3 byte stream into whole frames, and batching them for decoding.

edge-tts delivers MP3 as a sequence of network chunks that need not align
with frame boundaries. The native decoder (``jarvis_audio.decode_mp3``)
decodes a complete buffer and keeps no state between calls, so the Python
side has to (a) hand it only whole frames and (b) hide the fact that an MP3
frame depends on its predecessors. ``Mp3FrameSplitter`` does (a) by parsing
frame headers; ``Mp3Batcher`` does (b) by re-decoding a few already-played
frames in front of every batch and discarding their output.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

_SYNC_BYTE: Final = 0xFF
_SYNC_MASK: Final = 0xE0
"""The top three bits of the second header byte complete the 11-bit sync word."""
_HEADER_BYTES: Final = 4
_LAYER_III: Final = 1
_VERSION_RESERVED: Final = 1
_VERSION_MPEG1: Final = 3
_BITRATE_FREE: Final = 0
_BITRATE_INVALID: Final = 15
_SAMPLE_RATE_INVALID: Final = 3
_CHANNEL_MODE_MONO: Final = 3

_BITRATES_KBPS: Final[dict[bool, tuple[int, ...]]] = {
    True: (0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320),
    False: (0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160),
}
"""Layer III bitrate tables, keyed by "is MPEG-1"."""

_SAMPLE_RATES: Final[dict[int, tuple[int, int, int]]] = {
    0: (11025, 12000, 8000),  # MPEG-2.5
    2: (22050, 24000, 16000),  # MPEG-2
    3: (44100, 48000, 32000),  # MPEG-1
}
"""Sample-rate tables keyed by the version bits."""

_SAMPLES_PER_FRAME: Final[dict[bool, int]] = {True: 1152, False: 576}
"""Layer III frame length in samples, keyed by "is MPEG-1"."""

_SLOTS_PER_FRAME: Final[dict[bool, int]] = {True: 144, False: 72}
"""Layer III frame-size coefficient, keyed by "is MPEG-1"."""

DEFAULT_BATCH_FRAMES: Final = 20
"""Frames per decode batch: 20 x 24 ms = ~0.5 s of audio per chunk at 24 kHz."""

DEFAULT_OVERLAP_FRAMES: Final = 6
"""Frames re-decoded in front of a batch; see the design notes."""


@dataclass(frozen=True, slots=True)
class Mp3FrameHeader:
    """The parts of a Layer III frame header the splitter needs.

    Attributes:
        length: Frame size in bytes, header included.
        samples: PCM samples per channel this frame decodes to.
        sample_rate: Output rate in Hz.
        channels: 1 for mono, else 2.
    """

    length: int
    samples: int
    sample_rate: int
    channels: int


@dataclass(frozen=True, slots=True)
class Mp3Frame:
    """One complete frame and what it decodes to."""

    data: bytes
    header: Mp3FrameHeader


@dataclass(frozen=True, slots=True)
class DecodeBatch:
    """Bytes to decode and which part of the decoder's output is new.

    Attributes:
        data: Whole frames, the overlap (already played) ones first.
        keep_samples: How many samples at the *end* of the decoded output
            belong to the new frames, or ``None`` when there is no overlap
            and everything is new. Counted from the end because a decoder
            started cold may emit nothing for the first overlap frames (no
            bit reservoir yet) and the exact number it drops is its own
            business.
    """

    data: bytes
    keep_samples: int | None


def parse_header(data: bytes | bytearray, offset: int = 0) -> Mp3FrameHeader | None:
    """Parse the Layer III frame header at ``offset``, or ``None`` if there is none.

    Free-format and reserved field values are treated as "no header" so a
    byte pattern that merely looks like a sync word is rejected.
    """
    if offset + _HEADER_BYTES > len(data):
        return None
    first, second, third, fourth = data[offset : offset + _HEADER_BYTES]
    if first != _SYNC_BYTE or (second & _SYNC_MASK) != _SYNC_MASK:
        return None
    version = (second >> 3) & 0x3
    layer = (second >> 1) & 0x3
    bitrate_index = third >> 4
    rate_index = (third >> 2) & 0x3
    if (
        version == _VERSION_RESERVED
        or layer != _LAYER_III
        or bitrate_index in (_BITRATE_FREE, _BITRATE_INVALID)
        or rate_index == _SAMPLE_RATE_INVALID
    ):
        return None
    is_mpeg1 = version == _VERSION_MPEG1
    bitrate = _BITRATES_KBPS[is_mpeg1][bitrate_index] * 1000
    sample_rate = _SAMPLE_RATES[version][rate_index]
    padding = (third >> 1) & 0x1
    length = _SLOTS_PER_FRAME[is_mpeg1] * bitrate // sample_rate + padding
    channels = 1 if (fourth >> 6) == _CHANNEL_MODE_MONO else 2
    return Mp3FrameHeader(length, _SAMPLES_PER_FRAME[is_mpeg1], sample_rate, channels)


class Mp3FrameSplitter:
    """Turns arbitrary byte chunks into whole, validated MP3 frames.

    A frame is released only when the header of the frame after it is also
    present and valid, which is how a real decoder tells a frame from a byte
    pattern that happens to look like one. Bytes that belong to no frame
    (ID3 tags, junk) are skipped and counted.
    """

    def __init__(self) -> None:
        self._buffer = bytearray()
        self._received = 0
        self._emitted = 0

    @property
    def skipped_bytes(self) -> int:
        """Bytes dropped so far because they were not part of any frame."""
        return self._received - self._emitted - len(self._buffer)

    def push(self, data: bytes) -> list[Mp3Frame]:
        """Add ``data`` and return every frame that is now complete and confirmed."""
        self._buffer.extend(data)
        self._received += len(data)
        frames: list[Mp3Frame] = []
        position = 0
        while (found := self._next_frame(position)) is not None:
            start, header = found
            end = start + header.length
            if end + _HEADER_BYTES > len(self._buffer):
                position = start  # the confirming header is not here yet
                break
            if parse_header(self._buffer, end) is None:
                position = start + 1  # false sync; keep scanning
                continue
            frames.append(self._take(start, header))
            position = end
        else:
            # No header from `position` on; keep the last bytes, they may be
            # the beginning of one.
            position = max(position, len(self._buffer) - (_HEADER_BYTES - 1))
        del self._buffer[:position]
        return frames

    def flush(self) -> list[Mp3Frame]:
        """Release a final frame that has no successor to confirm it, then reset."""
        frames: list[Mp3Frame] = []
        found = self._next_frame(0)
        if found is not None:
            start, header = found
            if start + header.length <= len(self._buffer):
                frames.append(self._take(start, header))
        self._buffer.clear()
        return frames

    def _take(self, start: int, header: Mp3FrameHeader) -> Mp3Frame:
        self._emitted += header.length
        return Mp3Frame(bytes(self._buffer[start : start + header.length]), header)

    def _next_frame(self, position: int) -> tuple[int, Mp3FrameHeader] | None:
        for offset in range(position, len(self._buffer) - _HEADER_BYTES + 1):
            header = parse_header(self._buffer, offset)
            if header is not None:
                return offset, header
        return None


class Mp3Batcher:
    """Groups confirmed frames into decode batches with overlap priming.

    Every batch after the first starts with the last ``overlap_frames``
    frames of the previous batch so that the stateless decoder has the bit
    reservoir and filterbank history the first new frame depends on; only
    the output of the new frames is kept (:attr:`DecodeBatch.keep_samples`).
    """

    def __init__(
        self,
        *,
        batch_frames: int = DEFAULT_BATCH_FRAMES,
        overlap_frames: int = DEFAULT_OVERLAP_FRAMES,
    ) -> None:
        if batch_frames < 1 or overlap_frames < 0:
            raise ValueError("batch_frames must be >= 1 and overlap_frames >= 0")
        self._batch_frames = batch_frames
        self._overlap_frames = overlap_frames
        self._splitter = Mp3FrameSplitter()
        self._frames: list[Mp3Frame] = []
        self._emitted = 0

    @property
    def skipped_bytes(self) -> int:
        """Bytes the splitter dropped as non-frame data."""
        return self._splitter.skipped_bytes

    @property
    def frames_seen(self) -> int:
        """Confirmed frames so far (batched or still pending), for logging."""
        return self._emitted + len(self._frames) - self._kept_overlap()

    def push(self, data: bytes) -> DecodeBatch | None:
        """Add raw bytes; return a batch once enough new frames are available."""
        self._frames.extend(self._splitter.push(data))
        if len(self._frames) - self._kept_overlap() >= self._batch_frames:
            return self._take_batch()
        return None

    def finish(self) -> DecodeBatch | None:
        """Flush the splitter and return the final batch, if any frames remain."""
        self._frames.extend(self._splitter.flush())
        if len(self._frames) > self._kept_overlap():
            return self._take_batch()
        return None

    def _kept_overlap(self) -> int:
        return min(self._overlap_frames, self._emitted)

    def _take_batch(self) -> DecodeBatch:
        overlap = self._kept_overlap()
        new_samples = sum(frame.header.samples for frame in self._frames[overlap:])
        keep = new_samples if overlap else None
        batch = DecodeBatch(b"".join(frame.data for frame in self._frames), keep)
        self._emitted += len(self._frames) - overlap
        keep = min(self._overlap_frames, len(self._frames))
        self._frames = self._frames[len(self._frames) - keep :]
        return batch


# Design notes:
# - Why not a stateful native decoder fed incrementally: miniaudio's dr_mp3
#   is pull-based and treats a short read as end of stream, so it cannot be
#   paused mid-stream to wait for the next network chunk. Overlap priming
#   gives the same output from a stateless decode-a-buffer call.
# - Six overlap frames: the Layer III bit reservoir holds at most 511 bytes,
#   so after four 144-byte frames the decoder's reservoir is byte-identical
#   to a continuous decode; the remaining two frames decode exactly and
#   warm up the synthesis filterbank. The native test compares a batched
#   decode of a real clip against a whole-file decode to prove it.
# - dr_mp3 emits nothing for a frame whose reservoir it does not have yet
#   rather than garbage, which is why the batch records how much output to
#   keep from the end instead of how much to discard from the start.
# - Layer III only: edge-tts (and every TTS service we might swap in)
#   produces it; supporting Layers I/II would add tables nobody exercises.
