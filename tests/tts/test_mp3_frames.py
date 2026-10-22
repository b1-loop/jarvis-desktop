from pathlib import Path

import pytest

from jarvis.tts.mp3_frames import Mp3Batcher, Mp3FrameSplitter, parse_header
from tests.tts.fakes import EDGE_HEADER, FRAME_BYTES, FRAME_SAMPLES, frame, frames

FIXTURE = Path(__file__).parent.parent / "fixtures" / "edge_tts_hello.mp3"
MPEG1_STEREO_128K = bytes([0xFF, 0xFB, 0x90, 0x00])  # 44.1 kHz, 128 kbps, stereo


def test_parse_header_reads_the_edge_tts_format() -> None:
    header = parse_header(EDGE_HEADER)

    assert header is not None
    assert header.length == FRAME_BYTES
    assert header.samples == FRAME_SAMPLES
    assert header.sample_rate == 24_000
    assert header.channels == 1


def test_parse_header_handles_padding_and_mpeg1() -> None:
    padded = parse_header(bytes([0xFF, 0xF3, 0x66, 0xC0]))
    mpeg1 = parse_header(MPEG1_STEREO_128K)

    assert padded is not None and padded.length == FRAME_BYTES + 1
    assert mpeg1 is not None
    assert (mpeg1.length, mpeg1.samples, mpeg1.sample_rate, mpeg1.channels) == (
        417,
        1152,
        44_100,
        2,
    )


@pytest.mark.parametrize(
    "header",
    [
        b"",
        b"\xff\xf3\x64",  # too short
        b"\x00\xf3\x64\xc0",  # no sync
        b"\xff\xeb\x64\xc0",  # reserved version
        b"\xff\xf5\x64\xc0",  # layer II
        b"\xff\xf3\x04\xc0",  # free-format bitrate
        b"\xff\xf3\xf4\xc0",  # invalid bitrate
        b"\xff\xf3\x6c\xc0",  # reserved sample rate
    ],
)
def test_parse_header_rejects_things_that_are_not_layer_iii_headers(header: bytes) -> None:
    assert parse_header(header) is None


def test_parse_header_at_an_offset() -> None:
    assert parse_header(b"junk" + EDGE_HEADER, 4) is not None
    assert parse_header(b"junk" + EDGE_HEADER, 3) is None


def test_frames_are_released_only_once_their_successor_is_seen() -> None:
    splitter = Mp3FrameSplitter()
    stream = frames(3)
    released = []
    for offset in range(0, len(stream), 100):  # chunks never align with frames
        released.extend(splitter.push(stream[offset : offset + 100]))

    assert [f.data for f in released] == [frame(0), frame(1)]
    assert [f.data for f in splitter.flush()] == [frame(2)]
    assert splitter.skipped_bytes == 0


def test_junk_before_the_first_frame_is_skipped_and_counted() -> None:
    splitter = Mp3FrameSplitter()

    released = splitter.push(b"ID3\x04\x00" + frames(2))
    released.extend(splitter.flush())

    assert [f.data for f in released] == [frame(0), frame(1)]
    assert splitter.skipped_bytes == 5


def test_a_false_sync_word_is_not_mistaken_for_a_frame() -> None:
    splitter = Mp3FrameSplitter()
    false_sync = EDGE_HEADER + bytes(50)  # looks like a header, no frame follows it

    released = splitter.push(false_sync + frames(2))
    released.extend(splitter.flush())

    assert [f.data for f in released] == [frame(0), frame(1)]
    assert splitter.skipped_bytes == len(false_sync)


def test_padded_frames_are_measured_correctly() -> None:
    splitter = Mp3FrameSplitter()

    released = splitter.push(frame(0, padded=True) + frame(1))
    released.extend(splitter.flush())

    assert [len(f.data) for f in released] == [FRAME_BYTES + 1, FRAME_BYTES]


def test_flush_discards_an_incomplete_trailing_frame() -> None:
    splitter = Mp3FrameSplitter()
    released = splitter.push(frame(0) + frame(1)[:50])  # frame 1's header confirms frame 0

    assert [f.data for f in released] == [frame(0)]
    assert splitter.flush() == []
    assert splitter.skipped_bytes == 50


def test_batcher_emits_once_enough_new_frames_arrived_and_primes_with_overlap() -> None:
    batcher = Mp3Batcher(batch_frames=2, overlap_frames=1)

    assert batcher.push(frame(0)) is None
    assert batcher.push(frame(1)) is None  # frame 1 is not confirmed yet
    first = batcher.push(frame(2) + frame(3))
    assert first is not None
    assert first.data == frame(0) + frame(1) + frame(2)
    assert first.keep_samples is None  # nothing to prime with: keep everything

    final = batcher.finish()
    assert final is not None
    assert final.data == frame(2) + frame(3)  # frame 2 is the overlap
    assert final.keep_samples == FRAME_SAMPLES  # only frame 3 is new
    assert batcher.frames_seen == 4
    assert batcher.finish() is None


def test_batcher_without_overlap_never_repeats_frames() -> None:
    batcher = Mp3Batcher(batch_frames=1, overlap_frames=0)

    batches = [batcher.push(frame(0) + frame(1)), batcher.finish()]

    assert [b.data for b in batches if b is not None] == [frame(0), frame(1)]
    assert all(b.keep_samples is None for b in batches if b is not None)


def test_batcher_rejects_nonsense_sizes() -> None:
    with pytest.raises(ValueError, match="batch_frames"):
        Mp3Batcher(batch_frames=0)


def test_the_real_edge_tts_clip_splits_into_whole_frames() -> None:
    data = FIXTURE.read_bytes()
    splitter = Mp3FrameSplitter()
    released = []
    for offset in range(0, len(data), 720):  # edge-tts' own chunk size
        released.extend(splitter.push(data[offset : offset + 720]))
    released.extend(splitter.flush())

    assert len(released) == len(data) // FRAME_BYTES
    assert all(f.header.sample_rate == 24_000 and f.header.channels == 1 for f in released)
    assert b"".join(f.data for f in released) == data
    assert splitter.skipped_bytes == 0
