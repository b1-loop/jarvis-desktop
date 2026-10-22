import base64
import importlib.util

import pytest

from jarvis.vision.errors import VisionError, VisionUnavailableError
from jarvis.vision.imaging import (
    MIN_LONG_EDGE,
    base64_length,
    candidate_long_edges,
    ensure_pillow,
    fit_long_edge,
    prepare_image,
)
from tests.vision.fakes import image_size, make_png, requires_pillow


@pytest.mark.parametrize(
    ("size", "max_edge", "expected"),
    [
        ((3840, 2160), 1920, (1920, 1080)),  # 4K landscape halves exactly
        ((2560, 1600), 1920, (1920, 1200)),  # 16:10 keeps its ratio
        ((1080, 1920), 1280, (720, 1280)),  # portrait scales on the height
        ((1920, 1080), 1920, (1920, 1080)),  # already fits
        ((800, 600), 1920, (800, 600)),  # never upscales
        ((10, 5000), 100, (1, 100)),  # a sliver keeps at least one pixel
    ],
)
def test_fit_long_edge_keeps_aspect_ratio(
    size: tuple[int, int], max_edge: int, expected: tuple[int, int]
) -> None:
    assert fit_long_edge(*size, max_edge) == expected


@pytest.mark.parametrize("args", [(0, 10, 10), (10, -1, 10), (10, 10, 0)])
def test_fit_long_edge_rejects_non_positive_sizes(args: tuple[int, int, int]) -> None:
    with pytest.raises(ValueError, match="positive"):
        fit_long_edge(*args)


@pytest.mark.parametrize("count", [0, 1, 2, 3, 4, 5, 6, 1000, 1001])
def test_base64_length_matches_the_real_encoding(count: int) -> None:
    assert base64_length(count) == len(base64.standard_b64encode(b"x" * count))


def test_candidate_edges_shrink_to_the_floor() -> None:
    assert list(candidate_long_edges(3840, 1920)) == [1920, 1440, 1080, 810, 607]


def test_candidate_edges_try_a_tiny_image_once_at_its_own_size() -> None:
    assert list(candidate_long_edges(300, 1920)) == [300]
    assert MIN_LONG_EDGE > 300


@requires_pillow
def test_flat_screenshot_is_downscaled_and_stays_png() -> None:
    prepared = prepare_image(make_png(3840, 2160), max_long_edge=1920, max_encoded_bytes=5_000_000)

    assert prepared.media_type == "image/png"
    assert (prepared.width, prepared.height) == (1920, 1080)
    assert (prepared.source_width, prepared.source_height) == (3840, 2160)
    assert image_size(prepared.data) == (1920, 1080)
    assert prepared.encoded_size == base64_length(len(prepared.data))


@requires_pillow
def test_over_budget_png_is_re_encoded_as_jpeg_at_the_same_size() -> None:
    capture = make_png(800, 600, noisy=True)  # ~1.4 MB of incompressible noise
    budget = 1_000_000

    prepared = prepare_image(capture, max_long_edge=1920, max_encoded_bytes=budget)

    assert prepared.media_type == "image/jpeg"
    assert (prepared.width, prepared.height) == (800, 600)
    assert prepared.encoded_size <= budget
    assert image_size(prepared.data) == (800, 600)


@requires_pillow
def test_when_no_encoding_fits_the_image_shrinks_until_one_does() -> None:
    capture = make_png(1600, 1200, noisy=True)
    budget = 400_000  # too small for JPEG q70 at full size

    prepared = prepare_image(capture, max_long_edge=1920, max_encoded_bytes=budget)

    assert prepared.width < 1600
    assert prepared.encoded_size <= budget
    assert prepared.width / prepared.height == pytest.approx(1600 / 1200, rel=0.01)


@requires_pillow
def test_rgba_captures_can_fall_back_to_jpeg() -> None:
    capture = make_png(600, 400, noisy=True, mode="RGBA")

    prepared = prepare_image(capture, max_long_edge=1920, max_encoded_bytes=500_000)

    assert prepared.media_type == "image/jpeg"


@requires_pillow
def test_an_impossible_budget_is_a_clear_error() -> None:
    with pytest.raises(VisionError, match="JARVIS_VISION_MAX_IMAGE_BYTES"):
        prepare_image(make_png(1024, 768, noisy=True), max_long_edge=1920, max_encoded_bytes=100)


@requires_pillow
def test_bytes_that_are_not_an_image_are_a_clear_error() -> None:
    with pytest.raises(VisionError, match="not a readable image"):
        prepare_image(b"not a png", max_long_edge=1920, max_encoded_bytes=5_000_000)


def test_missing_pillow_names_the_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(importlib.util, "find_spec", lambda _name: None)

    with pytest.raises(VisionUnavailableError, match="desktop") as raised:
        ensure_pillow()
    assert "Pillow" in raised.value.reason
