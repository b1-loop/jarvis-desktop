"""Turn a raw screenshot into an image a vision model accepts cheaply.

A 4K desktop PNG is tens of megabytes of pixels the model never sees:
providers downscale anything above their native resolution and bill by
pixel area. Preparing the image locally therefore saves upload time and
tokens without losing information, and it keeps each request under the
provider's per-image byte limit. Everything here is synchronous CPU work;
:class:`~jarvis.vision.screen.ScreenReader` runs it in a worker thread.

Pillow comes with the ``desktop`` extra and is imported only inside the
functions that need it, so the rest of Jarvis imports this module freely.
"""

from __future__ import annotations

import importlib.util
import io
from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from jarvis.llm.base import ImageMediaType
from jarvis.vision.errors import VisionError, VisionUnavailableError

if TYPE_CHECKING:
    from PIL.Image import Image

JPEG_QUALITIES: Final = (90, 80, 70)
"""Tried in order after PNG; below 70 JPEG artefacts start eating small text."""

SHRINK_FACTOR: Final = 0.75
"""How much the long edge shrinks per round when no encoding fits the budget."""

MIN_LONG_EDGE: Final = 512
"""Below this, on-screen text is unreadable; failing beats a confident guess."""

_BASE64_GROUP_BYTES: Final = 3
_BASE64_GROUP_CHARS: Final = 4
_JPEG_COMPATIBLE_MODES: Final = frozenset({"RGB", "L"})


@dataclass(frozen=True, slots=True)
class PreparedImage:
    """An encoded image ready to attach to a user message.

    Attributes:
        data: Encoded bytes (PNG or JPEG).
        media_type: MIME type of ``data``.
        width: Pixel width after downscaling.
        height: Pixel height after downscaling.
        source_width: Pixel width of the capture.
        source_height: Pixel height of the capture.
    """

    data: bytes
    media_type: ImageMediaType
    width: int
    height: int
    source_width: int
    source_height: int

    @property
    def encoded_size(self) -> int:
        """Length of the base64 text that goes on the wire."""
        return base64_length(len(self.data))


def base64_length(byte_count: int) -> int:
    """Characters of padded standard base64 for ``byte_count`` raw bytes.

    Provider image limits count the base64 text, which is a third larger
    than the file, so the byte budget is checked against this number.
    """
    groups = -(-byte_count // _BASE64_GROUP_BYTES)  # ceiling division
    return groups * _BASE64_GROUP_CHARS


def fit_long_edge(width: int, height: int, max_long_edge: int) -> tuple[int, int]:
    """Scale ``(width, height)`` so the longer side is at most ``max_long_edge``.

    Keeps the aspect ratio and never upscales: enlarging adds tokens and no
    information.

    Raises:
        ValueError: If any argument is not positive.
    """
    if width <= 0 or height <= 0 or max_long_edge <= 0:
        raise ValueError(f"dimensions must be positive, got {width}x{height} -> {max_long_edge}")
    long_edge = max(width, height)
    if long_edge <= max_long_edge:
        return width, height
    scale = max_long_edge / long_edge
    return max(1, round(width * scale)), max(1, round(height * scale))


def ensure_pillow() -> None:
    """Fail with a speakable remedy when Pillow is not installed.

    Raises:
        VisionUnavailableError: If the ``PIL`` package cannot be found.
    """
    if importlib.util.find_spec("PIL") is None:
        raise VisionUnavailableError(
            "the image library Pillow is not installed; run 'uv sync --extra desktop'"
        )


def prepare_image(capture: bytes, *, max_long_edge: int, max_encoded_bytes: int) -> PreparedImage:
    """Downscale and re-encode ``capture`` until it fits the byte budget.

    PNG is tried first because it keeps text crisp; JPEG at falling quality
    follows; if nothing fits, the long edge shrinks and the round repeats.

    Args:
        capture: Encoded screenshot as the platform returned it.
        max_long_edge: Longest side, in pixels, the provider should receive.
        max_encoded_bytes: Upper bound on the base64 length of the result.

    Raises:
        VisionUnavailableError: If Pillow is missing.
        VisionError: If ``capture`` is not an image, or no size down to
            :data:`MIN_LONG_EDGE` fits the budget.
    """
    ensure_pillow()
    image = _decode(capture)
    source_width, source_height = image.size
    for long_edge in candidate_long_edges(max(image.size), max_long_edge):
        resized = _resize(image, fit_long_edge(source_width, source_height, long_edge))
        for media_type, data in _encodings(resized):
            if base64_length(len(data)) <= max_encoded_bytes:
                width, height = resized.size
                return PreparedImage(data, media_type, width, height, source_width, source_height)
    raise VisionError(
        f"the screenshot does not fit in {max_encoded_bytes} bytes even at "
        f"{MIN_LONG_EDGE} px; raise JARVIS_VISION_MAX_IMAGE_BYTES"
    )


def candidate_long_edges(source_long_edge: int, max_long_edge: int) -> Iterator[int]:
    """Long-edge sizes to try, largest first, ending at :data:`MIN_LONG_EDGE`.

    An image already smaller than the floor is tried once at its own size.
    """
    edge = min(source_long_edge, max_long_edge)
    yield edge
    edge = int(edge * SHRINK_FACTOR)
    while edge >= MIN_LONG_EDGE:
        yield edge
        edge = int(edge * SHRINK_FACTOR)


def _decode(capture: bytes) -> Image:
    from PIL import Image

    try:
        image = Image.open(io.BytesIO(capture))
        image.load()
    except OSError as error:  # UnidentifiedImageError derives from OSError
        raise VisionError(f"the screenshot is not a readable image: {error}") from error
    return image


def _resize(image: Image, size: tuple[int, int]) -> Image:
    from PIL import Image as PilImage

    if image.size == size:
        return image
    return image.resize(size, PilImage.Resampling.LANCZOS)


def _encodings(image: Image) -> Iterator[tuple[ImageMediaType, bytes]]:
    """Encodings in order of preference; lazily, because encoding costs CPU."""
    yield "image/png", _encode(image, "PNG")
    rgb = image if image.mode in _JPEG_COMPATIBLE_MODES else image.convert("RGB")
    for quality in JPEG_QUALITIES:
        yield "image/jpeg", _encode(rgb, "JPEG", quality=quality)


def _encode(image: Image, image_format: str, **options: int) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format=image_format, **options)
    return buffer.getvalue()


# Design notes:
# - The budget is on base64 length, not file size, because that is what the
#   provider limits (Claude: 5 MB on Bedrock and Vertex, 10 MB direct).
# - PNG before JPEG: screenshots are mostly flat colour and text, where PNG
#   is both smaller and lossless at typical sizes; JPEG only wins on photos
#   and video frames, which is exactly when the budget is at risk.
# - Shrinking is the last resort and has a floor, so an absurdly small budget
#   produces a clear configuration error instead of an unreadable thumbnail.
# - LANCZOS keeps thin text strokes legible when downscaling; its cost is
#   irrelevant next to a network round trip.
