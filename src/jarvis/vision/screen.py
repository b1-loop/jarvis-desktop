"""``ScreenReader``: screenshot, prepare, ask a vision model, return the answer.

This is the service behind "what does this error mean?" and "summarise what
I'm looking at". It captures through the injected
:class:`~jarvis.os_integration.base.Platform`, passes the capture through a
redaction hook, downscales and re-encodes it (:mod:`jarvis.vision.imaging`),
and sends exactly one user message with the image and the question to a
vision-capable :class:`~jarvis.llm.base.LLMClient`.

Privacy: a capture happens only inside :meth:`ScreenReader.ask`, and ``ask``
runs only because the user asked a question that the agent routed to a
vision tool. There are no background or periodic captures, and
``vision_enabled=False`` turns the whole feature off.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Final, Protocol

from jarvis.llm.base import (
    Done,
    ImageContent,
    ImageMediaType,
    LLMClient,
    LLMError,
    StopReason,
    SystemMessage,
    TextDelta,
    UserMessage,
)
from jarvis.os_integration.errors import PlatformError, UnsupportedFeatureError
from jarvis.utils.logging import get_logger
from jarvis.utils.timing import Clock, SystemClock
from jarvis.vision.errors import VisionError, VisionUnavailableError
from jarvis.vision.imaging import PreparedImage, ensure_pillow, prepare_image

if TYPE_CHECKING:
    from jarvis.config import Settings
    from jarvis.os_integration.base import Platform

log = get_logger(__name__)

SCREEN_SYSTEM_PROMPT: Final = (
    "You are the eyes of Jarvis, a voice assistant. You receive one screenshot of the "
    "user's computer and a request about it. Answer only from what is visible; say so "
    "when something is cut off or unreadable instead of guessing. Your answer is read "
    "aloud: use short plain sentences, no markdown, no tables, no lists of more than "
    "three items. Never read out passwords, API keys, tokens or full card numbers even "
    "if they are visible; refer to them generically."
)
"""Frames every vision request; kept short because it is sent with each image."""

DEFAULT_SCREENSHOT_SUBDIR: Final = "screenshots"
_MS_PER_SECOND: Final = 1000.0


class ScreenTarget(StrEnum):
    """What to capture.

    ``Platform`` exposes the primary monitor, the whole virtual desktop and a
    single window, so those are the targets; a per-monitor index would need
    a new ``Platform`` method first.
    """

    PRIMARY = "primary"
    ALL_MONITORS = "all_monitors"
    ACTIVE_WINDOW = "active_window"


@dataclass(frozen=True, slots=True)
class ScreenAnswer:
    """The model's answer plus what was sent, for transparency and the dashboard.

    Attributes:
        text: The model's answer, stripped.
        target: What was captured.
        width: Pixel width of the image sent.
        height: Pixel height of the image sent.
        image_bytes: Base64 length of the image sent.
        media_type: Format of the image sent.
        latency_ms: Capture to final token, in milliseconds.
        model: Model that answered.
        provider: Human name of the provider, e.g. ``"Anthropic"``.
        sent_to_cloud: Whether the screenshot left this computer.
        original_path: Full-resolution capture saved for the user, if kept.
    """

    text: str
    target: ScreenTarget
    width: int
    height: int
    image_bytes: int
    media_type: ImageMediaType
    latency_ms: float
    model: str
    provider: str
    sent_to_cloud: bool
    original_path: Path | None = None


class ScreenReaderProtocol(Protocol):
    """What tools see of the vision service (``ToolContext.vision``)."""

    @property
    def provider(self) -> str:
        """Human name of the provider the screenshot goes to."""

    @property
    def sends_to_cloud(self) -> bool:
        """Whether a screenshot leaves this computer."""

    async def ask(self, question: str, target: ScreenTarget = ScreenTarget.PRIMARY) -> ScreenAnswer:
        """Capture ``target`` and answer ``question`` about it.

        Raises:
            VisionUnavailableError: If the feature cannot work as configured.
            VisionError: If this attempt failed.
        """


class Redactor(Protocol):
    """Hook that may blur or black out sensitive regions before upload.

    It receives the capture before downscaling (so regions are located at
    full resolution) and must return an encoded image. This is the single
    place where on-device detection of password fields, notifications or
    banking pages would plug in.
    """

    async def redact(self, capture: bytes, target: ScreenTarget) -> bytes:
        """Return ``capture`` with sensitive regions obscured."""


class NoRedaction:
    """The default :class:`Redactor`: returns the capture unchanged."""

    async def redact(self, capture: bytes, target: ScreenTarget) -> bytes:
        """Pass the capture through; nothing is detected yet."""
        del target
        return capture


@dataclass(frozen=True, slots=True)
class VisionBackend:
    """The model that answers screen questions and where it runs.

    Attributes:
        llm: A vision-capable client, or ``None`` when the configured
            provider has no vision model.
        provider: Human name of the provider, used in spoken messages.
        sends_to_cloud: Whether requests leave this computer.
        unavailable_reason: Speakable cause when ``llm`` is ``None``.
    """

    llm: LLMClient | None
    provider: str
    sends_to_cloud: bool
    unavailable_reason: str = "no vision-capable model is configured"


@dataclass(frozen=True, slots=True)
class VisionOptions:
    """Knobs for :class:`ScreenReader`, normally read from ``Settings``.

    Attributes:
        enabled: Master switch; ``False`` refuses every capture.
        max_long_edge: Longest side, in pixels, of the image sent.
        max_image_bytes: Upper bound on the base64 length of the image sent.
        screenshot_dir: Where full-resolution captures are kept for the
            user; ``None`` keeps nothing on disk.
    """

    enabled: bool = True
    max_long_edge: int = 1920
    max_image_bytes: int = 5_000_000
    screenshot_dir: Path | None = None

    @classmethod
    def from_settings(cls, settings: Settings) -> VisionOptions:
        """Options from the ``vision_*`` settings; captures go under ``data_dir``."""
        directory = settings.data_dir / DEFAULT_SCREENSHOT_SUBDIR
        return cls(
            enabled=settings.vision_enabled,
            max_long_edge=settings.vision_max_long_edge,
            max_image_bytes=settings.vision_max_image_bytes,
            screenshot_dir=directory if settings.vision_save_screenshots else None,
        )


class ScreenReader:
    """Answers questions about what is on the user's screen."""

    def __init__(
        self,
        platform: Platform | None,
        backend: VisionBackend,
        *,
        options: VisionOptions | None = None,
        redactor: Redactor | None = None,
        clock: Clock | None = None,
    ) -> None:
        self._platform = platform
        self._backend = backend
        self._options = options or VisionOptions()
        self._redactor = redactor or NoRedaction()
        self._clock = clock or SystemClock()

    @property
    def provider(self) -> str:
        """Human name of the provider the screenshot goes to."""
        return self._backend.provider

    @property
    def sends_to_cloud(self) -> bool:
        """Whether a screenshot leaves this computer."""
        return self._backend.sends_to_cloud

    async def ask(self, question: str, target: ScreenTarget = ScreenTarget.PRIMARY) -> ScreenAnswer:
        """Capture ``target``, send it with ``question``, and collect the answer.

        Raises:
            VisionUnavailableError: If vision is disabled, there is no capture
                backend, the provider has no vision model or Pillow is missing.
            VisionError: If the question is blank, the capture fails, the
                image cannot fit the budget, or the model fails.
        """
        platform, llm = self._require_ready()
        if not question.strip():
            raise VisionError("the question about the screen is empty")
        started = self._clock.monotonic()
        capture = await _capture(platform, target)
        redacted = await self._redactor.redact(capture, target)
        image = await asyncio.to_thread(
            prepare_image,
            redacted,
            max_long_edge=self._options.max_long_edge,
            max_encoded_bytes=self._options.max_image_bytes,
        )
        original_path = await self._keep_original(capture)
        text = await _collect_answer(llm, question.strip(), image)
        latency_ms = (self._clock.monotonic() - started) * _MS_PER_SECOND
        answer = self._answer(
            text, target, image, latency_ms=latency_ms, model=llm.model, original_path=original_path
        )
        _log_answer(answer)
        return answer

    def _require_ready(self) -> tuple[Platform, LLMClient]:
        if not self._options.enabled:
            raise VisionUnavailableError(
                "screen understanding is turned off; set JARVIS_VISION_ENABLED=true to allow it"
            )
        if self._platform is None:
            raise VisionUnavailableError("there is no screen-capture backend in this mode")
        if self._backend.llm is None:
            raise VisionUnavailableError(self._backend.unavailable_reason)
        ensure_pillow()
        return self._platform, self._backend.llm

    async def _keep_original(self, capture: bytes) -> Path | None:
        directory = self._options.screenshot_dir
        if directory is None:
            return None
        path = directory / f"screen-{self._clock.now():%Y%m%d-%H%M%S-%f}.png"
        try:
            await asyncio.to_thread(_write_file, path, capture)
        except OSError as error:
            # Losing the local copy must not cost the user their answer.
            log.warning("vision.save_failed", path=str(path), error=str(error))
            return None
        return path

    def _answer(
        self,
        text: str,
        target: ScreenTarget,
        image: PreparedImage,
        *,
        latency_ms: float,
        model: str,
        original_path: Path | None,
    ) -> ScreenAnswer:
        return ScreenAnswer(
            text=text,
            target=target,
            width=image.width,
            height=image.height,
            image_bytes=image.encoded_size,
            media_type=image.media_type,
            latency_ms=latency_ms,
            model=model,
            provider=self._backend.provider,
            sent_to_cloud=self._backend.sends_to_cloud,
            original_path=original_path,
        )


async def _capture(platform: Platform, target: ScreenTarget) -> bytes:
    """One PNG of ``target``, with platform failures turned into vision errors."""
    try:
        if target is ScreenTarget.ACTIVE_WINDOW:
            window = await platform.active_window()
            if window is None:
                raise VisionError("no window has focus, so there is no active window to capture")
            return await platform.screenshot_window(window.handle)
        return await platform.screenshot(all_monitors=target is ScreenTarget.ALL_MONITORS)
    except UnsupportedFeatureError as error:
        raise VisionUnavailableError(f"this computer cannot take screenshots ({error})") from error
    except PlatformError as error:
        raise VisionError(f"could not capture the screen: {error}") from error


async def _collect_answer(llm: LLMClient, question: str, image: PreparedImage) -> str:
    """Stream the model's answer to completion and return its text."""
    messages = (
        SystemMessage(SCREEN_SYSTEM_PROMPT),
        UserMessage(question, images=(ImageContent(image.data, image.media_type),)),
    )
    parts: list[str] = []
    stop_reason: StopReason | None = None
    try:
        async for event in llm.stream(messages):
            if isinstance(event, TextDelta):
                parts.append(event.text)
            elif isinstance(event, Done):
                stop_reason = event.stop_reason
    except LLMError as error:
        raise VisionError(f"the vision model failed: {error}") from error
    if stop_reason is StopReason.REFUSAL:
        raise VisionError("the vision model declined to answer about this screen")
    text = "".join(parts).strip()
    if not text:
        raise VisionError("the vision model returned an empty answer")
    return text


def _write_file(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def _log_answer(answer: ScreenAnswer) -> None:
    # Sizes and timings only: never the image, the question or the answer.
    log.info(
        "vision.answered",
        target=answer.target.value,
        width=answer.width,
        height=answer.height,
        image_bytes=answer.image_bytes,
        media_type=answer.media_type,
        latency_ms=round(answer.latency_ms, 1),
        model=answer.model,
        sent_to_cloud=answer.sent_to_cloud,
    )


# Design notes:
# - The active window is captured with Platform.screenshot_window rather
#   than by cropping a full screenshot: the rest of the desktop is then
#   never captured at all, which is the stronger privacy property, and
#   WindowInfo carries no rectangle to crop with anyway.
# - One user message, image first then text, with no tools offered: the
#   Claude docs recommend image-before-text, and a screen question is a
#   single-shot request, so the answer is collected rather than streamed to
#   TTS. The tool's spoken summary is what the user hears.
# - The capture saved for the user is the unredacted original: it never
#   leaves the machine, and it is what the user would want to look at. Only
#   the prepared, redacted copy is uploaded.
# - A failed local save is logged and the answer still returned; a failed
#   capture or model call raises, because then there is no answer to give.
# - Logs carry sizes and timings only. Screens show private data, so the
#   image, the question and the answer stay out of structured logs.
