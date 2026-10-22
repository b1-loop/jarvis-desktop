"""Fakes and image helpers for the vision tests.

Images are generated in memory with Pillow; tests that need them are marked
``requires_pillow`` so the suite still runs without the desktop extra.
"""

from __future__ import annotations

import importlib.util
import io
import random
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field

import pytest

from jarvis.llm.base import (
    AssistantMessage,
    Done,
    LLMClient,
    LLMEvent,
    Message,
    StopReason,
    TextDelta,
    ToolDefinition,
    Usage,
)
from jarvis.os_integration.stub import StubPlatform
from jarvis.vision.screen import ScreenAnswer, ScreenTarget
from tests.fakes import FakeClock

HAS_PILLOW = importlib.util.find_spec("PIL") is not None
requires_pillow = pytest.mark.skipif(
    not HAS_PILLOW, reason="Pillow (desktop extra) is not installed"
)


def make_png(width: int, height: int, *, noisy: bool = False, mode: str = "RGB") -> bytes:
    """A PNG of the given size: flat grey, or random noise that defeats compression."""
    from PIL import Image

    if noisy:
        channels = len(mode)
        pixels = random.Random(width * height).randbytes(width * height * channels)
        image = Image.frombytes(mode, (width, height), pixels)
    else:
        image = Image.new(mode, (width, height), "grey")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def image_size(data: bytes) -> tuple[int, int]:
    from PIL import Image

    with Image.open(io.BytesIO(data)) as image:
        return image.size


@dataclass
class ScreenPlatform(StubPlatform):
    """``StubPlatform`` whose screenshots are scripted images or errors."""

    capture: bytes = b""
    window_capture: bytes = b""
    capture_error: Exception | None = None

    async def screenshot(self, *, all_monitors: bool = False) -> bytes:
        self._record("screenshot", all_monitors)
        if self.capture_error is not None:
            raise self.capture_error
        return self.capture

    async def screenshot_window(self, handle: int) -> bytes:
        self._record("screenshot_window", handle)
        if self.capture_error is not None:
            raise self.capture_error
        return self.window_capture


@dataclass
class FakeVisionLLM(LLMClient):
    """Records every request and replays scripted text, a stop reason or an error."""

    answer: Sequence[str] = ("The screen shows ", "a code editor.")
    stop_reason: StopReason = StopReason.END_TURN
    error: Exception | None = None
    clock: FakeClock | None = None
    seconds: float = 0.0
    requests: list[list[Message]] = field(default_factory=list)

    @property
    def model(self) -> str:
        return "fake-vision"

    async def stream(
        self, messages: Sequence[Message], tools: Sequence[ToolDefinition] = ()
    ) -> AsyncIterator[LLMEvent]:
        self.requests.append(list(messages))
        if self.clock is not None:
            self.clock.advance(self.seconds)
        if self.error is not None:
            raise self.error
        for part in self.answer:
            yield TextDelta(part)
        message = AssistantMessage(text="".join(self.answer))
        yield Done(stop_reason=self.stop_reason, usage=Usage(), message=message)


@dataclass
class FakeScreenReader:
    """``ScreenReaderProtocol`` for tool tests: a scripted answer or error."""

    answer: ScreenAnswer | None = None
    error: Exception | None = None
    provider: str = "Anthropic"
    sends_to_cloud: bool = True
    asked: list[tuple[str, ScreenTarget]] = field(default_factory=list)

    async def ask(self, question: str, target: ScreenTarget = ScreenTarget.PRIMARY) -> ScreenAnswer:
        self.asked.append((question, target))
        if self.error is not None:
            raise self.error
        assert self.answer is not None
        return self.answer


def make_answer(
    text: str = "A terminal shows a failing test. It is red.", **overrides: object
) -> ScreenAnswer:
    fields: dict[str, object] = {
        "text": text,
        "target": ScreenTarget.PRIMARY,
        "width": 1920,
        "height": 1080,
        "image_bytes": 123_456,
        "media_type": "image/png",
        "latency_ms": 1234.56,
        "model": "claude-sonnet-5-5",
        "provider": "Anthropic",
        "sent_to_cloud": True,
        "original_path": None,
    }
    fields.update(overrides)
    return ScreenAnswer(**fields)  # type: ignore[arg-type]
