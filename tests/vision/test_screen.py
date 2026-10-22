from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

import jarvis.vision.screen as screen_module
from jarvis.llm.base import LLMError, StopReason, SystemMessage, UserMessage
from jarvis.os_integration.base import WindowInfo
from jarvis.os_integration.errors import PlatformError, UnsupportedFeatureError
from jarvis.vision.errors import VisionError, VisionUnavailableError
from jarvis.vision.screen import (
    SCREEN_SYSTEM_PROMPT,
    NoRedaction,
    ScreenReader,
    ScreenTarget,
    VisionBackend,
    VisionOptions,
)
from tests.fakes import FakeClock
from tests.vision.fakes import (
    HAS_PILLOW,
    FakeVisionLLM,
    ScreenPlatform,
    image_size,
    make_png,
    requires_pillow,
)

EDITOR = WindowInfo(
    handle=7, title="main.py - Code", process_name="Code.exe", executable_path="", pid=42
)


def make_reader(
    platform: ScreenPlatform | None,
    llm: FakeVisionLLM | None,
    *,
    options: VisionOptions | None = None,
    clock: FakeClock | None = None,
    **backend: object,
) -> ScreenReader:
    vision = VisionBackend(llm, provider="Anthropic", sends_to_cloud=True, **backend)  # type: ignore[arg-type]
    return ScreenReader(platform, vision, options=options, clock=clock or FakeClock())


@dataclass
class RecordingLog:
    """Stands in for the module's structlog logger, which caches its config."""

    entries: list[dict[str, Any]] = field(default_factory=list)

    def info(self, event: str, **fields: Any) -> None:
        self.entries.append({"event": event, **fields})

    warning = info


@pytest.fixture
def logs(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    recorder = RecordingLog()
    monkeypatch.setattr(screen_module, "log", recorder)
    return recorder.entries


@pytest.fixture
def platform() -> ScreenPlatform:
    capture = make_png(3840, 2160) if HAS_PILLOW else b""
    return ScreenPlatform(capture=capture, windows=[EDITOR], active=EDITOR)


@requires_pillow
async def test_sends_one_user_message_with_exactly_one_image(platform: ScreenPlatform) -> None:
    llm = FakeVisionLLM()

    answer = await make_reader(platform, llm).ask("What is this?")

    (request,) = llm.requests
    system, user = request
    assert system == SystemMessage(SCREEN_SYSTEM_PROMPT)
    assert isinstance(user, UserMessage)
    assert user.text == "What is this?"
    (image,) = user.images
    assert image.media_type == "image/png"
    assert image.data.startswith(b"\x89PNG")
    assert answer.text == "The screen shows a code editor."
    assert answer.model == "fake-vision"


@requires_pillow
async def test_the_image_sent_is_downscaled_to_the_configured_long_edge(
    platform: ScreenPlatform,
) -> None:
    llm = FakeVisionLLM()
    options = VisionOptions(max_long_edge=1280)

    answer = await make_reader(platform, llm, options=options).ask("Summarise")

    sent = llm.requests[0][1]
    assert isinstance(sent, UserMessage)
    assert image_size(sent.images[0].data) == (1280, 720)
    assert (answer.width, answer.height) == (1280, 720)
    assert answer.image_bytes <= options.max_image_bytes


@requires_pillow
async def test_a_tight_byte_budget_sends_jpeg_with_the_matching_media_type() -> None:
    platform = ScreenPlatform(capture=make_png(800, 600, noisy=True))
    llm = FakeVisionLLM()

    answer = await make_reader(platform, llm, options=VisionOptions(max_image_bytes=1_000_000)).ask(
        "What is this?"
    )

    sent = llm.requests[0][1]
    assert isinstance(sent, UserMessage)
    assert sent.images[0].media_type == "image/jpeg"
    assert sent.images[0].data.startswith(b"\xff\xd8")  # JPEG magic
    assert answer.media_type == "image/jpeg"


@requires_pillow
async def test_active_window_captures_only_that_window(platform: ScreenPlatform) -> None:
    platform.window_capture = make_png(1000, 700)
    llm = FakeVisionLLM()

    answer = await make_reader(platform, llm).ask("Read it", ScreenTarget.ACTIVE_WINDOW)

    assert ("screenshot_window", (7,)) in platform.calls
    assert not any(name == "screenshot" for name, _ in platform.calls)
    assert (answer.width, answer.height) == (1000, 700)
    assert answer.target is ScreenTarget.ACTIVE_WINDOW


@requires_pillow
async def test_all_monitors_asks_the_platform_for_the_whole_desktop(
    platform: ScreenPlatform,
) -> None:
    await make_reader(platform, FakeVisionLLM()).ask("Summarise", ScreenTarget.ALL_MONITORS)

    assert ("screenshot", (True,)) in platform.calls


@requires_pillow
async def test_no_focused_window_is_a_clear_error(platform: ScreenPlatform) -> None:
    platform.active = None

    with pytest.raises(VisionError, match="no window has focus"):
        await make_reader(platform, FakeVisionLLM()).ask("Read it", ScreenTarget.ACTIVE_WINDOW)


@requires_pillow
async def test_platform_failure_becomes_a_vision_error(platform: ScreenPlatform) -> None:
    platform.capture_error = PlatformError("BitBlt failed")
    llm = FakeVisionLLM()

    with pytest.raises(VisionError, match="could not capture the screen: BitBlt failed"):
        await make_reader(platform, llm).ask("What is this?")
    assert llm.requests == []


@requires_pillow
async def test_platform_without_screenshots_makes_vision_unavailable(
    platform: ScreenPlatform,
) -> None:
    platform.capture_error = UnsupportedFeatureError("no desktop session")

    with pytest.raises(VisionUnavailableError, match="cannot take screenshots"):
        await make_reader(platform, FakeVisionLLM()).ask("What is this?")


@requires_pillow
async def test_llm_failure_becomes_a_vision_error(platform: ScreenPlatform) -> None:
    llm = FakeVisionLLM(error=LLMError("Ollama returned HTTP 500"))

    with pytest.raises(VisionError, match="vision model failed: Ollama returned HTTP 500"):
        await make_reader(platform, llm).ask("What is this?")


@requires_pillow
async def test_refusal_is_reported_not_returned_as_an_answer(platform: ScreenPlatform) -> None:
    llm = FakeVisionLLM(answer=("I can't help with that.",), stop_reason=StopReason.REFUSAL)

    with pytest.raises(VisionError, match="declined"):
        await make_reader(platform, llm).ask("What is this?")


@requires_pillow
async def test_an_empty_answer_is_an_error(platform: ScreenPlatform) -> None:
    with pytest.raises(VisionError, match="empty answer"):
        await make_reader(platform, FakeVisionLLM(answer=("  ",))).ask("What is this?")


async def test_disabled_vision_never_captures(platform: ScreenPlatform) -> None:
    llm = FakeVisionLLM()
    reader = make_reader(platform, llm, options=VisionOptions(enabled=False))

    with pytest.raises(VisionUnavailableError, match="turned off") as raised:
        await reader.ask("What is this?")

    assert "JARVIS_VISION_ENABLED" in raised.value.reason
    assert platform.calls == []
    assert llm.requests == []


async def test_provider_without_vision_reports_its_reason(platform: ScreenPlatform) -> None:
    reader = make_reader(platform, None, unavailable_reason="the local model cannot see images")

    with pytest.raises(VisionUnavailableError, match="cannot see images"):
        await reader.ask("What is this?")
    assert platform.calls == []


async def test_text_only_mode_without_a_platform_is_unavailable() -> None:
    with pytest.raises(VisionUnavailableError, match="no screen-capture backend"):
        await make_reader(None, FakeVisionLLM()).ask("What is this?")


@requires_pillow
async def test_a_blank_question_is_rejected_before_capturing(platform: ScreenPlatform) -> None:
    with pytest.raises(VisionError, match="empty"):
        await make_reader(platform, FakeVisionLLM()).ask("   ")
    assert platform.calls == []


@requires_pillow
async def test_redactor_output_is_what_gets_uploaded(platform: ScreenPlatform) -> None:
    redacted = make_png(640, 360)
    seen: list[tuple[bytes, ScreenTarget]] = []

    class BlackOut:
        async def redact(self, capture: bytes, target: ScreenTarget) -> bytes:
            seen.append((capture, target))
            return redacted

    llm = FakeVisionLLM()
    vision = VisionBackend(llm, provider="Anthropic", sends_to_cloud=True)
    reader = ScreenReader(platform, vision, redactor=BlackOut(), clock=FakeClock())

    answer = await reader.ask("What is this?")

    assert seen == [(platform.capture, ScreenTarget.PRIMARY)]
    assert (answer.width, answer.height) == (640, 360)


async def test_default_redaction_is_a_no_op() -> None:
    assert await NoRedaction().redact(b"png", ScreenTarget.PRIMARY) == b"png"


@requires_pillow
async def test_original_capture_is_kept_for_the_user(
    platform: ScreenPlatform, tmp_path: Path
) -> None:
    options = VisionOptions(screenshot_dir=tmp_path / "shots")

    answer = await make_reader(platform, FakeVisionLLM(), options=options).ask("Summarise")

    assert answer.original_path is not None
    assert answer.original_path.parent == tmp_path / "shots"
    assert answer.original_path.name == "screen-20261021-154207-000000.png"
    assert answer.original_path.read_bytes() == platform.capture


@requires_pillow
async def test_a_failed_save_still_returns_the_answer(
    platform: ScreenPlatform, tmp_path: Path, logs: list[dict[str, Any]]
) -> None:
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("x")

    answer = await make_reader(
        platform, FakeVisionLLM(), options=VisionOptions(screenshot_dir=blocker)
    ).ask("Summarise")

    assert answer.original_path is None
    assert any(entry["event"] == "vision.save_failed" for entry in logs)


@requires_pillow
async def test_latency_and_provider_are_reported(platform: ScreenPlatform) -> None:
    clock = FakeClock()
    llm = FakeVisionLLM(clock=clock, seconds=1.5)

    answer = await make_reader(platform, llm, clock=clock).ask("Summarise")

    assert answer.latency_ms == pytest.approx(1500.0)
    assert answer.provider == "Anthropic"
    assert answer.sent_to_cloud


@requires_pillow
async def test_logs_carry_sizes_but_never_image_question_or_answer(
    platform: ScreenPlatform, logs: list[dict[str, Any]]
) -> None:
    await make_reader(platform, FakeVisionLLM()).ask("my secret question")

    (entry,) = (entry for entry in logs if entry["event"] == "vision.answered")
    assert entry["width"] == 1920
    rendered = repr(logs)
    assert "my secret question" not in rendered
    assert "code editor" not in rendered
    assert "iVBOR" not in rendered  # base64 PNG header


def test_reader_exposes_provider_facts() -> None:
    reader = make_reader(None, None)

    assert reader.provider == "Anthropic"
    assert reader.sends_to_cloud
