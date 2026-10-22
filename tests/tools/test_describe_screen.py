import dataclasses
from collections.abc import Callable
from pathlib import Path

import pytest

from jarvis.agent.registry import ToolRegistry
from jarvis.agent.tool import ToolContext
from jarvis.config import Settings
from jarvis.tools.describe_screen import DESCRIBE_PROMPT, DescribeScreenTool, describe_prompt
from jarvis.vision.errors import VisionError, VisionUnavailableError
from jarvis.vision.screen import ScreenTarget
from tests.agent.fakes import make_context
from tests.vision.fakes import FakeScreenReader, make_answer


@pytest.fixture
def settings(make_settings: Callable[..., Settings]) -> Settings:
    return make_settings()


def with_vision(settings: Settings, reader: FakeScreenReader | None) -> ToolContext:
    return dataclasses.replace(make_context(settings), vision=reader)


def test_is_discovered_and_needs_no_confirmation() -> None:
    tool = ToolRegistry.discover().get("describe_screen")

    assert isinstance(tool, DescribeScreenTool)
    assert not tool.requires_confirmation


async def test_summary_mentions_the_cloud_upload(settings: Settings) -> None:
    reader = FakeScreenReader(answer=make_answer(original_path=Path("shots/screen.png")))

    result = await DescribeScreenTool().call({}, with_vision(settings, reader))

    assert result.ok
    assert reader.asked == [(DESCRIBE_PROMPT, ScreenTarget.PRIMARY)]
    assert result.spoken == (
        "A terminal shows a failing test. I sent a screenshot to Anthropic to answer that."
    )
    assert result.detail == "A terminal shows a failing test. It is red."
    assert result.data is not None
    assert result.data["sent_to_cloud"] is True
    assert result.data["screenshot_path"] == str(Path("shots/screen.png"))
    assert result.data["width"] == 1920


async def test_local_provider_needs_no_upload_notice(settings: Settings) -> None:
    answer = make_answer(provider="Ollama", sent_to_cloud=False)
    reader = FakeScreenReader(answer=answer, provider="Ollama", sends_to_cloud=False)

    result = await DescribeScreenTool().call({}, with_vision(settings, reader))

    assert result.spoken == "A terminal shows a failing test."
    assert result.data is not None
    assert result.data["screenshot_path"] is None


async def test_focus_and_target_are_passed_through(settings: Settings) -> None:
    reader = FakeScreenReader(answer=make_answer())

    await DescribeScreenTool().call(
        {"focus": "the chart", "target": "active_window"}, with_vision(settings, reader)
    )

    ((prompt, target),) = reader.asked
    assert prompt.endswith("Concentrate on: the chart.")
    assert target is ScreenTarget.ACTIVE_WINDOW


def test_blank_focus_keeps_the_plain_prompt() -> None:
    assert describe_prompt("  ") == DESCRIBE_PROMPT


async def test_missing_vision_service_is_a_clear_failure(settings: Settings) -> None:
    result = await DescribeScreenTool().call({}, with_vision(settings, None))

    assert not result.ok
    assert result.spoken == "I can't look at your screen in this setup."


async def test_unavailable_vision_speaks_the_reason(settings: Settings) -> None:
    reader = FakeScreenReader(error=VisionUnavailableError("screen understanding is turned off"))

    result = await DescribeScreenTool().call({}, with_vision(settings, reader))

    assert not result.ok
    assert result.spoken == "I can't look at your screen: screen understanding is turned off."


async def test_failed_attempt_is_reported_with_detail(settings: Settings) -> None:
    reader = FakeScreenReader(error=VisionError("could not capture the screen: BitBlt failed"))

    result = await DescribeScreenTool().call({}, with_vision(settings, reader))

    assert not result.ok
    assert result.spoken == "I couldn't make sense of your screen just now."
    assert "BitBlt failed" in result.detail


async def test_unknown_target_is_rejected(settings: Settings) -> None:
    reader = FakeScreenReader(answer=make_answer())

    result = await DescribeScreenTool().call({"target": "webcam"}, with_vision(settings, reader))

    assert not result.ok
    assert reader.asked == []
