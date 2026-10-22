import dataclasses
from collections.abc import Callable

import pytest

from jarvis.agent.registry import ToolRegistry
from jarvis.agent.tool import ToolCategory
from jarvis.config import Settings
from jarvis.tools.explain_screen_error import ERROR_PROMPT, ExplainScreenErrorTool, error_prompt
from jarvis.vision.errors import VisionUnavailableError
from jarvis.vision.screen import ScreenTarget
from tests.agent.fakes import make_context
from tests.vision.fakes import FakeScreenReader, make_answer


@pytest.fixture
def settings(make_settings: Callable[..., Settings]) -> Settings:
    return make_settings()


def test_is_a_read_only_developer_tool() -> None:
    tool = ToolRegistry.discover().get("explain_screen_error")

    assert isinstance(tool, ExplainScreenErrorTool)
    assert tool.category is ToolCategory.DEVELOPER
    assert not tool.requires_confirmation


def test_prompt_asks_for_the_error_line_cause_and_fix() -> None:
    assert "Quote the single most important error line exactly" in ERROR_PROMPT
    assert "most likely cause" in ERROR_PROMPT
    assert "fix" in ERROR_PROMPT
    assert "do not invent one" in ERROR_PROMPT


def test_user_details_are_appended() -> None:
    assert error_prompt(" after npm install ").endswith("The user adds: after npm install")
    assert error_prompt(None) == ERROR_PROMPT


async def test_explains_with_the_error_prompt(settings: Settings) -> None:
    answer = make_answer(
        "ModuleNotFoundError: no module named requests. Install it with uv add requests."
    )
    reader = FakeScreenReader(answer=answer)
    context = dataclasses.replace(make_context(settings), vision=reader)

    result = await ExplainScreenErrorTool().call(
        {"details": "after I pulled", "target": "active_window"}, context
    )

    assert result.ok
    ((prompt, target),) = reader.asked
    assert prompt.startswith(ERROR_PROMPT)
    assert prompt.endswith("The user adds: after I pulled")
    assert target is ScreenTarget.ACTIVE_WINDOW
    assert result.spoken.startswith("ModuleNotFoundError: no module named requests.")
    assert result.spoken.endswith("I sent a screenshot to Anthropic to answer that.")
    assert "uv add requests" in result.detail


async def test_degrades_when_the_model_cannot_see(settings: Settings) -> None:
    reader = FakeScreenReader(
        error=VisionUnavailableError("the local model cannot see images"),
        provider="Ollama",
        sends_to_cloud=False,
    )
    context = dataclasses.replace(make_context(settings), vision=reader)

    result = await ExplainScreenErrorTool().call({}, context)

    assert not result.ok
    assert "the local model cannot see images" in result.spoken
