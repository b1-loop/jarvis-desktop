import dataclasses
from collections.abc import Callable

import pytest

from jarvis.agent.registry import ToolRegistry
from jarvis.config import Settings
from jarvis.tools._screen import MAX_SPOKEN_CHARS, first_sentence
from jarvis.tools.ask_about_screen import AskAboutScreenTool
from jarvis.vision.screen import ScreenTarget
from tests.agent.fakes import make_context
from tests.vision.fakes import FakeScreenReader, make_answer


@pytest.fixture
def settings(make_settings: Callable[..., Settings]) -> Settings:
    return make_settings()


def test_is_discovered_and_needs_no_confirmation() -> None:
    tool = ToolRegistry.discover().get("ask_about_screen")

    assert isinstance(tool, AskAboutScreenTool)
    assert not tool.requires_confirmation


async def test_passes_the_question_through(settings: Settings) -> None:
    reader = FakeScreenReader(answer=make_answer("The total is 1,240 euros."))
    context = dataclasses.replace(make_context(settings), vision=reader)

    result = await AskAboutScreenTool().call(
        {"question": "What is the total?", "target": "all_monitors"}, context
    )

    assert reader.asked == [("What is the total?", ScreenTarget.ALL_MONITORS)]
    assert result.ok
    assert result.spoken.startswith("The total is 1,240 euros.")
    assert result.data is not None
    assert result.data["answer"] == "The total is 1,240 euros."


async def test_an_empty_question_is_rejected(settings: Settings) -> None:
    reader = FakeScreenReader(answer=make_answer())
    context = dataclasses.replace(make_context(settings), vision=reader)

    result = await AskAboutScreenTool().call({"question": ""}, context)

    assert not result.ok
    assert reader.asked == []


def test_schema_documents_the_targets() -> None:
    schema = AskAboutScreenTool.to_llm_schema()["input_schema"]

    assert schema["required"] == ["question"]
    assert "active_window" in schema["properties"]["target"]["description"]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("One. Two.", "One."),
        ("Is it red? Yes.", "Is it red?"),
        ("No break at all", "No break at all"),
        ("  padded.  ", "padded."),
    ],
)
def test_first_sentence(text: str, expected: str) -> None:
    assert first_sentence(text) == expected


def test_a_very_long_sentence_is_cut_on_a_word_boundary() -> None:
    spoken = first_sentence("word " * 100)

    assert len(spoken) <= MAX_SPOKEN_CHARS + 1
    assert spoken.endswith("word…")
