import pytest

from jarvis.llm.base import LLMError
from jarvis.llm.streaming import SentenceSplitter, ToolCallAccumulator


def feed(splitter: SentenceSplitter, *deltas: str) -> list[str]:
    sentences: list[str] = []
    for delta in deltas:
        sentences.extend(splitter.push(delta))
    return sentences


def test_sentences_are_released_as_soon_as_their_boundary_arrives() -> None:
    splitter = SentenceSplitter()

    assert feed(splitter, "The weather in ", "Sofia is sunny. ", "It is 24") == [
        "The weather in Sofia is sunny."
    ]
    assert splitter.pending == "It is 24"


def test_boundary_needs_whitespace_after_the_punctuation() -> None:
    splitter = SentenceSplitter()

    assert feed(splitter, "Pi is about 3.14159 and e is 2.718. ") == [
        "Pi is about 3.14159 and e is 2.718."
    ]


def test_punctuation_at_the_end_of_a_delta_waits_for_what_follows() -> None:
    splitter = SentenceSplitter()

    assert feed(splitter, "Visit jarvis.local") == []
    assert feed(splitter, "host now. Done") == ["Visit jarvis.localhost now."]


def test_newlines_split_even_without_punctuation() -> None:
    splitter = SentenceSplitter()

    assert feed(splitter, "Here is your list\n- milk and eggs\n- bread") == [
        "Here is your list",
        "- milk and eggs",
    ]


def test_short_fragments_merge_with_the_next_sentence() -> None:
    splitter = SentenceSplitter(min_chars=12)

    assert feed(splitter, "Yes. I can do that for you. ") == ["Yes. I can do that for you."]


def test_question_and_exclamation_marks_with_closing_quotes() -> None:
    splitter = SentenceSplitter()

    assert feed(splitter, 'She asked "are you sure?" Then she left! Next') == [
        'She asked "are you sure?"',
        "Then she left!",
    ]


def test_flush_returns_the_tail_and_resets() -> None:
    splitter = SentenceSplitter()
    splitter.push("Almost done")

    assert splitter.flush() == "Almost done"
    assert splitter.flush() is None
    assert splitter.pending == ""


def test_whitespace_only_input_yields_nothing() -> None:
    splitter = SentenceSplitter()

    assert feed(splitter, "   \n\n  ") == []
    assert splitter.flush() is None


def test_tool_call_accumulator_assembles_fragments_into_a_call() -> None:
    calls = ToolCallAccumulator()
    calls.start(1, "toolu_1", "open_app")
    calls.append(1, '{"name": "co')
    calls.append(1, 'de", "maximised": true}')

    call = calls.finish(1)

    assert call.id == "toolu_1"
    assert call.name == "open_app"
    assert call.arguments == {"name": "code", "maximised": True}
    assert not calls.is_tracking(1)


def test_tool_call_without_arguments_has_an_empty_dict() -> None:
    calls = ToolCallAccumulator()
    calls.start(0, "toolu_2", "lock_screen")

    assert calls.finish(0).arguments == {}


def test_fragments_for_untracked_blocks_are_ignored() -> None:
    calls = ToolCallAccumulator()

    calls.append(7, "{}")  # a text block index, never started

    assert not calls.is_tracking(7)


def test_invalid_json_arguments_raise_llm_error() -> None:
    calls = ToolCallAccumulator()
    calls.start(0, "toolu_3", "open_app")
    calls.append(0, '{"name": ')

    with pytest.raises(LLMError, match="not valid JSON"):
        calls.finish(0)


def test_non_object_arguments_raise_llm_error() -> None:
    calls = ToolCallAccumulator()
    calls.start(0, "toolu_4", "open_app")
    calls.append(0, "[1, 2]")

    with pytest.raises(LLMError, match="non-object"):
        calls.finish(0)


def test_finishing_an_unknown_index_is_a_programming_error() -> None:
    with pytest.raises(KeyError):
        ToolCallAccumulator().finish(3)
