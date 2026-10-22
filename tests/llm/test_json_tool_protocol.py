import json
from collections.abc import Iterable

import pytest
from pydantic import BaseModel

from jarvis.llm.base import ToolDefinition
from jarvis.llm.json_tool_protocol import (
    MAX_TOOL_CALL_CHARS,
    JsonObjectScanner,
    ParsedToolCall,
    ToolCallParser,
    ToolCallRejectedError,
    json_start,
    parse_tool_call,
    render_tool_call,
    render_tool_instructions,
    schema_problems,
)


class WeatherInput(BaseModel):
    city: str
    days: int = 1


WEATHER = ToolDefinition.from_model("get_weather", "Weather for a city", WeatherInput)
VOLUME = ToolDefinition(
    "set_volume",
    "Set the volume",
    {
        "type": "object",
        "properties": {"level": {"type": "integer"}, "mode": {"enum": ["up", "down"]}},
        "required": ["level"],
        "additionalProperties": False,
    },
)
NO_ARGS = ToolDefinition("lock_screen", "Lock the screen", {"type": "object", "properties": {}})
TOOLS = {tool.name: tool for tool in (WEATHER, VOLUME, NO_ARGS)}


def run(pieces: Iterable[str]) -> tuple[str, ParsedToolCall | None]:
    parser = ToolCallParser(list(TOOLS.values()))
    emitted = []
    for piece in pieces:
        emitted.append(parser.push(piece))
        if parser.call is not None:
            break
    emitted.append(parser.finish())
    return "".join(emitted), parser.call


def chars(text: str) -> list[str]:
    """Worst case for a streaming parser: one character per token."""
    return list(text)


# --- prompt rendering ------------------------------------------------------


def test_instructions_list_every_tool_with_a_compact_schema_without_titles() -> None:
    text = render_tool_instructions([WEATHER, NO_ARGS])

    assert '{"tool": "<tool name>", "arguments": {<arguments>}}' in text
    assert "- get_weather: Weather for a city" in text
    assert "- lock_screen: Lock the screen" in text
    assert '"city":{"type":"string"}' in text
    assert "title" not in text.split("Available tools:")[1]


def test_rendered_call_round_trips_through_the_parser() -> None:
    rendered = render_tool_call("get_weather", {"city": "Paris"})

    assert json.loads(rendered) == {"tool": "get_weather", "arguments": {"city": "Paris"}}
    assert parse_tool_call(rendered, TOOLS) == ParsedToolCall("get_weather", {"city": "Paris"})


# --- streaming classification ----------------------------------------------


def test_plain_text_is_released_immediately() -> None:
    parser = ToolCallParser([WEATHER])

    assert parser.push("Hello") == "Hello"
    assert parser.push(" there.") == " there."
    assert parser.finish() == ""
    assert parser.call is None


def test_leading_whitespace_is_held_until_the_reply_declares_itself() -> None:
    parser = ToolCallParser([WEATHER])

    assert parser.push("  \n") == ""
    assert parser.push("Sure") == "  \nSure"


@pytest.mark.parametrize(
    "reply",
    [
        '{"tool": "get_weather", "arguments": {"city": "Paris"}}',
        ' \n{"tool":"get_weather","arguments":{"city":"Paris"}}',
        '```json\n{"tool": "get_weather", "arguments": {"city": "Paris"}}\n```',
        '```\n{"tool": "get_weather", "arguments": {"city": "Paris"}}\n```',
        '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris"}}\n</tool_call>',
        '{"tool": "get_weather", "city": "Paris"}',
    ],
)
def test_tool_call_split_across_single_character_tokens_is_detected(reply: str) -> None:
    text, call = run(chars(reply))

    assert text == ""
    assert call == ParsedToolCall("get_weather", {"city": "Paris"})


def test_nothing_after_the_closing_brace_is_consumed() -> None:
    parser = ToolCallParser([NO_ARGS])
    parser.push('{"tool": "lock_screen"}')

    assert parser.call == ParsedToolCall("lock_screen", {})
    assert parser.push(" and some chatter") == ""
    assert parser.finish() == ""


def test_braces_and_escaped_quotes_inside_strings_do_not_close_the_object() -> None:
    reply = '{"tool": "get_weather", "arguments": {"city": "Par}is \\"{x\\""}}'

    text, call = run(chars(reply))

    assert text == ""
    assert call == ParsedToolCall("get_weather", {"city": 'Par}is "{x"'})


@pytest.mark.parametrize(
    "reply",
    [
        '{"tool": "get_weather", "arguments": {"city": "Paris",}}',  # trailing comma
        '{"tool": "delete_everything", "arguments": {}}',  # not offered
        '{"tool": "get_weather", "arguments": {"days": 2}}',  # missing required
        '{"tool": "get_weather", "arguments": {"city": 7}}',  # wrong type
        '{"tool": "get_weather", "arguments": "Paris"}',  # arguments not an object
        '{"weather": "sunny", "temperature": 21}',  # JSON data, not a call
    ],
)
def test_invalid_calls_fall_back_to_the_original_text(reply: str) -> None:
    text, call = run(chars(reply))

    assert call is None
    assert text == reply


def test_text_after_a_rejected_candidate_keeps_streaming() -> None:
    parser = ToolCallParser([WEATHER])

    released = parser.push('{"oops": 1}')

    assert released == '{"oops": 1}'
    assert parser.push(" more") == " more"


def test_unfinished_object_is_released_when_the_stream_ends() -> None:
    text, call = run(chars('{"tool": "get_weather", "arguments": {"ci'))

    assert call is None
    assert text == '{"tool": "get_weather", "arguments": {"ci'


def test_oversized_candidate_is_released_as_text() -> None:
    parser = ToolCallParser([WEATHER])
    parser.push('{"tool": "get_weather", "arguments": {"city": "')

    released = parser.push("x" * MAX_TOOL_CALL_CHARS)

    assert released.startswith('{"tool"')
    assert parser.push("y") == "y"


def test_a_code_fence_around_something_else_is_text() -> None:
    text, call = run(chars("```python\nprint('hi')\n```"))

    assert call is None
    assert text == "```python\nprint('hi')\n```"


def test_partial_openers_are_held_but_not_forever() -> None:
    assert json_start("") is None
    assert json_start("   ") is None
    assert json_start("``") is None
    assert json_start("```js") is None
    assert json_start("<tool") is None
    assert json_start("```json\n") is None
    assert json_start("```json\n{") == 8
    assert json_start("Hello") == -1
    assert json_start("<b>bold</b>") == -1


# --- validation --------------------------------------------------------------


def test_schema_checks_required_types_enums_and_extra_keys() -> None:
    schema = VOLUME.input_schema

    assert schema_problems({"level": 3}, schema) == []
    assert schema_problems({}, schema) == ["missing 'level'"]
    assert schema_problems({"level": "3"}, schema) == ["'level' must be integer"]
    assert schema_problems({"level": True}, schema) == ["'level' must be integer"]
    assert schema_problems({"level": 3, "mode": "sideways"}, schema) == [
        "'mode' must be one of ['up', 'down']"
    ]
    assert schema_problems({"level": 3, "x": 1}, schema) == ["unexpected 'x'"]


def test_union_and_unknown_types_are_accepted_when_any_member_matches() -> None:
    schema = {
        "properties": {
            "value": {"type": ["string", "null"]},
            "blob": {"type": "custom"},
            "nested": {"anyOf": [{"type": "string"}]},
        }
    }

    assert schema_problems({"value": None, "blob": object(), "nested": 1}, schema) == []
    assert schema_problems({"value": 1.5}, schema) == ["'value' must be string or null"]


def test_null_arguments_mean_no_arguments() -> None:
    assert parse_tool_call('{"tool": "lock_screen", "arguments": null}', TOOLS) == (
        ParsedToolCall("lock_screen", {})
    )


@pytest.mark.parametrize(
    ("candidate", "reason"),
    [
        ("[1, 2]", "not a JSON object"),
        ("{nope}", "not valid JSON"),
        ('{"arguments": {}}', "unknown tool None"),
    ],
)
def test_rejections_explain_why(candidate: str, reason: str) -> None:
    with pytest.raises(ToolCallRejectedError, match=reason):
        parse_tool_call(candidate, TOOLS)


def test_scanner_reports_the_end_index_within_the_fed_chunk() -> None:
    scanner = JsonObjectScanner()

    assert scanner.feed('{"a": {') is None
    assert scanner.feed('"b": 1}} trailing') == 8
