import json
from collections.abc import Callable, Sequence
from typing import Any

import httpx
import pytest

from jarvis.llm.base import (
    AssistantMessage,
    Done,
    ImageContent,
    LLMError,
    LLMEvent,
    Message,
    StopReason,
    SystemMessage,
    TextDelta,
    ToolCall,
    ToolCallRequested,
    ToolDefinition,
    ToolResult,
    ToolResultMessage,
    UserMessage,
)
from jarvis.llm.ollama_client import OllamaClient, is_retryable, to_message_params, to_tool_param
from jarvis.utils.retry import RetryPolicy

TOOL = ToolDefinition("open_app", "Open an application", {"type": "object", "properties": {}})


def ndjson(*chunks: dict[str, Any]) -> bytes:
    return "".join(json.dumps(chunk) + "\n" for chunk in chunks).encode()


def text_chunk(text: str) -> dict[str, Any]:
    return {"model": "qwen2.5:7b", "message": {"role": "assistant", "content": text}, "done": False}


def done_chunk(reason: str = "stop", prompt: int = 26, output: int = 9) -> dict[str, Any]:
    return {
        "model": "qwen2.5:7b",
        "message": {"role": "assistant", "content": ""},
        "done": True,
        "done_reason": reason,
        "prompt_eval_count": prompt,
        "eval_count": output,
    }


class FakeServer:
    """Scripted ``httpx.MockTransport`` handler recording every request body."""

    def __init__(self, responses: list[httpx.Response | Exception]) -> None:
        self.responses = responses
        self.requests: list[dict[str, Any]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(json.loads(request.content))
        outcome = self.responses.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def make_client(server: FakeServer, **kwargs: Any) -> OllamaClient:
    async def no_sleep(_seconds: float) -> None:
        return None

    http = httpx.AsyncClient(
        base_url="http://ollama.test:11434", transport=httpx.MockTransport(server)
    )
    return OllamaClient(http, model="qwen2.5:7b", sleep=no_sleep, **kwargs)


async def collect(
    client: OllamaClient, messages: Sequence[Message], tools: Sequence[ToolDefinition] = ()
) -> list[LLMEvent]:
    return [event async for event in client.stream(messages, tools)]


async def test_text_turn_streams_deltas_then_done() -> None:
    server = FakeServer(
        [httpx.Response(200, content=ndjson(text_chunk("Hel"), text_chunk("lo."), done_chunk()))]
    )
    client = make_client(server)

    events = await collect(client, [SystemMessage("Be brief."), UserMessage("hi")])

    assert events[:2] == [TextDelta("Hel"), TextDelta("lo.")]
    done = events[2]
    assert isinstance(done, Done)
    assert done.stop_reason is StopReason.END_TURN
    assert done.usage.input_tokens == 26
    assert done.usage.output_tokens == 9
    assert done.message == AssistantMessage("Hello.")
    assert client.model == "qwen2.5:7b"
    await client.aclose()


async def test_request_body_targets_the_chat_endpoint_with_streaming_and_tools() -> None:
    server = FakeServer([httpx.Response(200, content=ndjson(done_chunk()))])

    await collect(make_client(server), [UserMessage("hi")], [TOOL])

    body = server.requests[0]
    assert body["model"] == "qwen2.5:7b"
    assert body["stream"] is True
    assert body["messages"] == [{"role": "user", "content": "hi"}]
    assert body["tools"] == [to_tool_param(TOOL)]


async def test_tools_key_is_absent_when_no_tools_are_offered() -> None:
    server = FakeServer([httpx.Response(200, content=ndjson(done_chunk()))])

    await collect(make_client(server), [UserMessage("hi")])

    assert "tools" not in server.requests[0]


async def test_tool_calls_get_ids_and_mark_the_turn_as_tool_use() -> None:
    call_chunk = {
        "message": {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"function": {"name": "open_app", "arguments": {"name": "code"}}}],
        },
        "done": False,
    }
    server = FakeServer([httpx.Response(200, content=ndjson(call_chunk, done_chunk()))])

    events = await collect(make_client(server), [UserMessage("open code")], [TOOL])

    requested = events[0]
    assert isinstance(requested, ToolCallRequested)
    assert requested.call.name == "open_app"
    assert requested.call.arguments == {"name": "code"}
    assert requested.call.id.startswith("call_")
    done = events[1]
    assert isinstance(done, Done)
    assert done.stop_reason is StopReason.TOOL_USE
    assert done.message.tool_calls == (requested.call,)


async def test_string_arguments_and_server_ids_are_accepted() -> None:
    call_chunk = {
        "message": {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "abc", "function": {"name": "open_app", "arguments": '{"name": "x"}'}}
            ],
        },
        "done": True,
        "done_reason": "stop",
    }
    server = FakeServer([httpx.Response(200, content=ndjson(call_chunk))])

    events = await collect(make_client(server), [UserMessage("hi")])

    assert events[0] == ToolCallRequested(ToolCall("abc", "open_app", {"name": "x"}))


async def test_length_stop_reason_maps_to_max_tokens() -> None:
    server = FakeServer(
        [httpx.Response(200, content=ndjson(text_chunk("x"), done_chunk("length")))]
    )

    done = (await collect(make_client(server), [UserMessage("hi")]))[-1]

    assert isinstance(done, Done)
    assert done.stop_reason is StopReason.MAX_TOKENS


async def test_missing_model_explains_how_to_pull_it() -> None:
    server = FakeServer([httpx.Response(404, json={"error": "model not found"})])

    with pytest.raises(LLMError, match=r"ollama pull qwen2\.5:7b"):
        await collect(make_client(server), [UserMessage("hi")])

    assert len(server.requests) == 1


async def test_server_errors_are_retried() -> None:
    server = FakeServer(
        [
            httpx.Response(500, text="boom"),
            httpx.Response(200, content=ndjson(text_chunk("ok"), done_chunk())),
        ]
    )
    client = make_client(server, policy=RetryPolicy(max_attempts=2, base_delay=0.0))

    events = await collect(client, [UserMessage("hi")])

    assert events[0] == TextDelta("ok")
    assert len(server.requests) == 2


async def test_unreachable_server_explains_how_to_start_it() -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    server = FakeServer([httpx.ConnectError("connection refused")])
    client = make_client(server, policy=RetryPolicy(max_attempts=1))

    with pytest.raises(LLMError, match="ollama serve"):
        await collect(client, [UserMessage("hi")])


async def test_error_chunk_in_the_stream_is_reported() -> None:
    server = FakeServer([httpx.Response(200, content=ndjson({"error": "out of memory"}))])

    with pytest.raises(LLMError, match="out of memory"):
        await collect(make_client(server), [UserMessage("hi")])


async def test_non_json_line_is_reported() -> None:
    server = FakeServer([httpx.Response(200, content=b"<html>proxy error</html>\n")])

    with pytest.raises(LLMError, match="not JSON"):
        await collect(make_client(server), [UserMessage("hi")])


async def test_stream_without_a_done_chunk_is_an_error() -> None:
    server = FakeServer([httpx.Response(200, content=ndjson(text_chunk("partial")))])

    with pytest.raises(LLMError, match="final chunk"):
        await collect(make_client(server), [UserMessage("hi")])


async def test_tool_call_without_a_name_is_rejected() -> None:
    bad = {
        "message": {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"function": {"arguments": {}}}],
        },
        "done": False,
    }
    server = FakeServer([httpx.Response(200, content=ndjson(bad))])

    with pytest.raises(LLMError, match="function name"):
        await collect(make_client(server), [UserMessage("hi")])


def test_is_retryable_classifies_httpx_errors() -> None:
    request = httpx.Request("POST", "http://ollama.test/api/chat")
    assert is_retryable(httpx.ConnectError("refused", request=request))
    assert is_retryable(httpx.ReadTimeout("slow", request=request))
    assert is_retryable(
        httpx.HTTPStatusError("500", request=request, response=httpx.Response(500, request=request))
    )
    assert not is_retryable(
        httpx.HTTPStatusError("404", request=request, response=httpx.Response(404, request=request))
    )
    assert not is_retryable(ValueError("nope"))


def test_history_is_converted_to_ollama_chat_messages() -> None:
    image = ImageContent(b"\x89PNG", "image/png")
    history: list[Message] = [
        SystemMessage("You are Jarvis."),
        UserMessage("What is this?", images=(image,)),
        AssistantMessage("Looking.", tool_calls=(ToolCall("c1", "screenshot", {"region": "all"}),)),
        ToolResultMessage(
            (ToolResult("c1", "screenshot", "a terminal"), ToolResult("c2", "clock", "noon"))
        ),
    ]

    params = [param for message in history for param in to_message_params(message)]

    assert params == [
        {"role": "system", "content": "You are Jarvis."},
        {"role": "user", "content": "What is this?", "images": [image.as_base64()]},
        {
            "role": "assistant",
            "content": "Looking.",
            "tool_calls": [{"function": {"name": "screenshot", "arguments": {"region": "all"}}}],
        },
        {"role": "tool", "content": "a terminal", "tool_name": "screenshot"},
        {"role": "tool", "content": "noon", "tool_name": "clock"},
    ]


def test_tool_param_uses_the_function_calling_shape() -> None:
    assert to_tool_param(TOOL) == {
        "type": "function",
        "function": {
            "name": "open_app",
            "description": "Open an application",
            "parameters": {"type": "object", "properties": {}},
        },
    }


async def test_from_settings_points_at_the_configured_server(
    make_settings: Callable[..., Any],
) -> None:
    client = OllamaClient.from_settings(
        make_settings(ollama_base_url="http://box:11434", ollama_model="llama3.2")
    )

    assert client.model == "llama3.2"
    await client.aclose()


# -- thinking models (qwen3) ---------------------------------------------------


def thinking_chunk(thought: str) -> dict[str, Any]:
    return {
        "model": "qwen3:8b",
        "message": {"role": "assistant", "content": "", "thinking": thought},
        "done": False,
    }


@pytest.mark.parametrize(("think", "sent"), [(False, False), (True, True)])
async def test_think_is_sent_as_configured(think: bool, sent: bool) -> None:
    server = FakeServer([httpx.Response(200, content=ndjson(text_chunk("Hej."), done_chunk()))])

    await collect(make_client(server, think=think), [UserMessage("hi")])

    assert server.requests[0]["think"] is sent


async def test_think_is_omitted_when_the_model_should_decide() -> None:
    server = FakeServer([httpx.Response(200, content=ndjson(done_chunk()))])

    await collect(make_client(server, think=None), [UserMessage("hi")])

    assert "think" not in server.requests[0]


async def test_thinking_is_never_part_of_the_answer() -> None:
    chunks = ndjson(
        thinking_chunk("The user wants RAM. "),
        thinking_chunk("I will guess 12 GB."),
        text_chunk("Du använder 7,9 GB."),
        done_chunk(),
    )
    server = FakeServer([httpx.Response(200, content=chunks)])

    events = await collect(make_client(server, think=True), [UserMessage("RAM?")])

    assert [e for e in events if isinstance(e, TextDelta)] == [TextDelta("Du använder 7,9 GB.")]
    done = events[-1]
    assert isinstance(done, Done)
    assert done.message.text == "Du använder 7,9 GB."


async def test_thinking_alongside_a_tool_call_keeps_only_the_call() -> None:
    call_chunk = {
        "model": "qwen3:8b",
        "message": {
            "role": "assistant",
            "content": "",
            "thinking": "Need the clock.",
            "tool_calls": [{"function": {"name": "open_app", "arguments": {}}}],
        },
        "done": False,
    }
    server = FakeServer([httpx.Response(200, content=ndjson(call_chunk, done_chunk()))])

    events = await collect(make_client(server, think=True), [UserMessage("hi")], [TOOL])

    done = events[-1]
    assert isinstance(done, Done)
    assert done.message.text == ""
    assert [call.name for call in done.message.tool_calls] == ["open_app"]
    assert not any(isinstance(event, TextDelta) for event in events)


async def test_think_on_a_model_without_thinking_names_the_setting() -> None:
    refusal = httpx.Response(400, json={"error": '"qwen2.5:7b" does not support thinking'})
    server = FakeServer([refusal])

    with pytest.raises(LLMError, match="JARVIS_OLLAMA_THINK=false"):
        await collect(make_client(server, think=True), [UserMessage("hi")])


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [({}, False), ({"ollama_think": "default"}, None), ({"ollama_think": True}, True)],
)
async def test_from_settings_passes_the_think_setting(
    make_settings: Callable[..., Any], overrides: dict[str, Any], expected: bool | None
) -> None:
    server = FakeServer([httpx.Response(200, content=ndjson(text_chunk("Hej."), done_chunk()))])
    settings = make_settings(ollama_base_url="http://ollama.test:11434", **overrides)
    client = OllamaClient.from_settings(settings)
    await client.aclose()
    client._http = httpx.AsyncClient(
        base_url=settings.ollama_base_url, transport=httpx.MockTransport(server)
    )

    await collect(client, [UserMessage("hi")])

    assert server.requests[0].get("think") is expected
    await client.aclose()


async def test_an_empty_reply_with_thinking_off_is_asked_for_once_more() -> None:
    empty = httpx.Response(200, content=ndjson(done_chunk()))
    answer = httpx.Response(200, content=ndjson(text_chunk("Det blir 391."), done_chunk()))
    server = FakeServer([empty, answer])

    events = await collect(make_client(server, think=False), [UserMessage("17 gånger 23?")])

    assert len(server.requests) == 2
    assert [e for e in events if isinstance(e, TextDelta)] == [TextDelta("Det blir 391.")]
    assert sum(isinstance(event, Done) for event in events) == 1


async def test_a_second_empty_reply_is_passed_on() -> None:
    server = FakeServer([httpx.Response(200, content=ndjson(done_chunk())) for _ in range(2)])

    events = await collect(make_client(server, think=False), [UserMessage("hi")])

    assert len(server.requests) == 2
    assert len(events) == 1
    assert isinstance(events[0], Done)
    assert events[0].message == AssistantMessage("")


async def test_a_tool_call_without_text_is_not_an_empty_reply() -> None:
    call_chunk = {
        "model": "qwen3:8b",
        "message": {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"function": {"name": "open_app"}}],
        },
        "done": False,
    }
    server = FakeServer([httpx.Response(200, content=ndjson(call_chunk, done_chunk()))])

    await collect(make_client(server, think=False), [UserMessage("öppna")], [TOOL])

    assert len(server.requests) == 1
