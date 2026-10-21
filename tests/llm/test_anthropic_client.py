import os
from collections.abc import Callable, Sequence

import anthropic
import httpx2
import pytest
from anthropic import omit

from jarvis.config import Settings
from jarvis.llm.anthropic_client import (
    AnthropicClient,
    is_retryable,
    to_message_params,
    to_tool_param,
)
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
from jarvis.utils.errors import ConfigError
from jarvis.utils.retry import RetryPolicy
from tests.llm.fakes import (
    FakeStreamer,
    message_delta,
    message_start,
    message_stop,
    redacted_thinking_block,
    text_block,
    text_turn,
    thinking_block,
    tool_block,
)

REQUEST = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
TOOL = ToolDefinition(
    name="open_app",
    description="Open an application",
    input_schema={"type": "object", "properties": {"name": {"type": "string"}}},
)


def api_error(status: int) -> anthropic.APIStatusError:
    response = httpx2.Response(status, request=REQUEST)
    classes: dict[int, type[anthropic.APIStatusError]] = {
        400: anthropic.BadRequestError,
        401: anthropic.AuthenticationError,
        429: anthropic.RateLimitError,
        500: anthropic.InternalServerError,
        529: anthropic.APIStatusError,
    }
    return classes[status](f"status {status}", response=response, body=None)


async def collect(
    client: AnthropicClient, messages: Sequence[Message], tools: Sequence[ToolDefinition] = ()
) -> list[LLMEvent]:
    return [event async for event in client.stream(messages, tools)]


def make_client(streamer: FakeStreamer, **kwargs: object) -> AnthropicClient:
    async def no_sleep(_seconds: float) -> None:
        return None

    return AnthropicClient(streamer, model="claude-sonnet-5-5", sleep=no_sleep, **kwargs)  # type: ignore[arg-type]


async def test_text_turn_streams_deltas_then_done() -> None:
    streamer = FakeStreamer([text_turn("Hello", ", world.")])
    client = make_client(streamer)

    events = await collect(client, [UserMessage("hi")])

    assert events[:2] == [TextDelta("Hello"), TextDelta(", world.")]
    done = events[2]
    assert isinstance(done, Done)
    assert done.stop_reason is StopReason.END_TURN
    assert done.usage.input_tokens == 10
    assert done.usage.output_tokens == 7
    assert done.message.text == "Hello, world."
    assert done.message.tool_calls == ()
    assert done.message.provider_state == ({"type": "text", "text": "Hello, world."},)


async def test_request_carries_model_system_messages_and_tools() -> None:
    streamer = FakeStreamer([text_turn("ok")])
    client = make_client(streamer, max_tokens=512)

    await collect(client, [SystemMessage("Be brief."), UserMessage("hi")], [TOOL])

    request = streamer.requests[0]
    assert request["model"] == "claude-sonnet-5-5"
    assert request["max_tokens"] == 512
    assert request["system"] == "Be brief."
    assert request["messages"] == [{"role": "user", "content": "hi"}]
    assert request["tools"] == [to_tool_param(TOOL)]


async def test_tools_are_omitted_when_none_are_offered() -> None:
    streamer = FakeStreamer([text_turn("ok")])

    await collect(make_client(streamer), [UserMessage("hi")])

    assert streamer.requests[0]["tools"] is omit
    assert streamer.requests[0]["system"] is omit


async def test_tool_use_turn_yields_parsed_call_and_replayable_state() -> None:
    events = [
        message_start(),
        *text_block(0, "Opening it."),
        *tool_block(1, "toolu_1", "open_app", '{"name": "co', 'de"}'),
        message_delta("tool_use"),
        message_stop(),
    ]
    client = make_client(FakeStreamer([events]))

    result = await collect(client, [UserMessage("open code")], [TOOL])

    assert result[0] == TextDelta("Opening it.")
    assert result[1] == ToolCallRequested(ToolCall("toolu_1", "open_app", {"name": "code"}))
    done = result[2]
    assert isinstance(done, Done)
    assert done.stop_reason is StopReason.TOOL_USE
    assert done.message.tool_calls == (ToolCall("toolu_1", "open_app", {"name": "code"}),)
    assert done.message.provider_state == (
        {"type": "text", "text": "Opening it."},
        {"type": "tool_use", "id": "toolu_1", "name": "open_app", "input": {"name": "code"}},
    )


async def test_thinking_blocks_are_captured_for_replay_but_not_spoken() -> None:
    events = [
        message_start(),
        *thinking_block(0, "let me think", "sig123"),
        *redacted_thinking_block(1, "opaque"),
        *text_block(2, "Done."),
        message_delta("end_turn"),
        message_stop(),
    ]
    client = make_client(FakeStreamer([events]))

    result = await collect(client, [UserMessage("hi")])

    assert [e for e in result if isinstance(e, TextDelta)] == [TextDelta("Done.")]
    done = result[-1]
    assert isinstance(done, Done)
    assert done.message.provider_state == (
        {"type": "thinking", "thinking": "let me think", "signature": "sig123"},
        {"type": "redacted_thinking", "data": "opaque"},
        {"type": "text", "text": "Done."},
    )


async def test_empty_text_blocks_are_dropped_from_the_replay_state() -> None:
    client = make_client(FakeStreamer([text_turn()]))

    done = (await collect(client, [UserMessage("hi")]))[-1]

    assert isinstance(done, Done)
    assert done.message.provider_state == ()


async def test_refusal_and_max_tokens_stop_reasons_are_mapped() -> None:
    client = make_client(
        FakeStreamer(
            [
                text_turn("no", stop_reason="refusal"),
                text_turn("x", stop_reason="max_tokens"),
                text_turn("y", stop_reason="pause_turn"),
            ]
        )
    )

    refused = (await collect(client, [UserMessage("hi")]))[-1]
    truncated = (await collect(client, [UserMessage("hi")]))[-1]
    other = (await collect(client, [UserMessage("hi")]))[-1]

    assert isinstance(refused, Done) and refused.stop_reason is StopReason.REFUSAL
    assert isinstance(truncated, Done) and truncated.stop_reason is StopReason.MAX_TOKENS
    assert isinstance(other, Done) and other.stop_reason is StopReason.OTHER


async def test_rate_limit_is_retried_and_the_second_attempt_streams() -> None:
    sleeps: list[float] = []

    async def record_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    streamer = FakeStreamer([api_error(429), text_turn("after retry")])
    client = AnthropicClient(
        streamer,
        model="m",
        sleep=record_sleep,
        policy=RetryPolicy(max_attempts=2, base_delay=0.1, jitter=0),
    )

    events = await collect(client, [UserMessage("hi")])

    assert events[0] == TextDelta("after retry")
    assert sleeps == [0.1]
    assert len(streamer.requests) == 2


async def test_bad_request_is_not_retried_and_becomes_llm_error() -> None:
    streamer = FakeStreamer([api_error(400)])
    client = make_client(streamer)

    with pytest.raises(LLMError, match="Anthropic API error") as excinfo:
        await collect(client, [UserMessage("hi")])

    assert len(streamer.requests) == 1
    assert isinstance(excinfo.value.__cause__, anthropic.BadRequestError)


async def test_authentication_error_points_at_the_api_key() -> None:
    client = make_client(FakeStreamer([api_error(401)]))

    with pytest.raises(LLMError, match="ANTHROPIC_API_KEY"):
        await collect(client, [UserMessage("hi")])


async def test_exhausted_retries_surface_as_llm_error() -> None:
    streamer = FakeStreamer([api_error(500), api_error(500)])
    client = make_client(streamer, policy=RetryPolicy(max_attempts=2, base_delay=0.0))

    with pytest.raises(LLMError, match="Anthropic API error"):
        await collect(client, [UserMessage("hi")])

    assert len(streamer.requests) == 2


async def test_connection_error_is_described_as_unreachable() -> None:
    error = anthropic.APIConnectionError(request=REQUEST)
    client = make_client(FakeStreamer([error]), policy=RetryPolicy(max_attempts=1))

    with pytest.raises(LLMError, match="could not reach"):
        await collect(client, [UserMessage("hi")])


async def test_stream_without_message_stop_is_an_error() -> None:
    client = make_client(FakeStreamer([[message_start(), *text_block(0, "partial")]]))

    with pytest.raises(LLMError, match="message_stop"):
        await collect(client, [UserMessage("hi")])


def test_is_retryable_classifies_sdk_errors() -> None:
    assert is_retryable(api_error(429))
    assert is_retryable(api_error(500))
    assert is_retryable(api_error(529))
    assert is_retryable(anthropic.APIConnectionError(request=REQUEST))
    assert not is_retryable(api_error(400))
    assert not is_retryable(api_error(401))
    assert not is_retryable(ValueError("nope"))


def test_history_is_converted_to_the_messages_api_shape() -> None:
    image = ImageContent(b"\x89PNG", "image/png")
    history: list[Message] = [
        SystemMessage("You are Jarvis."),
        SystemMessage("Be brief."),
        UserMessage("What is on my screen?", images=(image,)),
        AssistantMessage("Let me look.", tool_calls=(ToolCall("t1", "screenshot", {}),)),
        ToolResultMessage((ToolResult("t1", "screenshot", "a terminal", is_error=False),)),
        AssistantMessage("A terminal."),
    ]

    system, params = to_message_params(history)

    assert system == "You are Jarvis.\n\nBe brief."
    assert params == [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": image.as_base64(),
                    },
                },
                {"type": "text", "text": "What is on my screen?"},
            ],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Let me look."},
                {"type": "tool_use", "id": "t1", "name": "screenshot", "input": {}},
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "t1",
                    "content": "a terminal",
                    "is_error": False,
                }
            ],
        },
        {"role": "assistant", "content": [{"type": "text", "text": "A terminal."}]},
    ]


def test_provider_state_is_replayed_verbatim() -> None:
    state = (
        {"type": "thinking", "thinking": "hmm", "signature": "sig"},
        {"type": "tool_use", "id": "t1", "name": "lock_screen", "input": {}},
    )
    history: list[Message] = [AssistantMessage("", (ToolCall("t1", "lock_screen"),), state)]

    _, params = to_message_params(history)

    assert params == [{"role": "assistant", "content": [dict(state[0]), dict(state[1])]}]


def test_tool_param_has_name_description_and_schema() -> None:
    assert to_tool_param(TOOL) == {
        "name": "open_app",
        "description": "Open an application",
        "input_schema": {"type": "object", "properties": {"name": {"type": "string"}}},
    }


def test_from_settings_requires_an_api_key(make_settings: Callable[..., Settings]) -> None:
    with pytest.raises(ConfigError, match="ANTHROPIC_API_KEY"):
        AnthropicClient.from_settings(make_settings())

    client = AnthropicClient.from_settings(
        make_settings(anthropic_api_key="sk-ant-test", llm_model="claude-opus-5-5")
    )
    assert client.model == "claude-opus-5-5"


@pytest.mark.integration
@pytest.mark.skipif(not os.environ.get("ANTHROPIC_API_KEY"), reason="needs ANTHROPIC_API_KEY")
async def test_real_api_streams_a_short_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = Settings(_env_file=None, anthropic_api_key=os.environ["ANTHROPIC_API_KEY"])
    client = AnthropicClient.from_settings(settings)

    events = await collect(client, [UserMessage("Reply with the single word OK.")])

    done = events[-1]
    assert isinstance(done, Done)
    assert "ok" in done.message.text.lower()
    assert done.usage.output_tokens > 0
