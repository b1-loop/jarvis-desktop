import asyncio
from collections.abc import Callable, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import BaseModel

from jarvis.config import Settings
from jarvis.llm import local_llama_client
from jarvis.llm.base import (
    AssistantMessage,
    Done,
    ImageContent,
    LLMClient,
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
from jarvis.llm.local_llama_client import (
    IMAGE_PLACEHOLDER,
    LocalLlamaClient,
    SamplingOptions,
    to_chat_messages,
)
from jarvis.utils.errors import ConfigError
from tests.llm.local_fakes import FakeGenerationStream, FakeLlama


class WeatherInput(BaseModel):
    city: str


WEATHER = ToolDefinition.from_model("get_weather", "Weather for a city", WeatherInput)
HELLO: list[Message] = [SystemMessage("Be brief."), UserMessage("Hi")]


def client_for(
    *streams: FakeGenerationStream, context: int = 4096
) -> tuple[LocalLlamaClient, FakeLlama]:
    llama = FakeLlama(streams=list(streams), context=context)
    return LocalLlamaClient(lambda: llama, model_name="tiny.gguf"), llama


async def collect(
    client: LLMClient, messages: Sequence[Message] = HELLO, tools: Sequence[ToolDefinition] = ()
) -> list[LLMEvent]:
    return [event async for event in client.stream(messages, tools)]


def text_of(events: list[LLMEvent]) -> str:
    return "".join(event.text for event in events if isinstance(event, TextDelta))


def done_of(events: list[LLMEvent]) -> Done:
    assert isinstance(events[-1], Done)
    return events[-1]


# --- text streaming ---------------------------------------------------------


async def test_pieces_stream_as_text_deltas_and_end_with_done() -> None:
    stream = FakeGenerationStream(["Hel", "lo", "!"])
    client, llama = client_for(stream)

    events = await collect(client)

    assert events[:3] == [TextDelta("Hel"), TextDelta("lo"), TextDelta("!")]
    done = done_of(events)
    assert done.stop_reason is StopReason.END_TURN
    assert done.message == AssistantMessage(text="Hello!")
    assert done.usage.output_tokens == 3
    assert done.usage.input_tokens == len(llama.options[0]["prompt"].split())
    assert client.model == "tiny.gguf"
    assert isinstance(client, LLMClient)


@pytest.mark.parametrize(
    ("native", "expected"),
    [
        ("end_of_text", StopReason.END_TURN),
        ("stop_string", StopReason.END_TURN),
        ("max_tokens", StopReason.MAX_TOKENS),
        ("context_full", StopReason.MAX_TOKENS),
        ("something_new", StopReason.OTHER),
    ],
)
async def test_native_finish_reasons_are_normalised(native: str, expected: StopReason) -> None:
    client, _ = client_for(FakeGenerationStream(["x"], reason=native))

    assert done_of(await collect(client)).stop_reason is expected


async def test_sampling_options_reach_the_native_generator() -> None:
    llama = FakeLlama(streams=[FakeGenerationStream(["x"])])
    sampling = SamplingOptions(max_tokens=64, temperature=0.1, top_p=0.5, top_k=7)
    client = LocalLlamaClient(lambda: llama, model_name="m", sampling=sampling)

    await collect(client)

    assert llama.options[0] | {"prompt": None} == {
        "prompt": None,
        "max_tokens": 64,
        "temperature": 0.1,
        "top_p": 0.5,
        "top_k": 7,
    }


async def test_model_loads_once_lazily() -> None:
    loads: list[int] = []
    llama = FakeLlama(streams=[FakeGenerationStream(["a"]), FakeGenerationStream(["b"])])

    def factory() -> FakeLlama:
        loads.append(1)
        return llama

    client = LocalLlamaClient(factory, model_name="m")
    await client.warm_up()
    await collect(client)
    await collect(client)

    assert loads == [1]


# --- tool calling -----------------------------------------------------------


async def test_tool_call_split_across_tokens_becomes_a_tool_call_request() -> None:
    pieces = ['{"', 'tool": "get', '_weather", "argu', 'ments": {"city": "Par', 'is"}}', " extra"]
    stream = FakeGenerationStream(pieces)
    client, llama = client_for(stream)

    events = await collect(client, tools=[WEATHER])

    assert not [event for event in events if isinstance(event, TextDelta)]
    request = events[0]
    assert isinstance(request, ToolCallRequested)
    assert request.call.name == "get_weather"
    assert request.call.arguments == {"city": "Paris"}
    assert request.call.id.startswith("call_")
    done = done_of(events)
    assert done.stop_reason is StopReason.TOOL_USE
    assert done.message.tool_calls == (request.call,)
    assert done.usage.output_tokens == 5
    assert stream.pulled == 5  # generation stopped at the closing brace
    assert stream.cancels >= 1
    assert "get_weather" in llama.prompts[0][0]["content"]


async def test_malformed_tool_json_falls_back_to_text() -> None:
    pieces = ['{"tool": "get_weather", ', '"arguments": {city: Paris}}']
    client, _ = client_for(FakeGenerationStream(pieces))

    events = await collect(client, tools=[WEATHER])

    assert text_of(events) == "".join(pieces)
    assert not any(isinstance(event, ToolCallRequested) for event in events)
    assert done_of(events).stop_reason is StopReason.END_TURN


async def test_unfinished_tool_json_is_spoken_rather_than_lost() -> None:
    client, _ = client_for(FakeGenerationStream(['{"tool": "get_wea'], reason="max_tokens"))

    events = await collect(client, tools=[WEATHER])

    assert text_of(events) == '{"tool": "get_wea'
    assert done_of(events).stop_reason is StopReason.MAX_TOKENS


async def test_without_tools_json_is_plain_text() -> None:
    client, llama = client_for(FakeGenerationStream(['{"tool": "get_weather"}']))

    events = await collect(client)

    assert text_of(events) == '{"tool": "get_weather"}'
    assert llama.prompts[0][0] == {"role": "system", "content": "Be brief."}


# --- cancellation and errors ---------------------------------------------------


async def test_closing_the_stream_early_cancels_native_generation() -> None:
    stream = FakeGenerationStream(["a", "b", "c"])
    client, _ = client_for(stream)

    events = client.stream(HELLO)
    first = await anext(events)
    await events.aclose()  # type: ignore[attr-defined]

    assert first == TextDelta("a")
    assert stream.cancels == 1
    assert stream.pulled == 1


async def test_cancelling_the_consumer_task_cancels_native_generation() -> None:
    stream = FakeGenerationStream(["a"], block_after=1)
    client, _ = client_for(stream)
    received: list[LLMEvent] = []
    first_arrived = asyncio.Event()

    async def consume() -> None:
        async for event in client.stream(HELLO):
            received.append(event)
            first_arrived.set()

    task = asyncio.create_task(consume())
    await first_arrived.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert stream.cancels >= 1
    assert received == [TextDelta("a")]


async def test_native_generation_error_becomes_llm_error() -> None:
    stream = FakeGenerationStream(["a"], failure=RuntimeError("decode failed"))
    client, _ = client_for(stream)

    with pytest.raises(LLMError, match="decode failed"):
        await collect(client)
    assert stream.cancels == 1


async def test_model_load_failure_becomes_llm_error() -> None:
    def broken() -> FakeLlama:
        raise RuntimeError("bad magic")

    client = LocalLlamaClient(broken, model_name="m.gguf")

    with pytest.raises(LLMError, match=r"m\.gguf: bad magic"):
        await collect(client)


async def test_config_errors_from_the_factory_pass_through() -> None:
    def unconfigured() -> FakeLlama:
        raise ConfigError("build it")

    with pytest.raises(ConfigError, match="build it"):
        await LocalLlamaClient(unconfigured, model_name="m").warm_up()


async def test_history_longer_than_the_context_is_rejected_with_a_remedy() -> None:
    client, llama = client_for(FakeGenerationStream(["x"]), context=2)

    with pytest.raises(LLMError, match="JARVIS_LOCAL_LLM_CONTEXT_SIZE"):
        await collect(client)
    assert not llama.options


# --- message conversion -------------------------------------------------------


def test_system_messages_are_hoisted_and_joined_with_the_tool_protocol() -> None:
    messages: list[Message] = [
        SystemMessage("Persona."),
        UserMessage("Weather?"),
        SystemMessage("Memory: likes Paris."),
    ]

    converted = to_chat_messages(messages, [WEATHER])

    assert converted[0]["role"] == "system"
    assert converted[0]["content"].startswith(
        "Persona.\n\nMemory: likes Paris.\n\nYou can use tools"
    )
    assert converted[1:] == [{"role": "user", "content": "Weather?"}]


def test_tool_turns_are_replayed_in_protocol_form() -> None:
    call = ToolCall("call_1", "get_weather", {"city": "Paris"})
    messages: list[Message] = [
        UserMessage("Weather?"),
        AssistantMessage(text="Checking.", tool_calls=(call,)),
        ToolResultMessage(
            (
                ToolResult("call_1", "get_weather", "Sunny, 21 C"),
                ToolResult("call_2", "get_news", "offline", is_error=True),
            )
        ),
    ]

    converted = to_chat_messages(messages)

    assert converted[1] == {
        "role": "assistant",
        "content": 'Checking.\n{"tool": "get_weather", "arguments": {"city": "Paris"}}',
    }
    assert converted[2] == {
        "role": "user",
        "content": "Tool get_weather returned: Sunny, 21 C\nTool get_news failed: offline",
    }


def test_images_are_replaced_by_a_placeholder() -> None:
    image = ImageContent(b"\x89PNG", "image/png")

    converted = to_chat_messages([UserMessage("What is this?", images=(image,))])

    assert converted == [{"role": "user", "content": f"What is this?\n{IMAGE_PLACEHOLDER}"}]


def test_unknown_message_types_are_a_programming_error() -> None:
    with pytest.raises(TypeError, match="str"):
        to_chat_messages(["not a message"])  # type: ignore[list-item]


# --- construction from settings -------------------------------------------------


def test_from_settings_requires_the_model_file(
    make_settings: Callable[..., Settings], tmp_path: Path
) -> None:
    settings = make_settings(local_llm_model=tmp_path / "missing.gguf")

    with pytest.raises(ConfigError, match=r"download_models\.py --local"):
        LocalLlamaClient.from_settings(settings)


async def test_from_settings_builds_a_lazy_native_generator(
    make_settings: Callable[..., Settings], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = tmp_path / "small.gguf"
    model.write_bytes(b"GGUF")
    constructed: list[dict[str, Any]] = []
    llama = FakeLlama(streams=[FakeGenerationStream(["ok"])])

    def generator(path: Path, *, n_ctx: int, n_threads: int) -> FakeLlama:
        constructed.append({"path": path, "n_ctx": n_ctx, "n_threads": n_threads})
        return llama

    module = SimpleNamespace(LlamaGenerator=generator)
    monkeypatch.setattr(local_llama_client, "load_inference_module", lambda: module)
    settings = make_settings(
        local_llm_model=model,
        local_llm_context_size=1024,
        local_llm_threads=3,
        local_llm_max_tokens=99,
        local_llm_temperature=0.2,
    )

    client = LocalLlamaClient.from_settings(settings)
    assert not constructed
    events = await collect(client)

    assert client.model == "small.gguf"
    assert constructed == [{"path": model, "n_ctx": 1024, "n_threads": 3}]
    assert llama.options[0]["max_tokens"] == 99
    assert llama.options[0]["temperature"] == 0.2
    assert text_of(events) == "ok"
