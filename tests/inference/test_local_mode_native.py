"""LOCAL-mode providers against the real native module, models and Piper voice.

These need ``scripts/build_native.py`` and ``scripts/download_models.py
--local`` (and the ``local`` extra for Piper); each test skips cleanly when
its prerequisite is missing. Model output is non-deterministic in content,
so the assertions check shape and plumbing, plus one offline round trip
(Piper speaks, whisper.cpp hears) whose words are stable.
"""

from __future__ import annotations

import importlib.util
from collections.abc import AsyncIterator, Callable

import numpy as np
import pytest

from jarvis.config import RuntimeMode, Settings
from jarvis.core.providers import build_llm, build_stt, build_tts
from jarvis.llm.base import (
    AssistantMessage,
    Done,
    LLMClient,
    LLMEvent,
    Message,
    SystemMessage,
    TextDelta,
    ToolCall,
    ToolDefinition,
    ToolResult,
    ToolResultMessage,
    UserMessage,
)
from jarvis.stt.base import STT
from jarvis.stt.faster_whisper import resample_linear
from jarvis.tts.base import TTS, AudioChunk
from jarvis.utils.errors import ConfigError

pytestmark = pytest.mark.native


@pytest.fixture(scope="module")
def settings() -> Settings:
    return Settings(_env_file=None, mode=RuntimeMode.LOCAL, local_llm_context_size=1024)


def build_or_skip[T](builder: Callable[[Settings], T], settings: Settings) -> T:
    try:
        return builder(settings)
    except ConfigError as error:
        pytest.skip(f"LOCAL mode prerequisite missing: {error}")


@pytest.fixture(scope="module")
def stt(settings: Settings) -> STT:
    return build_or_skip(build_stt, settings)


@pytest.fixture(scope="module")
def llm(settings: Settings) -> LLMClient:
    return build_or_skip(build_llm, settings)


@pytest.fixture(scope="module")
def tts(settings: Settings) -> TTS:
    if importlib.util.find_spec("piper") is None:
        pytest.skip("piper-tts is not installed (uv sync --extra local)")
    return build_or_skip(build_tts, settings)


async def collect_audio(chunks: AsyncIterator[AudioChunk]) -> list[AudioChunk]:
    return [chunk async for chunk in chunks]


async def test_whisper_cpp_hears_silence_as_an_empty_transcript(stt: STT) -> None:
    transcript = await stt.transcribe(np.zeros(16_000, dtype=np.float32))

    assert transcript.is_empty
    assert transcript.duration == pytest.approx(1.0)


async def test_llama_cpp_streams_text_and_finishes_with_done(llm: LLMClient) -> None:
    messages: list[Message] = [
        SystemMessage("Answer in one short sentence."),
        UserMessage("Say hello."),
    ]

    events: list[LLMEvent] = [event async for event in llm.stream(messages)]

    assert isinstance(events[-1], Done)
    assert all(isinstance(event, TextDelta) for event in events[:-1])
    assert events[-1].message.text.strip()
    assert events[-1].usage.input_tokens > 0


async def test_piper_streams_float32_at_the_voice_rate(tts: TTS) -> None:
    chunks = await collect_audio(tts.synthesize("Good morning. All systems are running."))

    assert len(chunks) == 2  # one per sentence
    assert all(chunk.sample_rate == tts.sample_rate for chunk in chunks)
    assert all(chunk.samples.dtype == np.float32 for chunk in chunks)
    assert max(float(np.abs(chunk.samples).max()) for chunk in chunks) > 0.05


async def test_offline_round_trip_piper_to_whisper_cpp(tts: TTS, stt: STT) -> None:
    chunks = await collect_audio(tts.synthesize("Hello world."))
    speech = np.concatenate([chunk.samples for chunk in chunks])

    transcript = await stt.transcribe(resample_linear(speech, tts.sample_rate, 16_000))

    assert "hello" in transcript.text.lower()


async def test_llama_cpp_answers_after_a_tool_call_with_a_lone_surrogate(llm: LLMClient) -> None:
    # Regression: replaying a call whose JSON held "\ud83d" raised a pybind11
    # cast error in format_prompt and crashed LOCAL mode after a tool round.
    clock = ToolDefinition("current_time", "The current time", {"type": "object", "properties": {}})
    call = ToolCall("call_1", "current_time", {"timezone": "\ud83d"})
    messages: list[Message] = [
        SystemMessage("Answer in one short sentence."),
        UserMessage("What time is it?"),
        AssistantMessage(text="", tool_calls=(call,)),
        ToolResultMessage((ToolResult("call_1", "current_time", "12:00"),)),
    ]

    events: list[LLMEvent] = [event async for event in llm.stream(messages, [clock])]

    assert isinstance(events[-1], Done)
