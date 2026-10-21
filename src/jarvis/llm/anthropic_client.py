"""Anthropic Claude client with streaming and native tool calling.

Uses the official async SDK's ``messages.stream`` and translates its raw
server-sent events into provider-neutral :mod:`jarvis.llm.base` events.
Retries live in :mod:`jarvis.utils.retry` rather than the SDK so that one
policy and one log format cover every provider; the SDK's own retries are
switched off. The SDK object is injected through a narrow Protocol so tests
replay recorded event objects instead of talking to the network.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Sequence
from contextlib import AsyncExitStack
from typing import TYPE_CHECKING, Any, Final, Protocol, cast

import anthropic
from anthropic import Omit, omit
from anthropic.lib.streaming import AsyncMessageStream, AsyncMessageStreamManager
from anthropic.types import (
    ContentBlockParam,
    MessageParam,
    RawContentBlockDeltaEvent,
    RawContentBlockStartEvent,
    RawContentBlockStopEvent,
    RawMessageDeltaEvent,
    RawMessageStartEvent,
    RawMessageStopEvent,
    ToolParam,
)

from jarvis.llm.base import (
    AssistantMessage,
    Done,
    LLMClient,
    LLMError,
    LLMEvent,
    Message,
    StopReason,
    SystemMessage,
    TextDelta,
    ToolCallRequested,
    ToolDefinition,
    ToolResultMessage,
    Usage,
    UserMessage,
)
from jarvis.llm.streaming import ToolCallAccumulator
from jarvis.utils.errors import ConfigError
from jarvis.utils.logging import get_logger
from jarvis.utils.retry import DEFAULT_RETRY_POLICY, RetryPolicy, SleepFn, retry_async

if TYPE_CHECKING:
    from jarvis.config import Settings

DEFAULT_MAX_TOKENS: Final = 4096
"""Enough for any spoken answer plus tool calls; long documents raise it per call."""

_SERVER_ERROR_STATUS: Final = 500

_STOP_REASONS: Final[dict[str, StopReason]] = {
    "end_turn": StopReason.END_TURN,
    "tool_use": StopReason.TOOL_USE,
    "max_tokens": StopReason.MAX_TOKENS,
    "refusal": StopReason.REFUSAL,
}

log = get_logger(__name__)


class MessageStreamer(Protocol):
    """The slice of ``anthropic.AsyncAnthropic.messages`` this client uses."""

    def stream(
        self,
        *,
        model: str,
        max_tokens: int,
        messages: list[MessageParam],
        system: str | Omit,
        tools: list[ToolParam] | Omit,
    ) -> AsyncMessageStreamManager[Any]:
        """Open a streaming request; the context manager performs it."""


class AnthropicClient(LLMClient):
    """``LLMClient`` over the Anthropic Messages API."""

    def __init__(
        self,
        streamer: MessageStreamer,
        *,
        model: str,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        policy: RetryPolicy = DEFAULT_RETRY_POLICY,
        sleep: SleepFn = asyncio.sleep,
    ) -> None:
        self._streamer = streamer
        self._model = model
        self._max_tokens = max_tokens
        self._policy = policy
        self._sleep = sleep

    @classmethod
    def from_settings(cls, settings: Settings) -> AnthropicClient:
        """Build the real client from ``Settings``.

        Raises:
            ConfigError: If no API key is configured.
        """
        if settings.anthropic_api_key is None:
            raise ConfigError(
                "ANTHROPIC_API_KEY is not set. Add it to .env, or set "
                "JARVIS_LLM_PROVIDER=ollama to use a local model."
            )
        # SDK retries are off: jarvis.utils.retry owns the policy and the logs.
        sdk = anthropic.AsyncAnthropic(
            api_key=settings.anthropic_api_key.get_secret_value(), max_retries=0
        )
        return cls(sdk.messages, model=settings.llm_model)

    @property
    def model(self) -> str:
        """The Claude model id requests are sent with."""
        return self._model

    async def stream(
        self, messages: Sequence[Message], tools: Sequence[ToolDefinition] = ()
    ) -> AsyncIterator[LLMEvent]:
        """Stream one turn; see :meth:`LLMClient.stream`."""
        system, params = to_message_params(messages)
        tool_params: list[ToolParam] | Omit = [to_tool_param(tool) for tool in tools] or omit
        translator = _EventTranslator()
        try:
            async with AsyncExitStack() as stack:
                stream = await retry_async(
                    lambda: self._open(stack, system, params, tool_params),
                    is_retryable=is_retryable,
                    policy=self._policy,
                    sleep=self._sleep,
                    operation_name="anthropic.messages.stream",
                )
                async for event in stream:
                    translated = translator.translate(event)
                    if translated is not None:
                        yield translated
        except anthropic.APIError as error:
            raise LLMError(_describe(error)) from error
        if not translator.finished:
            raise LLMError("Anthropic stream ended without a message_stop event")

    async def _open(
        self,
        stack: AsyncExitStack,
        system: str | Omit,
        params: list[MessageParam],
        tools: list[ToolParam] | Omit,
    ) -> AsyncMessageStream[Any]:
        manager = self._streamer.stream(
            model=self._model,
            max_tokens=self._max_tokens,
            system=system,
            messages=params,
            tools=tools,
        )
        return await stack.enter_async_context(manager)


def is_retryable(error: Exception) -> bool:
    """Rate limits, server errors and connection failures are transient."""
    if isinstance(error, anthropic.RateLimitError | anthropic.APIConnectionError):
        return True
    return isinstance(error, anthropic.APIStatusError) and error.status_code >= _SERVER_ERROR_STATUS


def _describe(error: anthropic.APIError) -> str:
    if isinstance(error, anthropic.AuthenticationError):
        return "Anthropic rejected the API key; check ANTHROPIC_API_KEY in .env"
    if isinstance(error, anthropic.APIConnectionError):
        return f"could not reach the Anthropic API: {error}"
    return f"Anthropic API error: {error}"


def to_message_params(messages: Sequence[Message]) -> tuple[str | Omit, list[MessageParam]]:
    """Split the history into the ``system`` string and the ``messages`` list.

    System messages are hoisted and joined because the Messages API takes
    them as a separate parameter rather than as turns.
    """
    system_parts: list[str] = []
    params: list[MessageParam] = []
    for message in messages:
        if isinstance(message, SystemMessage):
            system_parts.append(message.text)
        elif isinstance(message, UserMessage):
            params.append({"role": "user", "content": _user_content(message)})
        elif isinstance(message, AssistantMessage):
            params.append({"role": "assistant", "content": _assistant_content(message)})
        elif isinstance(message, ToolResultMessage):
            params.append({"role": "user", "content": _tool_result_content(message)})
    return ("\n\n".join(system_parts) or omit), params


def _user_content(message: UserMessage) -> str | list[ContentBlockParam]:
    if not message.images:
        return message.text
    blocks: list[ContentBlockParam] = [
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": image.media_type,
                "data": image.as_base64(),
            },
        }
        for image in message.images
    ]
    blocks.append({"type": "text", "text": message.text})
    return blocks


def _assistant_content(message: AssistantMessage) -> list[ContentBlockParam]:
    if message.provider_state:
        # Replayed verbatim: thinking blocks are signed and must not change.
        return [cast(ContentBlockParam, dict(block)) for block in message.provider_state]
    blocks: list[ContentBlockParam] = []
    if message.text:
        blocks.append({"type": "text", "text": message.text})
    blocks.extend(
        {"type": "tool_use", "id": call.id, "name": call.name, "input": call.arguments}
        for call in message.tool_calls
    )
    return blocks


def _tool_result_content(message: ToolResultMessage) -> list[ContentBlockParam]:
    return [
        {
            "type": "tool_result",
            "tool_use_id": result.tool_call_id,
            "content": result.content,
            "is_error": result.is_error,
        }
        for result in message.results
    ]


def to_tool_param(tool: ToolDefinition) -> ToolParam:
    """Convert a neutral tool definition to the API's shape."""
    return {
        "name": tool.name,
        "description": tool.description,
        "input_schema": dict(tool.input_schema),
    }


class _EventTranslator:
    """Folds raw stream events into LLM events and the replayable assistant turn."""

    def __init__(self) -> None:
        self._tool_calls = ToolCallAccumulator()
        self._blocks: dict[int, dict[str, Any]] = {}
        self._text_parts: list[str] = []
        self._calls: list[Any] = []
        self._usage = Usage()
        self._stop_reason = StopReason.OTHER
        self.finished = False

    def translate(self, event: object) -> LLMEvent | None:
        """Return the LLM event for ``event``, or ``None`` if it is bookkeeping only."""
        if isinstance(event, RawMessageStartEvent):
            self._usage = _usage_from(event.message.usage)
        elif isinstance(event, RawContentBlockStartEvent):
            self._start_block(event)
        elif isinstance(event, RawContentBlockDeltaEvent):
            return self._apply_delta(event)
        elif isinstance(event, RawContentBlockStopEvent):
            return self._stop_block(event.index)
        elif isinstance(event, RawMessageDeltaEvent):
            self._stop_reason = _STOP_REASONS.get(event.delta.stop_reason or "", StopReason.OTHER)
            self._usage = _merge_usage(self._usage, event.usage)
        elif isinstance(event, RawMessageStopEvent):
            self.finished = True
            return self._done()
        return None

    def _start_block(self, event: RawContentBlockStartEvent) -> None:
        block = event.content_block
        if block.type == "tool_use":
            self._tool_calls.start(event.index, block.id, block.name)
            self._blocks[event.index] = {"type": "tool_use", "id": block.id, "name": block.name}
        elif block.type == "text":
            self._blocks[event.index] = {"type": "text", "text": block.text}
        elif block.type == "thinking":
            self._blocks[event.index] = {
                "type": "thinking",
                "thinking": block.thinking,
                "signature": block.signature,
            }
        elif block.type == "redacted_thinking":
            self._blocks[event.index] = {"type": "redacted_thinking", "data": block.data}

    def _apply_delta(self, event: RawContentBlockDeltaEvent) -> LLMEvent | None:
        delta = event.delta
        block = self._blocks.get(event.index)
        if delta.type == "text_delta":
            self._text_parts.append(delta.text)
            if block is not None:
                block["text"] = block.get("text", "") + delta.text
            return TextDelta(delta.text)
        if delta.type == "input_json_delta":
            self._tool_calls.append(event.index, delta.partial_json)
        elif delta.type == "thinking_delta" and block is not None:
            block["thinking"] = block.get("thinking", "") + delta.thinking
        elif delta.type == "signature_delta" and block is not None:
            block["signature"] = delta.signature
        return None

    def _stop_block(self, index: int) -> LLMEvent | None:
        if not self._tool_calls.is_tracking(index):
            return None
        call = self._tool_calls.finish(index)
        self._blocks[index]["input"] = call.arguments
        self._calls.append(call)
        return ToolCallRequested(call)

    def _done(self) -> Done:
        blocks = tuple(self._blocks[index] for index in sorted(self._blocks))
        message = AssistantMessage(
            text="".join(self._text_parts),
            tool_calls=tuple(self._calls),
            # The API rejects empty text blocks on replay, so they are dropped.
            provider_state=tuple(b for b in blocks if b["type"] != "text" or b["text"]),
        )
        log.info(
            "llm.turn_done",
            provider="anthropic",
            stop_reason=self._stop_reason.value,
            input_tokens=self._usage.input_tokens,
            output_tokens=self._usage.output_tokens,
            cache_read_tokens=self._usage.cache_read_tokens,
            tool_calls=len(self._calls),
        )
        return Done(stop_reason=self._stop_reason, usage=self._usage, message=message)


def _usage_from(usage: Any) -> Usage:
    return Usage(
        input_tokens=getattr(usage, "input_tokens", None) or 0,
        output_tokens=getattr(usage, "output_tokens", None) or 0,
        cache_read_tokens=getattr(usage, "cache_read_input_tokens", None) or 0,
        cache_write_tokens=getattr(usage, "cache_creation_input_tokens", None) or 0,
    )


def _merge_usage(current: Usage, delta: Any) -> Usage:
    # message_delta carries cumulative counts; a None field means "unchanged".
    update = _usage_from(delta)
    return Usage(
        input_tokens=update.input_tokens or current.input_tokens,
        output_tokens=update.output_tokens or current.output_tokens,
        cache_read_tokens=update.cache_read_tokens or current.cache_read_tokens,
        cache_write_tokens=update.cache_write_tokens or current.cache_write_tokens,
    )


# Design notes:
# - No `thinking` parameter: the configured Sonnet runs adaptive thinking by
#   default, and the thinking blocks it streams are captured into
#   provider_state so tool-use turns replay correctly. Effort tuning is a
#   later, measured change (ADR-0005).
# - Tools are declared without `strict` or eager input streaming: our tool
#   inputs are a few short fields validated by pydantic in the agent anyway,
#   and eager streaming would move validation from the server to us for no
#   latency gain on inputs that small.
# - Refusals are not auto-retried on another model: the assistant should
#   say it declined rather than quietly answer with a different model.
# - Only establishing the stream is retried. Once events have been yielded
#   the caller may already be speaking them; replaying would repeat speech.
