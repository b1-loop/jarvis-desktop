"""Ollama backed ``LLMClient`` over the local HTTP API.

Ollama runs open models on the user's own machine and exposes a chat API
with tool calling (``POST /api/chat``, ``stream=true``). It is the middle
ground between Claude and the fully offline llama.cpp wrapper: private and
free, but an extra server to run and a smaller model. Only ``httpx`` is
needed, and tests drive the client through ``httpx.MockTransport``.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import AsyncIterator, Sequence
from contextlib import AsyncExitStack
from typing import TYPE_CHECKING, Any, Final

import httpx

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
    ToolCall,
    ToolCallRequested,
    ToolDefinition,
    ToolResultMessage,
    Usage,
    UserMessage,
)
from jarvis.utils.logging import get_logger
from jarvis.utils.retry import DEFAULT_RETRY_POLICY, RetryPolicy, SleepFn, retry_async

if TYPE_CHECKING:
    from jarvis.config import Settings

CHAT_PATH: Final = "/api/chat"
DEFAULT_TIMEOUT: Final = httpx.Timeout(120.0, connect=5.0)
"""Generation on a laptop GPU can take a while; connecting should not."""

_SERVER_ERROR_STATUS: Final = 500
_NOT_FOUND_STATUS: Final = 404

log = get_logger(__name__)


class OllamaClient(LLMClient):
    """``LLMClient`` over Ollama's streaming chat endpoint."""

    def __init__(
        self,
        http: httpx.AsyncClient,
        *,
        model: str,
        policy: RetryPolicy = DEFAULT_RETRY_POLICY,
        sleep: SleepFn = asyncio.sleep,
    ) -> None:
        self._http = http
        self._model = model
        self._policy = policy
        self._sleep = sleep

    @classmethod
    def from_settings(cls, settings: Settings) -> OllamaClient:
        """Build the real client pointed at ``settings.ollama_base_url``."""
        http = httpx.AsyncClient(base_url=settings.ollama_base_url, timeout=DEFAULT_TIMEOUT)
        return cls(http, model=settings.ollama_model)

    @property
    def model(self) -> str:
        """The Ollama model tag requests are sent with."""
        return self._model

    async def aclose(self) -> None:
        """Release the HTTP connection pool."""
        await self._http.aclose()

    async def stream(
        self, messages: Sequence[Message], tools: Sequence[ToolDefinition] = ()
    ) -> AsyncIterator[LLMEvent]:
        """Stream one turn; see :meth:`LLMClient.stream`."""
        payload = self._payload(messages, tools)
        translator = _ChunkTranslator()
        try:
            async with AsyncExitStack() as stack:
                response = await retry_async(
                    lambda: self._open(stack, payload),
                    is_retryable=is_retryable,
                    policy=self._policy,
                    sleep=self._sleep,
                    operation_name="ollama.chat",
                )
                async for line in response.aiter_lines():
                    if line.strip():
                        for event in translator.translate(_parse_line(line)):
                            yield event
        except httpx.HTTPError as error:
            raise LLMError(self._describe(error)) from error
        if not translator.finished:
            raise LLMError("Ollama stream ended before the final chunk")

    def _payload(
        self, messages: Sequence[Message], tools: Sequence[ToolDefinition]
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": [param for message in messages for param in to_message_params(message)],
            "stream": True,
        }
        if tools:
            payload["tools"] = [to_tool_param(tool) for tool in tools]
        return payload

    async def _open(self, stack: AsyncExitStack, payload: dict[str, Any]) -> httpx.Response:
        request = self._http.stream("POST", CHAT_PATH, json=payload)
        response = await stack.enter_async_context(request)
        if response.is_error:
            await response.aread()
            response.raise_for_status()
        return response

    def _describe(self, error: httpx.HTTPError) -> str:
        if isinstance(error, httpx.HTTPStatusError):
            if error.response.status_code == _NOT_FOUND_STATUS:
                return (
                    f"Ollama does not have the model '{self._model}'; "
                    f"run 'ollama pull {self._model}' or change JARVIS_OLLAMA_MODEL"
                )
            return f"Ollama returned HTTP {error.response.status_code}: {error.response.text}"
        return (
            f"could not reach Ollama at {self._http.base_url}: {error}. "
            "Is it running? Start it with 'ollama serve' or fix JARVIS_OLLAMA_BASE_URL."
        )


def is_retryable(error: Exception) -> bool:
    """Connection problems and server errors are transient; 4xx are not."""
    if isinstance(error, httpx.TransportError):
        return True
    return (
        isinstance(error, httpx.HTTPStatusError)
        and error.response.status_code >= _SERVER_ERROR_STATUS
    )


def to_message_params(message: Message) -> list[dict[str, Any]]:
    """Convert one neutral message into Ollama chat messages (one per tool result)."""
    if isinstance(message, SystemMessage):
        return [{"role": "system", "content": message.text}]
    if isinstance(message, UserMessage):
        param: dict[str, Any] = {"role": "user", "content": message.text}
        if message.images:
            param["images"] = [image.as_base64() for image in message.images]
        return [param]
    if isinstance(message, AssistantMessage):
        param = {"role": "assistant", "content": message.text}
        if message.tool_calls:
            param["tool_calls"] = [
                {"function": {"name": call.name, "arguments": call.arguments}}
                for call in message.tool_calls
            ]
        return [param]
    if isinstance(message, ToolResultMessage):
        return [
            {"role": "tool", "content": result.content, "tool_name": result.name}
            for result in message.results
        ]
    raise TypeError(f"unsupported message type: {type(message).__name__}")


def to_tool_param(tool: ToolDefinition) -> dict[str, Any]:
    """Convert a neutral tool definition to Ollama's OpenAI-style shape."""
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": dict(tool.input_schema),
        },
    }


def _parse_line(line: str) -> dict[str, Any]:
    try:
        chunk = json.loads(line)
    except json.JSONDecodeError as error:
        raise LLMError(f"Ollama sent a line that is not JSON: {line[:80]!r}") from error
    if not isinstance(chunk, dict):
        raise LLMError("Ollama sent a JSON value that is not an object")
    if "error" in chunk:
        raise LLMError(f"Ollama reported an error: {chunk['error']}")
    return chunk


class _ChunkTranslator:
    """Folds NDJSON chunks into LLM events."""

    def __init__(self) -> None:
        self._text_parts: list[str] = []
        self._calls: list[ToolCall] = []
        self.finished = False

    def translate(self, chunk: dict[str, Any]) -> list[LLMEvent]:
        """Events for one chunk; the final chunk yields ``Done``."""
        events: list[LLMEvent] = []
        message = chunk.get("message") or {}
        text = message.get("content") or ""
        if text:
            self._text_parts.append(text)
            events.append(TextDelta(text))
        for raw_call in message.get("tool_calls") or []:
            call = _to_tool_call(raw_call)
            self._calls.append(call)
            events.append(ToolCallRequested(call))
        if chunk.get("done"):
            self.finished = True
            events.append(self._done(chunk))
        return events

    def _done(self, chunk: dict[str, Any]) -> Done:
        usage = Usage(
            input_tokens=int(chunk.get("prompt_eval_count") or 0),
            output_tokens=int(chunk.get("eval_count") or 0),
        )
        stop_reason = _stop_reason(chunk.get("done_reason"), has_tool_calls=bool(self._calls))
        log.info(
            "llm.turn_done",
            provider="ollama",
            model=chunk.get("model"),
            stop_reason=stop_reason.value,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            tool_calls=len(self._calls),
        )
        message = AssistantMessage(text="".join(self._text_parts), tool_calls=tuple(self._calls))
        return Done(stop_reason=stop_reason, usage=usage, message=message)


def _to_tool_call(raw: dict[str, Any]) -> ToolCall:
    function = raw.get("function") or {}
    name = function.get("name")
    arguments = function.get("arguments")
    if not isinstance(name, str) or not name:
        raise LLMError("Ollama sent a tool call without a function name")
    if isinstance(arguments, str):
        # Some models emit the arguments as a JSON string instead of an object.
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError as error:
            raise LLMError(f"Ollama tool call '{name}' has arguments that are not JSON") from error
    if arguments is None:
        arguments = {}
    if not isinstance(arguments, dict):
        raise LLMError(f"Ollama tool call '{name}' sent a non-object argument")
    call_id = raw.get("id") or f"call_{uuid.uuid4().hex}"
    return ToolCall(id=str(call_id), name=name, arguments=arguments)


def _stop_reason(done_reason: object, *, has_tool_calls: bool) -> StopReason:
    if has_tool_calls:
        return StopReason.TOOL_USE
    if done_reason == "stop":
        return StopReason.END_TURN
    if done_reason == "length":
        return StopReason.MAX_TOKENS
    return StopReason.OTHER


# Design notes:
# - Ollama issues no tool-call ids (older versions) or optional ones, so the
#   client mints one per call; results are matched by tool name, which is
#   what Ollama's `tool` role uses (`tool_name`).
# - Tool calls arrive whole in one chunk rather than as JSON fragments, so
#   there is no accumulator here; the shared SentenceSplitter still applies
#   to the text deltas.
# - A 404 almost always means the model was never pulled, so the error
#   message says exactly which command fixes it.
