"""llama.cpp backed ``LLMClient`` via ``jarvis_inference`` (the LOCAL-mode LLM).

The native ``LlamaGenerator`` renders the model's own chat template and
generates on a native thread; Python pulls one piece of text per token from
a ``GenerationStream``. This client turns that pull API into the same
:data:`~jarvis.llm.base.LLMEvent` stream the cloud clients produce, and adds
prompt-level tool calling (:mod:`jarvis.llm.json_tool_protocol`) because
the wrapper has no structured tool channel.

Every blocking native call (load, prompt rendering, ``next_token``) runs in
``asyncio.to_thread``. When the consumer stops early (barge-in cancels the
task, or the agent closes the stream) the ``finally`` block cancels the
native generation, so the CPU is free again within one token.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Protocol, cast

from jarvis.inference.native import load_inference_module, require_model_file
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
    ToolResult,
    ToolResultMessage,
    Usage,
    UserMessage,
)
from jarvis.llm.json_tool_protocol import (
    ParsedToolCall,
    ToolCallParser,
    render_tool_call,
    render_tool_instructions,
)
from jarvis.utils.errors import JarvisError
from jarvis.utils.logging import get_logger

if TYPE_CHECKING:
    from pathlib import Path

    from jarvis.config import Settings

ChatMessage = dict[str, str]
"""``{"role": ..., "content": ...}``, what the native binding accepts."""

IMAGE_PLACEHOLDER: Final = "[image omitted: the local model cannot see images]"

_FINISH_REASONS: Final[Mapping[str, StopReason]] = {
    "end_of_text": StopReason.END_TURN,
    "stop_string": StopReason.END_TURN,
    "max_tokens": StopReason.MAX_TOKENS,
    "context_full": StopReason.MAX_TOKENS,
}

log = get_logger(__name__)


class NativeGenerationStream(Protocol):
    """Shape of ``jarvis_inference.GenerationStream``."""

    def next_token(self) -> str | None:
        """Block (GIL released) until the next piece; ``None`` at the end."""

    def cancel(self) -> None:
        """Stop generating after the current token."""

    def finished(self) -> bool:
        """True once the native producer has returned."""

    def finish_reason(self) -> str | None:
        """Why generation ended, or ``None`` while it runs."""

    def generated_tokens(self) -> int:
        """Tokens generated, once finished."""


class NativeLlama(Protocol):
    """Shape of ``jarvis_inference.LlamaGenerator``."""

    def format_prompt(self, messages: Sequence[Mapping[str, str]]) -> str:
        """Render ``messages`` with the model's chat template."""

    def count_tokens(self, text: str) -> int:
        """Tokens ``text`` occupies in the context."""

    def context_size(self) -> int:
        """The context window the model was loaded with."""

    def generate_from_prompt(
        self,
        prompt: str,
        *,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 40,
        stop: Sequence[str] = (),
        seed: int | None = None,
    ) -> NativeGenerationStream:
        """Start generating on a native thread and return the stream."""


LlamaFactory = Callable[[], NativeLlama]
"""Loads the GGUF on demand; the real one takes ~1 s for a 0.5B model."""


@dataclass(frozen=True, slots=True)
class SamplingOptions:
    """Per-request sampling knobs passed through to llama.cpp."""

    max_tokens: int = 512
    temperature: float = 0.4
    top_p: float = 0.9
    top_k: int = 40


class LocalLlamaClient(LLMClient):
    """``LLMClient`` over the in-process llama.cpp wrapper, fully offline."""

    def __init__(
        self, factory: LlamaFactory, *, model_name: str, sampling: SamplingOptions | None = None
    ) -> None:
        self._factory = factory
        self._model_name = model_name
        self._sampling = sampling or SamplingOptions()
        self._llama: NativeLlama | None = None
        self._load_lock = asyncio.Lock()

    @classmethod
    def from_settings(cls, settings: Settings) -> LocalLlamaClient:
        """Build the real client from ``Settings``.

        Raises:
            ConfigError: If ``jarvis_inference`` is unavailable or the GGUF
                file is missing.
        """
        path = require_model_file(settings.local_llm_model, setting="JARVIS_LOCAL_LLM_MODEL")
        factory = native_llama_factory(
            path, context_size=settings.local_llm_context_size, threads=settings.local_llm_threads
        )
        sampling = SamplingOptions(
            max_tokens=settings.local_llm_max_tokens, temperature=settings.local_llm_temperature
        )
        return cls(factory, model_name=path.name, sampling=sampling)

    @property
    def model(self) -> str:
        """The GGUF file name, which identifies model, size and quantisation."""
        return self._model_name

    async def warm_up(self) -> None:
        """Load the model now so the first turn is not slowed by it."""
        await self._get_llama()

    async def stream(
        self, messages: Sequence[Message], tools: Sequence[ToolDefinition] = ()
    ) -> AsyncIterator[LLMEvent]:
        """Stream one turn; see :meth:`LLMClient.stream`.

        Raises:
            LLMError: If the model cannot be loaded, the conversation does not
                fit the context window, or generation fails.
        """
        llama = await self._get_llama()
        prompt, input_tokens = await asyncio.to_thread(
            _render_prompt, llama, to_chat_messages(messages, tools)
        )
        native = await asyncio.to_thread(self._start, llama, prompt)
        turn = _Turn(ToolCallParser(tools) if tools else None, input_tokens)
        try:
            while not turn.has_call and (piece := await _next_piece(native)) is not None:
                for event in turn.accept(piece):
                    yield event
            for event in turn.finish(native):
                yield event
        finally:
            # Idempotent and non-blocking: also covers early exit by the
            # consumer (aclose, task cancellation on barge-in).
            native.cancel()

    async def _get_llama(self) -> NativeLlama:
        async with self._load_lock:
            if self._llama is None:
                started = time.perf_counter()
                self._llama = await asyncio.to_thread(self._load)
                elapsed_ms = round((time.perf_counter() - started) * 1000)
                log.info("llm.model_loaded", provider="llama.cpp", elapsed_ms=elapsed_ms)
            return self._llama

    def _load(self) -> NativeLlama:
        try:
            return self._factory()
        except JarvisError:
            raise
        except Exception as error:
            raise LLMError(f"could not load the local model {self._model_name}: {error}") from error

    def _start(self, llama: NativeLlama, prompt: str) -> NativeGenerationStream:
        options = self._sampling
        return llama.generate_from_prompt(
            prompt,
            max_tokens=options.max_tokens,
            temperature=options.temperature,
            top_p=options.top_p,
            top_k=options.top_k,
        )


async def _next_piece(native: NativeGenerationStream) -> str | None:
    try:
        return await asyncio.to_thread(native.next_token)
    except Exception as error:
        raise LLMError(f"local generation failed: {error}") from error


def _render_prompt(llama: NativeLlama, messages: list[ChatMessage]) -> tuple[str, int]:
    prompt = llama.format_prompt(messages)
    tokens = llama.count_tokens(prompt)
    if tokens >= llama.context_size():
        raise LLMError(
            f"the conversation ({tokens} tokens) does not fit the local model's context of "
            f"{llama.context_size()} tokens; raise JARVIS_LOCAL_LLM_CONTEXT_SIZE or start over"
        )
    return prompt, tokens


class _Turn:
    """Folds native pieces into LLM events for one request."""

    def __init__(self, parser: ToolCallParser | None, input_tokens: int) -> None:
        self._parser = parser
        self._input_tokens = input_tokens
        self._text_parts: list[str] = []
        self._pieces = 0

    @property
    def has_call(self) -> bool:
        """True once a validated tool call closed; generation can stop."""
        return self._parser is not None and self._parser.call is not None

    def accept(self, piece: str) -> list[LLMEvent]:
        """Events for one generated piece (possibly none while a call is held)."""
        self._pieces += 1
        return self._text(self._parser.push(piece) if self._parser else piece)

    def finish(self, native: NativeGenerationStream) -> list[LLMEvent]:
        """Trailing text, the tool call if any, and ``Done``."""
        events = self._text(self._parser.finish() if self._parser else "")
        parsed = self._parser.call if self._parser else None
        call = _to_tool_call(parsed) if parsed else None
        if call is not None:
            events.append(ToolCallRequested(call))
        events.append(self._done(native, call))
        return events

    def _text(self, text: str) -> list[LLMEvent]:
        if not text:
            return []
        self._text_parts.append(text)
        return [TextDelta(text)]

    def _done(self, native: NativeGenerationStream, call: ToolCall | None) -> Done:
        reason = native.finish_reason() if native.finished() else None
        stop_reason = (
            StopReason.TOOL_USE if call else _FINISH_REASONS.get(reason or "", StopReason.OTHER)
        )
        output = native.generated_tokens() if native.finished() and not call else self._pieces
        usage = Usage(input_tokens=self._input_tokens, output_tokens=output)
        log.info(
            "llm.turn_done",
            provider="llama.cpp",
            stop_reason=stop_reason.value,
            native_finish=reason,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            tool_calls=int(call is not None),
        )
        calls = (call,) if call else ()
        message = AssistantMessage(text="".join(self._text_parts), tool_calls=calls)
        return Done(stop_reason=stop_reason, usage=usage, message=message)


def _to_tool_call(parsed: ParsedToolCall) -> ToolCall:
    return ToolCall(id=f"call_{uuid.uuid4().hex}", name=parsed.name, arguments=parsed.arguments)


def to_chat_messages(
    messages: Sequence[Message], tools: Sequence[ToolDefinition] = ()
) -> list[ChatMessage]:
    """Flatten the neutral history into role/content dicts for the chat template.

    All system text (plus the tool protocol, when tools are offered) becomes
    one leading system message, because most chat templates accept a system
    message only in first position.
    """
    system_parts = [message.text for message in messages if isinstance(message, SystemMessage)]
    if tools:
        system_parts.append(render_tool_instructions(tools))
    converted = [
        _to_chat_message(message) for message in messages if not isinstance(message, SystemMessage)
    ]
    if system_parts:
        converted.insert(0, {"role": "system", "content": "\n\n".join(system_parts)})
    return converted


def _to_chat_message(message: Message) -> ChatMessage:
    if isinstance(message, UserMessage):
        if not message.images:
            return {"role": "user", "content": message.text}
        log.warning("llm.local.images_dropped", count=len(message.images))
        return {"role": "user", "content": f"{message.text}\n{IMAGE_PLACEHOLDER}"}
    if isinstance(message, AssistantMessage):
        calls = [render_tool_call(call.name, call.arguments) for call in message.tool_calls]
        content = "\n".join(part for part in (message.text, *calls) if part)
        return {"role": "assistant", "content": content}
    if isinstance(message, ToolResultMessage):
        return {"role": "user", "content": "\n".join(map(_describe_result, message.results))}
    raise TypeError(f"unsupported message type: {type(message).__name__}")


def _describe_result(result: ToolResult) -> str:
    outcome = "failed" if result.is_error else "returned"
    return f"Tool {result.name} {outcome}: {result.content}"


def native_llama_factory(model_path: Path, *, context_size: int, threads: int) -> LlamaFactory:
    """The real ``LlamaFactory`` over ``jarvis_inference.LlamaGenerator``.

    Raises:
        ConfigError: If the native module is unavailable (checked now, so a
            missing build fails at startup rather than on the first turn).
    """
    module = load_inference_module()

    def factory() -> NativeLlama:
        generator = module.LlamaGenerator(model_path, n_ctx=context_size, n_threads=threads)
        return cast(NativeLlama, generator)

    return factory


# Design notes:
# - Streaming uses one asyncio.to_thread hop per next_token() instead of a
#   dedicated reader thread feeding an asyncio.Queue. The native stream
#   already *is* a producer thread plus a queue; a second Python-side pair
#   would duplicate buffering and need its own shutdown protocol. A thread
#   hop costs tens of microseconds against 10-50 ms per token on a CPU, and
#   cancellation stays trivial: the awaiting coroutine is cancelled, the
#   finally block raises the native CancellationToken, and the worker's
#   blocked next_token() returns within one token.
# - Prompt rendering and token counting happen up front so an over-long
#   history fails with a remedy instead of a native "context_full" stop
#   mid-sentence; the count doubles as Usage.input_tokens.
# - Tool results are replayed as user messages and past calls as the JSON
#   the protocol asked for. A "tool" role exists in some templates but not
#   in all; plain user/assistant turns work with every GGUF chat model.
# - Generation stops as soon as a tool call validates (the protocol allows
#   one per turn), which saves the tokens a small model tends to append.
#   Output tokens are then the pieces received, because the native count is
#   only final once the producer has finished.
# - Images are replaced by a placeholder rather than rejected: a history
#   that contains an earlier screenshot must stay usable after switching to
#   LOCAL mode, and the model is told what it is missing.
# - No Python lock around generation: LlamaGenerator serialises generate()
#   with its own mutex, so a second concurrent stream waits natively.
