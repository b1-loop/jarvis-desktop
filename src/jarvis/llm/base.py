"""``LLMClient`` interface and the provider-neutral message, tool and event types.

The agent loop (step 6) speaks only these types. Each provider converts them
to and from its wire format, so switching between Claude, Ollama and the
local llama.cpp wrapper is a configuration change and the agent never
learns which one answered. Streaming is the only mode: text arrives as
deltas so TTS can start on the first sentence, and tool calls arrive as
soon as their arguments are complete.
"""

from __future__ import annotations

import base64
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel

from jarvis.utils.errors import JarvisError


class LLMError(JarvisError):
    """The provider failed after retries or returned something unusable.

    The message always says which provider failed and, where known, what to
    do about it (pull a model, check a key), because the orchestrator reads
    it aloud as the fallback answer.
    """


ImageMediaType = Literal["image/png", "image/jpeg", "image/webp", "image/gif"]
"""Image formats every supported vision model accepts."""


@dataclass(frozen=True, slots=True)
class ImageContent:
    """An image attached to a user message, for screen understanding.

    Attributes:
        data: Raw encoded image bytes.
        media_type: MIME type matching ``data``.
    """

    data: bytes
    media_type: ImageMediaType

    def as_base64(self) -> str:
        """Standard base64 text, which is what both providers want on the wire."""
        return base64.standard_b64encode(self.data).decode("ascii")


@dataclass(frozen=True, slots=True)
class SystemMessage:
    """Instructions that frame the whole conversation (persona, rules)."""

    text: str


@dataclass(frozen=True, slots=True)
class UserMessage:
    """What the user said, optionally with images."""

    text: str
    images: tuple[ImageContent, ...] = ()


@dataclass(frozen=True, slots=True)
class ToolCall:
    """A request from the model to run one tool.

    Attributes:
        id: Provider-issued identifier the matching result must quote.
        name: Tool name exactly as declared in :class:`ToolDefinition`.
        arguments: Parsed JSON arguments; validated by the tool's pydantic
            schema before anything runs.
    """

    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class AssistantMessage:
    """One completed assistant turn, as replayed in later requests.

    Attributes:
        text: Everything the model said in this turn.
        tool_calls: Tools it asked for, in order.
        provider_state: Opaque content blocks the producing provider needs
            replayed verbatim (Claude's signed thinking blocks). Clients set
            it on the message they return in :class:`Done`; other providers
            ignore it and rebuild the turn from ``text`` and ``tool_calls``.
    """

    text: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    provider_state: tuple[Mapping[str, Any], ...] = ()


@dataclass(frozen=True, slots=True)
class ToolResult:
    """The outcome of one tool call, fed back to the model.

    Attributes:
        tool_call_id: The :attr:`ToolCall.id` this answers.
        name: Tool name; providers without call ids match on it.
        content: Text for the model; keep it short, it costs tokens.
        is_error: True when the tool failed, so the model can recover.
    """

    tool_call_id: str
    name: str
    content: str
    is_error: bool = False


@dataclass(frozen=True, slots=True)
class ToolResultMessage:
    """Results for every tool call of the preceding assistant turn."""

    results: tuple[ToolResult, ...]


Message = SystemMessage | UserMessage | AssistantMessage | ToolResultMessage
"""Anything that can appear in a conversation history."""


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    """A tool offered to the model.

    Attributes:
        name: Identifier the model uses to call it.
        description: What it does and when to use it; the model reads this.
        input_schema: JSON Schema for the arguments.
    """

    name: str
    description: str
    input_schema: Mapping[str, Any]

    @classmethod
    def from_model(cls, name: str, description: str, model: type[BaseModel]) -> ToolDefinition:
        """Build the schema from the pydantic model a tool validates with.

        One schema serves both validation and the model-facing definition,
        so the two can never disagree.
        """
        return cls(name=name, description=description, input_schema=model.model_json_schema())


class StopReason(StrEnum):
    """Why the model stopped generating, normalised across providers."""

    END_TURN = "end_turn"
    TOOL_USE = "tool_use"
    MAX_TOKENS = "max_tokens"
    REFUSAL = "refusal"
    OTHER = "other"


@dataclass(frozen=True, slots=True)
class Usage:
    """Token accounting for one request, for the dashboard and cost tracking."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        """Billable tokens in and out, ignoring the cache discount."""
        return self.input_tokens + self.output_tokens


@dataclass(frozen=True, slots=True)
class TextDelta:
    """A fragment of the answer text, in order."""

    text: str


@dataclass(frozen=True, slots=True)
class ToolCallRequested:
    """The model wants a tool run; arguments are complete and parsed."""

    call: ToolCall


@dataclass(frozen=True, slots=True)
class Done:
    """End of the turn.

    Attributes:
        stop_reason: Why generation ended; ``TOOL_USE`` means run the tools
            and call ``stream`` again with their results.
        usage: Token counts for this request.
        message: The assistant turn to append to the history before the
            next request.
    """

    stop_reason: StopReason
    usage: Usage
    message: AssistantMessage


LLMEvent = TextDelta | ToolCallRequested | Done
"""Everything ``LLMClient.stream`` can yield; ``Done`` is always last."""


class LLMClient(ABC):
    """Streaming chat with native tool calling, independent of the provider."""

    @property
    @abstractmethod
    def model(self) -> str:
        """Identifier of the model answering, for logs and the dashboard."""

    @abstractmethod
    def stream(
        self, messages: Sequence[Message], tools: Sequence[ToolDefinition] = ()
    ) -> AsyncIterator[LLMEvent]:
        """Send the conversation and stream the model's turn.

        Args:
            messages: Full history, oldest first. System messages may appear
                anywhere; providers hoist them as their API requires.
            tools: Tools the model may call this turn.

        Yields:
            Text deltas and tool-call requests as they arrive, then exactly
            one :class:`Done`.

        Raises:
            LLMError: After retries are exhausted or on an unusable reply.
        """


# Design notes:
# - Messages are frozen dataclasses rather than pydantic models: they are
#   constructed by our own code, never parsed from untrusted input, and the
#   agent loop builds many of them per turn.
# - A single ToolResultMessage carries every result of a turn because the
#   Anthropic API requires all tool_result blocks of one assistant turn in
#   one user message; splitting them across messages silently discourages
#   parallel tool use.
# - provider_state exists only for Claude's thinking blocks: with thinking
#   on, a tool-use turn must be replayed with its signed thinking block or
#   the API rejects the continuation. Every other provider ignores it.
# - StopReason.OTHER covers provider-specific reasons (pause_turn, stop
#   sequences) that the agent treats as "finished, nothing to run".
