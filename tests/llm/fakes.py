"""Fakes for the Anthropic SDK streamer, built from the SDK's own event objects."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterable
from dataclasses import dataclass, field
from typing import Any

from anthropic.types import (
    InputJSONDelta,
    Message,
    MessageDeltaUsage,
    RawContentBlockDeltaEvent,
    RawContentBlockStartEvent,
    RawContentBlockStopEvent,
    RawMessageDeltaEvent,
    RawMessageStartEvent,
    RawMessageStopEvent,
    RedactedThinkingBlock,
    SignatureDelta,
    TextBlock,
    TextDelta,
    ThinkingBlock,
    ThinkingDelta,
    ToolUseBlock,
    Usage,
)
from anthropic.types.raw_message_delta_event import Delta

Event = Any


def message_start(input_tokens: int = 10, cache_read: int | None = None) -> Event:
    message = Message(
        id="msg_test",
        content=[],
        model="claude-sonnet-5-5",
        role="assistant",
        stop_reason=None,
        stop_sequence=None,
        type="message",
        usage=Usage(input_tokens=input_tokens, output_tokens=1, cache_read_input_tokens=cache_read),
    )
    return RawMessageStartEvent(message=message, type="message_start")


def text_block(index: int, *deltas: str) -> list[Event]:
    events: list[Event] = [
        RawContentBlockStartEvent(
            index=index, content_block=TextBlock(text="", type="text"), type="content_block_start"
        )
    ]
    events.extend(
        RawContentBlockDeltaEvent(
            index=index, delta=TextDelta(text=delta, type="text_delta"), type="content_block_delta"
        )
        for delta in deltas
    )
    events.append(RawContentBlockStopEvent(index=index, type="content_block_stop"))
    return events


def tool_block(index: int, call_id: str, name: str, *json_parts: str) -> list[Event]:
    events: list[Event] = [
        RawContentBlockStartEvent(
            index=index,
            content_block=ToolUseBlock(id=call_id, input={}, name=name, type="tool_use"),
            type="content_block_start",
        )
    ]
    events.extend(
        RawContentBlockDeltaEvent(
            index=index,
            delta=InputJSONDelta(partial_json=part, type="input_json_delta"),
            type="content_block_delta",
        )
        for part in json_parts
    )
    events.append(RawContentBlockStopEvent(index=index, type="content_block_stop"))
    return events


def thinking_block(index: int, thinking: str, signature: str) -> list[Event]:
    return [
        RawContentBlockStartEvent(
            index=index,
            content_block=ThinkingBlock(thinking="", signature="", type="thinking"),
            type="content_block_start",
        ),
        RawContentBlockDeltaEvent(
            index=index,
            delta=ThinkingDelta(thinking=thinking, type="thinking_delta"),
            type="content_block_delta",
        ),
        RawContentBlockDeltaEvent(
            index=index,
            delta=SignatureDelta(signature=signature, type="signature_delta"),
            type="content_block_delta",
        ),
        RawContentBlockStopEvent(index=index, type="content_block_stop"),
    ]


def redacted_thinking_block(index: int, data: str) -> list[Event]:
    return [
        RawContentBlockStartEvent(
            index=index,
            content_block=RedactedThinkingBlock(data=data, type="redacted_thinking"),
            type="content_block_start",
        ),
        RawContentBlockStopEvent(index=index, type="content_block_stop"),
    ]


def message_delta(stop_reason: str, output_tokens: int = 7) -> Event:
    return RawMessageDeltaEvent(
        delta=Delta(stop_reason=stop_reason, stop_sequence=None),  # type: ignore[arg-type]
        usage=MessageDeltaUsage(output_tokens=output_tokens),
        type="message_delta",
    )


def message_stop() -> Event:
    return RawMessageStopEvent(type="message_stop")


def text_turn(*deltas: str, stop_reason: str = "end_turn") -> list[Event]:
    """A complete single-text-block turn."""
    return [message_start(), *text_block(0, *deltas), message_delta(stop_reason), message_stop()]


class FakeStream:
    def __init__(self, events: Iterable[Event]) -> None:
        self._events = list(events)

    def __aiter__(self) -> AsyncIterator[Event]:
        return self._iterate()

    async def _iterate(self) -> AsyncIterator[Event]:
        for event in self._events:
            yield event


class FakeStreamManager:
    """What ``messages.stream(...)`` returns: entering it performs the request."""

    def __init__(self, outcome: list[Event] | Exception) -> None:
        self._outcome = outcome

    async def __aenter__(self) -> FakeStream:
        if isinstance(self._outcome, Exception):
            raise self._outcome
        return FakeStream(self._outcome)

    async def __aexit__(self, *exc_info: object) -> None:
        return None


@dataclass
class FakeStreamer:
    """Scripted stand-in for ``AsyncAnthropic.messages``; one outcome per request."""

    outcomes: list[list[Event] | Exception] = field(default_factory=list)
    requests: list[dict[str, Any]] = field(default_factory=list)

    def stream(self, **params: Any) -> FakeStreamManager:
        self.requests.append(params)
        return FakeStreamManager(self.outcomes.pop(0))
