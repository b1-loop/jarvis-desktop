"""Scripted LLM, memory and chain fakes for the agent-loop tests."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from pydantic import Field

from jarvis.agent.confirmation import ConfirmationPolicy
from jarvis.agent.loop import Agent, AgentLimits
from jarvis.agent.persona import Persona
from jarvis.agent.registry import ToolRegistry
from jarvis.agent.tool import Tool, ToolCategory, ToolContext, ToolInput, ToolResult
from jarvis.agent.tool_selection import ToolSelector
from jarvis.config import Settings
from jarvis.core.events import EventBus
from jarvis.llm.base import (
    AssistantMessage,
    Done,
    LLMClient,
    LLMEvent,
    Message,
    StopReason,
    TextDelta,
    ToolCall,
    ToolCallRequested,
    ToolDefinition,
    Usage,
)
from jarvis.memory.errors import MemoryStoreError
from tests.agent.fakes import FakeConfirmation, make_context


class Block:
    """Marker in a script: the stream waits here until cancelled."""


Step = LLMEvent | Exception | Block
Round = list[Step] | Exception


def text_round(*deltas: str) -> list[Step]:
    """A reply that only talks; ``deltas`` arrive as separate fragments."""
    text = "".join(deltas)
    return [
        *(TextDelta(delta) for delta in deltas),
        Done(StopReason.END_TURN, Usage(), AssistantMessage(text=text)),
    ]


def tool_round(*calls: ToolCall, text: str = "") -> list[Step]:
    """A reply that (optionally says something and) asks for ``calls``."""
    steps: list[Step] = [TextDelta(text)] if text else []
    steps.extend(ToolCallRequested(call) for call in calls)
    steps.append(Done(StopReason.TOOL_USE, Usage(), AssistantMessage(text, tuple(calls))))
    return steps


def call(call_id: str, tool_name: str, /, **arguments: Any) -> ToolCall:
    return ToolCall(id=call_id, name=tool_name, arguments=arguments)


@dataclass
class ScriptedLLM(LLMClient):
    """Replays one scripted round per ``stream`` call and records every request."""

    rounds: list[Round] = field(default_factory=list)
    requests: list[list[Message]] = field(default_factory=list)
    tools_seen: list[list[ToolDefinition]] = field(default_factory=list)
    started: asyncio.Event = field(default_factory=asyncio.Event)
    closed: int = 0

    @property
    def model(self) -> str:
        return "scripted"

    async def stream(
        self, messages: Sequence[Message], tools: Sequence[ToolDefinition] = ()
    ) -> AsyncIterator[LLMEvent]:
        self.requests.append(list(messages))
        self.tools_seen.append(list(tools))
        script = self.rounds.pop(0)
        if isinstance(script, Exception):
            raise script
        try:
            for step in script:
                yield await self._play(step)
        finally:
            self.closed += 1

    async def _play(self, step: Step) -> LLMEvent:
        if isinstance(step, Block):
            self.started.set()
            await asyncio.Event().wait()
        if isinstance(step, Exception):
            raise step
        assert not isinstance(step, Block)
        return step


@dataclass
class FakeRecollection:
    text: str
    user_name: str | None = None

    def render(self) -> str:
        return self.text


@dataclass
class FakeMemory:
    """Records turns; can be told to fail on recall or on recording."""

    context: str = "User name: Bozhidar"
    user_name: str | None = None
    fail_recall: bool = False
    fail_record: bool = False
    recalled: list[str] = field(default_factory=list)
    turns: list[tuple[str, str, list[dict[str, Any]]]] = field(default_factory=list)

    async def recall(self, user_text: str) -> FakeRecollection:
        self.recalled.append(user_text)
        if self.fail_recall:
            raise MemoryStoreError("database is locked")
        return FakeRecollection(self.context, self.user_name)

    async def record_turn(
        self, user_text: str, assistant_text: str, tool_calls: Iterable[dict[str, Any]] = ()
    ) -> None:
        if self.fail_record:
            raise MemoryStoreError("disk full")
        self.turns.append((user_text, assistant_text, list(tool_calls)))


# -- fake tools for the "book Anna on Thursday" chain -----------------------


class ContactInput(ToolInput):
    name: str


class LookupContactTool(Tool[ContactInput]):
    name = "lookup_contact"
    description = "Find a contact's email address."
    Input = ContactInput

    async def run(self, params: ContactInput, context: ToolContext) -> ToolResult:
        return ToolResult.success(
            f"{params.name} is anna@example.com.", data={"email": "anna@example.com"}
        )


class SlotInput(ToolInput):
    day: str


class FreeSlotTool(Tool[SlotInput]):
    name = "find_free_slot"
    description = "Find a free hour on a day."
    Input = SlotInput

    async def run(self, params: SlotInput, context: ToolContext) -> ToolResult:
        return ToolResult.success(f"{params.day} at 10:00 is free.", data={"start": "10:00"})


class EventInput(ToolInput):
    title: str
    day: str
    start: str


class CreateEventTool(Tool[EventInput]):
    name = "create_event"
    description = "Create a calendar event."
    Input = EventInput
    category = ToolCategory.PRODUCTIVITY
    requires_confirmation = True

    def __init__(self) -> None:
        self.created: list[EventInput] = []

    async def run(self, params: EventInput, context: ToolContext) -> ToolResult:
        self.created.append(params)
        return ToolResult.success(f"Booked {params.title}.", data={"event_id": "evt-1"})


class InviteInput(ToolInput):
    event_id: str
    email: str


class SendInviteTool(Tool[InviteInput]):
    name = "send_invite"
    description = "Email an invitation for an event."
    Input = InviteInput
    category = ToolCategory.COMMUNICATION
    requires_confirmation = True

    def __init__(self) -> None:
        self.sent: list[InviteInput] = []

    async def run(self, params: InviteInput, context: ToolContext) -> ToolResult:
        self.sent.append(params)
        return ToolResult.success(f"Invite sent to {params.email}.")


class NoteInput(ToolInput):
    text: str = Field(min_length=1)


class AddNoteTool(Tool[NoteInput]):
    name = "add_note"
    description = "Add a note."
    Input = NoteInput
    category = ToolCategory.PRODUCTIVITY

    def __init__(self) -> None:
        self.notes: list[str] = []

    async def run(self, params: NoteInput, context: ToolContext) -> ToolResult:
        self.notes.append(params.text)
        return ToolResult.success("Added it to your notes.")


class WaitInput(ToolInput):
    label: str


class RendezvousTool(Tool[WaitInput]):
    """Each call waits until ``expected`` calls are running at the same time."""

    name = "rendezvous"
    description = "Waits for its siblings."
    Input = WaitInput

    def __init__(self, expected: int = 2) -> None:
        self.expected = expected
        self.running = 0
        self.peak = 0
        self.all_arrived = asyncio.Event()

    async def run(self, params: WaitInput, context: ToolContext) -> ToolResult:
        self.running += 1
        self.peak = max(self.peak, self.running)
        if self.running >= self.expected:
            self.all_arrived.set()
        await self.all_arrived.wait()
        self.running -= 1
        return ToolResult.success(f"{params.label} done.")


class CountingInput(ToolInput):
    label: str = "x"


class CountingTool(Tool[CountingInput]):
    """Counts runs and the peak number running at once (yields once per run)."""

    name = "counting"
    description = "Counts how often it runs."
    Input = CountingInput

    def __init__(self) -> None:
        self.runs = 0
        self.running = 0
        self.peak = 0

    async def run(self, params: CountingInput, context: ToolContext) -> ToolResult:
        self.runs += 1
        self.running += 1
        self.peak = max(self.peak, self.running)
        await asyncio.sleep(0)
        self.running -= 1
        return ToolResult.success(f"Counted {params.label}.")


def make_agent(
    settings: Settings,
    llm: ScriptedLLM,
    *tools: Tool[Any],
    confirmation: FakeConfirmation | None = None,
    bus: EventBus | None = None,
    memory: FakeMemory | None = None,
    limits: AgentLimits | None = None,
    cloud_provider: bool = False,
    selector: ToolSelector | None = None,
) -> Agent:
    """An ``Agent`` over fakes, with ``tools`` registered by hand."""
    registry = ToolRegistry()
    for tool in tools:
        registry.register(tool)
    return Agent(
        llm=llm,
        registry=registry,
        policy=ConfirmationPolicy.from_settings(settings),
        persona=Persona.from_settings(settings),
        context=make_context(settings, confirmation=confirmation, bus=bus),
        memory=memory,
        limits=limits,
        cloud_provider=cloud_provider,
        selector=selector,
    )


@dataclass
class SentenceCollector:
    """A sentence sink that remembers what it was given."""

    sentences: list[str] = field(default_factory=list)

    async def __call__(self, sentence: str) -> None:
        self.sentences.append(sentence)
