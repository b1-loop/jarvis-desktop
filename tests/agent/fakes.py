"""Fake tools and contexts for the agent tests."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from pydantic import Field

from jarvis.agent.tool import Tool, ToolCategory, ToolContext, ToolInput, ToolResult
from jarvis.config import Settings
from jarvis.core.events import EventBus
from jarvis.utils.errors import JarvisError
from tests.fakes import FakeClock


@dataclass
class FakeConfirmation:
    """Scripted answers to ``ask_confirmation``; records every prompt."""

    answers: list[bool] = field(default_factory=lambda: [True])
    prompts: list[str] = field(default_factory=list)

    async def __call__(self, prompt: str) -> bool:
        self.prompts.append(prompt)
        return self.answers.pop(0)


def make_context(
    settings: Settings,
    *,
    bus: EventBus | None = None,
    confirmation: FakeConfirmation | None = None,
    clock: FakeClock | None = None,
) -> ToolContext:
    """A ``ToolContext`` wired entirely with fakes."""
    return ToolContext(
        settings=settings,
        bus=bus or EventBus(),
        ask_confirmation=confirmation or FakeConfirmation(),
        clock=clock or FakeClock(),
    )


class EchoInput(ToolInput):
    text: str = Field(min_length=1)
    repeat: int = Field(default=1, ge=1, le=3)


class EchoTool(Tool[EchoInput]):
    """Repeats its input; the simplest possible well-formed tool."""

    name = "echo"
    description = "Repeat a piece of text."
    Input = EchoInput

    def __init__(self, on_run: Callable[[EchoInput], None] | None = None) -> None:
        self.on_run = on_run

    async def run(self, params: EchoInput, context: ToolContext) -> ToolResult:
        if self.on_run is not None:
            self.on_run(params)
        return ToolResult.success(
            " ".join([params.text] * params.repeat), data={"repeat": params.repeat}
        )


class DestructiveInput(ToolInput):
    target: str = "everything"


class DestructiveTool(Tool[DestructiveInput]):
    """Declares ``requires_confirmation`` like a real side-effecting tool."""

    name = "destroy"
    description = "Pretend to delete things."
    Input = DestructiveInput
    category = ToolCategory.FILES
    requires_confirmation = True

    def __init__(self) -> None:
        self.calls: list[DestructiveInput] = []

    async def run(self, params: DestructiveInput, context: ToolContext) -> ToolResult:
        self.calls.append(params)
        return ToolResult.success(f"Deleted {params.target}.")

    def confirmation_prompt(self, params: DestructiveInput) -> str:
        return f"Delete {params.target}?"


class SendInput(ToolInput):
    to: str = "someone"


class SendTool(Tool[SendInput]):
    """No confirmation flag of its own; only its category can demand one."""

    name = "send"
    description = "Pretend to send a message."
    Input = SendInput
    category = ToolCategory.COMMUNICATION

    async def run(self, params: SendInput, context: ToolContext) -> ToolResult:
        return ToolResult.success(f"Sent to {params.to}.")


class FlakyInput(ToolInput):
    mode: str = "jarvis"


class FlakyTool(Tool[FlakyInput]):
    """Raises on demand so the error conversion paths can be tested."""

    name = "flaky"
    description = "Fails in a configurable way."
    Input = FlakyInput

    async def run(self, params: FlakyInput, context: ToolContext) -> ToolResult:
        if params.mode == "jarvis":
            raise JarvisError("provider unavailable")
        if params.mode == "crash":
            raise ValueError("unexpected")
        return ToolResult.success("fine")


def tool_names(items: list[Any]) -> list[str]:
    return [item.name for item in items]
