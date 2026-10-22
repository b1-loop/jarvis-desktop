"""Scripted input, recording output and a fake agent for the orchestrator tests."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from jarvis.agent.chaining import TurnReport
from jarvis.agent.loop import SentenceSink
from jarvis.utils.errors import JarvisError


@dataclass
class ScriptedInput:
    """Returns the scripted lines in order, then ``None`` (end of input)."""

    lines: list[str | None] = field(default_factory=list)
    prompts: list[str] = field(default_factory=list)

    async def read_line(self, prompt: str) -> str | None:
        self.prompts.append(prompt)
        return self.lines.pop(0) if self.lines else None


@dataclass
class RecordingOutput:
    said: list[str] = field(default_factory=list)
    notices: list[str] = field(default_factory=list)
    flushes: int = 0

    async def say(self, sentence: str) -> None:
        self.said.append(sentence)

    async def notice(self, text: str) -> None:
        self.notices.append(text)

    async def flush(self) -> bool:
        self.flushes += 1
        return True


@dataclass
class FakeAgent:
    """Answers every request with ``replies``; can block or fail on demand."""

    replies: list[str] = field(default_factory=lambda: ["Certainly, sir."])
    block: bool = False
    error: JarvisError | None = None
    requests: list[str] = field(default_factory=list)
    started: asyncio.Event = field(default_factory=asyncio.Event)

    async def respond(self, user_text: str, sink: SentenceSink) -> TurnReport:
        self.requests.append(user_text)
        self.started.set()
        if self.error is not None:
            raise self.error
        for reply in self.replies:
            await sink(reply)
        if self.block:
            await asyncio.Event().wait()
        return TurnReport(user_text=user_text, reply=" ".join(self.replies))
