"""Scripted stand-ins for ``jarvis_inference.LlamaGenerator`` and ``GenerationStream``."""

from __future__ import annotations

import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any


@dataclass
class FakeGenerationStream:
    """Returns scripted pieces; can fail, or block until cancelled."""

    pieces: list[str]
    reason: str = "end_of_text"
    failure: Exception | None = None
    block_after: int | None = None
    pulled: int = 0
    cancels: int = 0
    _cancelled: threading.Event = field(default_factory=threading.Event)

    def next_token(self) -> str | None:
        if self.block_after is not None and self.pulled >= self.block_after:
            # Like the native stream: blocks until a token arrives or cancel().
            self._cancelled.wait(timeout=5)
            return None
        if self.pulled < len(self.pieces):
            self.pulled += 1
            return self.pieces[self.pulled - 1]
        if self.failure is not None:
            raise self.failure
        return None

    def cancel(self) -> None:
        self.cancels += 1
        self._cancelled.set()

    def finished(self) -> bool:
        return self.pulled >= len(self.pieces) or self._cancelled.is_set()

    def finish_reason(self) -> str | None:
        if not self.finished():
            return None
        return "cancelled" if self._cancelled.is_set() else self.reason

    def generated_tokens(self) -> int:
        return self.pulled if self.finished() else 0


@dataclass
class FakeLlama:
    """Records prompts and generation options; hands out scripted streams."""

    streams: list[FakeGenerationStream] = field(default_factory=list)
    context: int = 4096
    prompts: list[list[dict[str, str]]] = field(default_factory=list)
    options: list[dict[str, Any]] = field(default_factory=list)

    def format_prompt(self, messages: Sequence[Mapping[str, str]]) -> str:
        self.prompts.append([dict(message) for message in messages])
        return "|".join(f"{m['role']}:{m['content']}" for m in messages)

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def context_size(self) -> int:
        return self.context

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
    ) -> FakeGenerationStream:
        self.options.append(
            {
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
            }
        )
        return self.streams.pop(0)
