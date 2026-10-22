"""Multi-step task chaining: running the tools of one turn safely and fast.

The agent loop chains naturally: the model asks for tools, sees their
results and asks for more until the task is done. What this module adds is
what keeps that robust when the model is enthusiastic or confused:

- a per-turn **tool-call budget**, so a model stuck in a loop cannot run
  fifty tools for one request;
- **parallel execution** of the independent calls the model requests in
  one message (``asyncio.gather``, bounded by a semaphore), while
  confirmation prompts are still asked one at a time;
- **de-duplication** of identical calls within a turn, so "send the invite"
  requested twice sends one invite;
- a :class:`TurnReport` (tools used, durations, failures) for the dashboard.

Every call still goes through ``ConfirmationPolicy.execute_with_policy``;
nothing here decides whether to ask the user.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Any, Final

from pydantic import JsonValue

from jarvis.agent.confirmation import ConfirmationPolicy
from jarvis.agent.registry import ToolRegistry, UnknownToolError
from jarvis.agent.tool import ConfirmationAsker, ToolContext, ToolResult
from jarvis.llm.base import ToolCall, ToolResultMessage
from jarvis.llm.base import ToolResult as LLMToolResult
from jarvis.utils.logging import get_logger
from jarvis.utils.timing import Clock, SystemClock

log = get_logger(__name__)

DEFAULT_MAX_TOOL_CALLS: Final = 16
"""Tool executions allowed per user request; generous for real chains, fatal for loops."""

DEFAULT_MAX_PARALLEL_TOOLS: Final = 4
"""Calls from one model message that may run at the same time."""

MAX_RESULT_CHARS: Final = 4_000
"""Cap on the text one tool result feeds back to the model; results cost tokens."""

OVER_BUDGET_DETAIL: Final = (
    "Not run: the tool-call budget for this request is used up. "
    "Tell the user what was done and what is left."
)
_MS_PER_SECOND: Final = 1000.0
_TRUNCATION_MARK: Final = " …[truncated]"


class CallOutcome(StrEnum):
    """What happened to one requested tool call."""

    EXECUTED = "executed"
    DUPLICATE = "duplicate"
    OVER_BUDGET = "over_budget"
    UNKNOWN_TOOL = "unknown_tool"


@dataclass(frozen=True, slots=True)
class ToolCallRecord:
    """One requested call as the dashboard shows it.

    Attributes:
        round: 1-based tool round of the turn the call was requested in.
        name: Tool name the model asked for.
        arguments: Arguments exactly as the model sent them.
        outcome: Whether it ran, was answered from an identical earlier
            call, or was refused.
        ok: Whether the result counts as success.
        duration_ms: Wall time including any confirmation prompt; ``0`` for
            calls that did not run.
        spoken: The tool's one-sentence result, used for fallback summaries.
        error: Failure detail when ``ok`` is False.
    """

    round: int
    name: str
    arguments: dict[str, Any]
    outcome: CallOutcome
    ok: bool
    duration_ms: float = 0.0
    spoken: str = ""
    error: str | None = None


@dataclass(frozen=True, slots=True)
class TurnReport:
    """Everything that happened in one user request, for the dashboard and memory.

    Attributes:
        user_text: What the user asked.
        reply: Everything Jarvis said, sentences joined with spaces.
        calls: Every requested tool call in request order.
        llm_rounds: Model requests made, including a summary request.
        duration_ms: Wall time of the whole turn.
        round_limit_hit: The model still wanted tools when
            ``max_tool_rounds`` was reached.
        call_budget_hit: At least one call was refused by the call budget.
        error: The LLM failure the turn ended with, if any.
    """

    user_text: str
    reply: str
    calls: tuple[ToolCallRecord, ...] = ()
    llm_rounds: int = 0
    duration_ms: float = 0.0
    round_limit_hit: bool = False
    call_budget_hit: bool = False
    error: str | None = None

    @property
    def tools_used(self) -> tuple[str, ...]:
        """Names of the tools that actually ran, first use order, no repeats."""
        executed = (call.name for call in self.calls if call.outcome is CallOutcome.EXECUTED)
        return tuple(dict.fromkeys(executed))

    @property
    def failures(self) -> tuple[ToolCallRecord, ...]:
        """Calls that did not succeed, for whatever reason."""
        return tuple(call for call in self.calls if not call.ok)

    @property
    def tool_ms(self) -> float:
        """Total time spent inside tools (and their confirmations)."""
        return sum(call.duration_ms for call in self.calls)

    def memory_payload(self) -> list[dict[str, JsonValue]]:
        """The calls as JSON objects for ``MemoryManager.record_turn``."""
        return [
            {"name": c.name, "arguments": c.arguments, "outcome": c.outcome.value, "ok": c.ok}
            for c in self.calls
        ]


def call_key(call: ToolCall) -> str:
    """Identity of a call for de-duplication: name plus canonical arguments."""
    return f"{call.name}:{json.dumps(call.arguments, sort_keys=True, default=str)}"


def to_llm_result(call: ToolCall, result: ToolResult) -> LLMToolResult:
    """Render a tool's result as the text the model reads next."""
    content = result.detail or result.spoken
    if result.data:
        content += "\nData: " + json.dumps(result.data, default=str, sort_keys=True)
    if len(content) > MAX_RESULT_CHARS:
        content = content[: MAX_RESULT_CHARS - len(_TRUNCATION_MARK)] + _TRUNCATION_MARK
    return LLMToolResult(
        tool_call_id=call.id, name=call.name, content=content, is_error=not result.ok
    )


def serialized(ask: ConfirmationAsker) -> ConfirmationAsker:
    """Wrap ``ask`` so concurrent tool calls put their questions one at a time."""
    lock = asyncio.Lock()

    async def ask_one_at_a_time(prompt: str) -> bool:
        async with lock:
            return await ask(prompt)

    return ask_one_at_a_time


@dataclass
class _Executed:
    result: ToolResult
    duration_ms: float
    outcome: CallOutcome = CallOutcome.EXECUTED


@dataclass
class _RoundState:
    """Bookkeeping for one model message's calls."""

    fresh: dict[str, ToolCall] = field(default_factory=dict)
    answered: set[str] = field(default_factory=set)


class ToolChain:
    """Runs the tool calls of one turn: budget, parallelism, de-duplication.

    Create one per user request; the budget and the de-duplication cache
    are per turn on purpose, so "what time is it?" asked twice in one
    session is answered twice.

    Args:
        registry: Where tool names are resolved.
        policy: The confirmation policy every call goes through.
        context: Services for the tools; its ``ask_confirmation`` is
            wrapped so parallel calls never prompt at the same time.
        clock: Time source for durations.
        max_calls: Tool executions allowed in this turn.
        max_parallel: Calls from one message that may run concurrently.
    """

    def __init__(
        self,
        registry: ToolRegistry,
        policy: ConfirmationPolicy,
        context: ToolContext,
        *,
        clock: Clock | None = None,
        max_calls: int = DEFAULT_MAX_TOOL_CALLS,
        max_parallel: int = DEFAULT_MAX_PARALLEL_TOOLS,
    ) -> None:
        self._registry = registry
        self._policy = policy
        self._context = replace(context, ask_confirmation=serialized(context.ask_confirmation))
        self._clock = clock or SystemClock()
        self._max_calls = max_calls
        self._semaphore = asyncio.Semaphore(max_parallel)
        self._cache: dict[str, _Executed] = {}
        self._records: list[ToolCallRecord] = []

    @property
    def records(self) -> tuple[ToolCallRecord, ...]:
        """Every call requested so far this turn."""
        return tuple(self._records)

    @property
    def call_budget_hit(self) -> bool:
        """Whether any call was refused because the budget ran out."""
        return any(record.outcome is CallOutcome.OVER_BUDGET for record in self._records)

    async def run_round(self, calls: Sequence[ToolCall], *, round_index: int) -> ToolResultMessage:
        """Execute one message's calls and return their results in request order."""
        state = _RoundState()
        for call in calls:
            key = call_key(call)
            if key not in self._cache and key not in state.fresh and self._has_budget(state):
                state.fresh[key] = call
        results = await asyncio.gather(*(self._execute(call) for call in state.fresh.values()))
        self._cache.update(zip(state.fresh, results, strict=True))
        answers = tuple(self._answer(call, round_index, state) for call in calls)
        return ToolResultMessage(results=answers)

    def decline(
        self, calls: Sequence[ToolCall], *, round_index: int, detail: str
    ) -> ToolResultMessage:
        """Answer calls that will not run (round limit) so the history stays valid."""
        results = []
        for call in calls:
            result = ToolResult.failure("I stopped before running that.", detail=detail)
            self._record(call, round_index, _Executed(result, 0.0, CallOutcome.OVER_BUDGET))
            results.append(to_llm_result(call, result))
        return ToolResultMessage(results=tuple(results))

    def _has_budget(self, state: _RoundState) -> bool:
        return len(self._cache) + len(state.fresh) < self._max_calls

    async def _execute(self, call: ToolCall) -> _Executed:
        async with self._semaphore:
            started = self._clock.monotonic()
            try:
                tool = self._registry.get(call.name)
            except UnknownToolError as error:
                log.warning("agent.tool.unknown", tool=call.name)
                failure = ToolResult.failure(
                    f"I don't have a tool called {call.name}.", detail=str(error)
                )
                return _Executed(failure, 0.0, CallOutcome.UNKNOWN_TOOL)
            result = await self._policy.execute_with_policy(tool, call.arguments, self._context)
            return _Executed(result, (self._clock.monotonic() - started) * _MS_PER_SECOND)

    def _answer(self, call: ToolCall, round_index: int, state: _RoundState) -> LLMToolResult:
        key = call_key(call)
        executed = self._cache.get(key)
        if executed is None:
            log.warning("agent.tool.over_budget", tool=call.name, budget=self._max_calls)
            executed = _Executed(
                ToolResult.failure(
                    "I hit my limit of actions for one request.", detail=OVER_BUDGET_DETAIL
                ),
                0.0,
                CallOutcome.OVER_BUDGET,
            )
        elif key in state.answered or key not in state.fresh:
            log.info("agent.tool.duplicate", tool=call.name)
            executed = replace(executed, duration_ms=0.0, outcome=CallOutcome.DUPLICATE)
        state.answered.add(key)
        self._record(call, round_index, executed)
        return to_llm_result(call, executed.result)

    def _record(self, call: ToolCall, round_index: int, executed: _Executed) -> None:
        result = executed.result
        self._records.append(
            ToolCallRecord(
                round=round_index,
                name=call.name,
                arguments=dict(call.arguments),
                outcome=executed.outcome,
                ok=result.ok,
                duration_ms=executed.duration_ms,
                spoken=result.spoken,
                error=None if result.ok else result.detail,
            )
        )


# Design notes:
# - De-duplication is keyed on name plus canonical JSON arguments and lasts
#   for one turn. A duplicate gets the original result back (marked
#   ``duplicate`` in the report) rather than an error, so the model's view
#   stays consistent and it does not retry. Across turns nothing is cached:
#   the world may have changed.
# - Parallel calls share one context whose confirmation callback is behind
#   an ``asyncio.Lock``. Read-only tools run concurrently; a destructive one
#   waits its turn to ask, so two prompts never interleave in the terminal
#   or in speech, and the policy still decides per call.
# - Unknown tools and over-budget calls become ``is_error`` results instead
#   of exceptions: every ``tool_use`` must be answered or the provider
#   rejects the next request, and the model can explain the gap.
# - ToolCallStarted/ToolCallFinished are published by ``Tool.execute``
#   (via the policy) for calls that actually run; refused calls appear only
#   in the report and the log, because no tool started.
# - Durations include confirmation time. For the dashboard "how long did
#   this step take" is the useful number; tool-only time is in the
#   ToolCallFinished event.
