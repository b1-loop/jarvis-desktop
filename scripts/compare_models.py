"""Compare Ollama chat models on fixed Swedish requests through the real agent loop.

Usage::

    uv run python scripts/compare_models.py qwen2.5:7b qwen3:8b@off qwen3:8b@on
    uv run python scripts/compare_models.py qwen3:8b --embedding-model none
    uv run python scripts/compare_models.py qwen2.5:7b qwen3:8b --repeat 3

Each argument is an Ollama model tag, optionally with ``@off``, ``@on`` or
``@default`` for Ollama's ``think`` field (``JARVIS_OLLAMA_THINK``; off when
omitted). For every model the script warms the model up with one throwaway
request, then sends the requests below one at a time through the same
:class:`~jarvis.agent.loop.Agent`, tool registry, per-turn tool selection
and persona as the text mode, and checks each answer against ground truth:
the clock, ``psutil`` for RAM and battery, and the todo and reminder store.

Nothing outside a temporary directory changes. Memory (SQLite) lives in a
fresh temp ``data_dir`` per model, the OS layer is the ``StubPlatform``,
every confirmation is declined, and every tool outside :data:`SAFE_TOOLS`
is replaced by a stand-in that refuses to run, so the model still sees the
full, realistic tool list without being able to open, send or change
anything. Runs are sequential, so the Ollama server is shared politely with
a Jarvis that may be running; no model is unloaded.

Per request it prints the tools called, whether the answer was correct,
time to first answer token (what a listener waits for, including any tool
round) and total time, then totals per model.
"""

from __future__ import annotations

import argparse
import asyncio
import re
import statistics
import sys
import tempfile
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Final

import httpx
import psutil
import structlog

from jarvis.agent.chaining import TurnReport
from jarvis.agent.confirmation import ConfirmationPolicy
from jarvis.agent.loop import Agent
from jarvis.agent.persona import Persona
from jarvis.agent.registry import AnyTool, ToolRegistry
from jarvis.agent.tool import ToolContext, ToolResult
from jarvis.agent.tool_embeddings import EmbeddingToolRanker
from jarvis.agent.tool_selection import RelevanceToolSelector, ToolSelector
from jarvis.config import Settings
from jarvis.core.bootstrap import build_tool_selector
from jarvis.core.events import EventBus
from jarvis.llm.base import LLMClient, LLMEvent, Message, TextDelta, ToolDefinition
from jarvis.llm.ollama_client import OllamaClient
from jarvis.memory import InMemoryVectorStore, MemoryManager, SqliteStore
from jarvis.os_integration.stub import StubPlatform

SAFE_TOOLS: Final = frozenset(
    {
        "current_time",
        "system_monitor",
        "battery_status",
        "add_todo",
        "list_todos",
        "complete_todo",
        "add_reminder",
        "list_reminders",
        "add_note",
        "search_notes",
    }
)
"""Read-only tools, plus the ones that only write to the temporary memory database."""

THINK_FLAGS: Final = {"off": False, "on": True, "default": None}
OLLAMA_TIMEOUT: Final = httpx.Timeout(300.0, connect=5.0)
TIME_TOLERANCE: Final = timedelta(minutes=2)
REMINDER_DELAY: Final = timedelta(minutes=5)
REMINDER_TOLERANCE: Final = timedelta(minutes=2)
PERCENT_TOLERANCE: Final = 5.0
GIGABYTE_TOLERANCE: Final = 0.8
BYTES_PER_GB: Final = 1024**3
MILK: Final = ("mjölk", "milk")
WARM_UP_REQUEST: Final = "Hej!"
VOICE: Final = "sv-SE-MattiasNeural"
"""The user's voice; the persona answers in the voice's language (Swedish)."""

_NUMBER: Final = re.compile(r"\d+(?:[.,]\d+)?")
_CLOCK: Final = re.compile(r"\b(\d{1,2})[:.](\d{2})\b")
_PERCENT: Final = re.compile(r"(\d+(?:[.,]\d+)?)\s*(?:%|procent|percent)", re.IGNORECASE)
_AMOUNT: Final = re.compile(r"(\d+(?:[.,]\d+)?)\s*(MB|GB)\b", re.IGNORECASE)
MB_PER_GB: Final = 1024


# -- ground truth -------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Probe:
    """What the checks may consult after a turn: the answer and the memory store."""

    answer: str
    memory: MemoryManager
    asked_at: datetime


Check = Callable[[Probe], Awaitable[bool]]


def numbers_in(text: str) -> list[float]:
    """Every number in ``text``, accepting Swedish decimal commas ("7,9")."""
    return [float(match.replace(",", ".")) for match in _NUMBER.findall(text)]


def mentions_time(text: str, moment: datetime) -> bool:
    """Whether ``text`` names a clock time (HH:MM or HH.MM) within two minutes of ``moment``."""
    for hours, minutes in _CLOCK.findall(text):
        said = moment.replace(hour=int(hours) % 24, minute=int(minutes) % 60, second=0)
        if abs(said - moment) <= TIME_TOLERANCE:
            return True
    return False


async def check_time(probe: Probe) -> bool:
    """The answer names the current clock time."""
    return mentions_time(probe.answer, probe.asked_at)


def percents_in(text: str) -> list[float]:
    """Numbers written as percentages ("64%", "64 %", "64 procent")."""
    return [float(value.replace(",", ".")) for value in _PERCENT.findall(text)]


def amounts_in_gb(text: str) -> list[float]:
    """Memory amounts written in MB or GB, converted to GB."""
    amounts = []
    for value, unit in _AMOUNT.findall(text):
        number = float(value.replace(",", "."))
        amounts.append(number / MB_PER_GB if unit.upper() == "MB" else number)
    return amounts


async def check_ram(probe: Probe) -> bool:
    """The first stated percentage is psutil's; without one, the first stated amount.

    Only the first figure counts, so an answer that mixes up numbers (process
    memory reported as the total) is wrong even if a right number appears later.
    """
    memory = psutil.virtual_memory()
    print(f"    truth: RAM {memory.percent:.0f}% = {memory.used / BYTES_PER_GB:.1f} GB used")
    if percents := percents_in(probe.answer):
        return abs(percents[0] - memory.percent) <= PERCENT_TOLERANCE
    amounts = amounts_in_gb(probe.answer)
    return bool(amounts) and abs(amounts[0] - memory.used / BYTES_PER_GB) <= GIGABYTE_TOLERANCE


async def check_battery(probe: Probe) -> bool:
    """The answer gives the battery charge psutil reports (or says there is none)."""
    battery = psutil.sensors_battery()
    if battery is None:
        return any(word in probe.answer.lower() for word in ("inget batteri", "no battery"))
    print(f"    truth: battery {battery.percent:.0f}%")
    stated = percents_in(probe.answer)
    return bool(stated) and abs(stated[0] - battery.percent) <= PERCENT_TOLERANCE


async def check_todo_added(probe: Probe) -> bool:
    """The todo store now holds the milk (in Swedish or, translated, in English)."""
    todos = await probe.memory.list_todos()
    return any(mentions_milk(todo.text) for todo in todos)


async def check_todo_listed(probe: Probe) -> bool:
    """The answer reads the milk back from the list."""
    return mentions_milk(probe.answer)


def mentions_milk(text: str) -> bool:
    """Whether ``text`` names the todo item; a model may store it translated."""
    return any(word in text.lower() for word in MILK)


async def check_product(probe: Probe) -> bool:
    """The answer has 17 x 23 = 391."""
    return "391" in probe.answer.replace(" ", "")


async def check_reminder(probe: Probe) -> bool:
    """A reminder is due about five minutes after the request."""
    expected = probe.asked_at + REMINDER_DELAY
    reminders = await probe.memory.list_reminders()
    return any(abs(reminder.due_at - expected) <= REMINDER_TOLERANCE for reminder in reminders)


@dataclass(frozen=True, slots=True)
class Case:
    """One fixed request, the tools that can answer it and how to verify the answer."""

    label: str
    request: str
    tools: frozenset[str]
    check: Check


CASES: Final = (
    Case("time", "vad är klockan?", frozenset({"current_time"}), check_time),
    Case("RAM", "hur mycket RAM använder jag just nu?", frozenset({"system_monitor"}), check_ram),
    Case(
        "battery",
        "hur mycket batteri har jag?",
        frozenset({"battery_status", "system_monitor"}),
        check_battery,
    ),
    Case(
        "add todo",
        "lägg till mjölk på att-göra-listan",
        frozenset({"add_todo"}),
        check_todo_added,
    ),
    Case(
        "list todos",
        "vad står på min att-göra-lista?",
        frozenset({"list_todos"}),
        check_todo_listed,
    ),
    Case("calculation", "vad är 17 gånger 23?", frozenset(), check_product),
    Case(
        "reminder in 5 min",
        "påminn mig om fem minuter att ta ut tvätten",
        frozenset({"add_reminder"}),
        check_reminder,
    ),
)


# -- the agent under test ------------------------------------------------------------


class TimedLLM(LLMClient):
    """Wraps the real client to note when the first answer token of a turn arrives."""

    def __init__(self, inner: LLMClient) -> None:
        self.inner = inner
        self.turn_started = 0.0
        self.first_token: float | None = None

    @property
    def model(self) -> str:
        """The wrapped client's model tag."""
        return self.inner.model

    def start_turn(self) -> None:
        """Reset the clock for a new request."""
        self.turn_started = time.perf_counter()
        self.first_token = None

    async def stream(
        self, messages: Sequence[Message], tools: Sequence[ToolDefinition] = ()
    ) -> AsyncIterator[LLMEvent]:
        """Forward every event, noting the first non-blank text delta."""
        async for event in self.inner.stream(messages, tools):
            if isinstance(event, TextDelta) and event.text.strip() and self.first_token is None:
                self.first_token = time.perf_counter() - self.turn_started
            yield event


async def _refuse(self: AnyTool, params: Any, context: ToolContext) -> ToolResult:
    del params, context
    return ToolResult.failure(f"{self.spoken_name} is not available in this comparison.")


def sandboxed_registry() -> ToolRegistry:
    """Every real tool, with the unsafe ones swapped for look-alikes that refuse to run."""
    registry = ToolRegistry()
    for tool in ToolRegistry.discover().tools():
        if tool.name in SAFE_TOOLS:
            registry.register(tool)
            continue
        stand_in = type(f"Refusing{type(tool).__name__}", (type(tool),), {"run": _refuse})
        registry.register(stand_in())
    return registry


async def decline(prompt: str) -> bool:
    """Answer every confirmation question with no: nothing risky may run."""
    del prompt
    return False


def comparison_settings(data_dir: Path, args: argparse.Namespace, think: bool | None) -> Settings:
    """Default settings with the user's voice, a temp data_dir and one model."""
    return Settings(
        _env_file=None,
        data_dir=data_dir,
        llm_provider="ollama",
        ollama_base_url=args.ollama_url,
        tts_voice=VOICE,
        ollama_think=think,
        tool_embedding_model=args.embedding_model,
        user_settings_enabled=False,
    )


async def build_agent(
    settings: Settings, model: str, stack: AsyncExitStack
) -> tuple[Agent, TimedLLM, MemoryManager]:
    """The production agent over the sandboxed registry and a temp memory."""
    store = SqliteStore(settings.data_dir / "memory.sqlite3")
    await stack.enter_async_context(store)
    memory = MemoryManager(store, InMemoryVectorStore())
    await memory.start_conversation()
    http = httpx.AsyncClient(base_url=settings.ollama_base_url, timeout=OLLAMA_TIMEOUT)
    stack.push_async_callback(http.aclose)
    llm = TimedLLM(OllamaClient(http, model=model, think=settings.ollama_think))
    registry = sandboxed_registry()
    selector = build_tool_selector(settings, stack, registry)
    await _await_tool_vectors(selector, registry)
    bus = EventBus()
    stack.push_async_callback(bus.close)
    context = ToolContext(
        settings=settings, bus=bus, ask_confirmation=decline, platform=StubPlatform(), memory=memory
    )
    agent = Agent(
        llm=llm,
        registry=registry,
        policy=ConfirmationPolicy.from_settings(settings),
        persona=Persona.from_settings(settings),
        context=context,
        memory=memory,
        selector=selector,
    )
    return agent, llm, memory


async def _await_tool_vectors(selector: ToolSelector, registry: ToolRegistry) -> None:
    """Measure turns with the semantic ranking ready, as in a session past its first seconds."""
    if not isinstance(selector, RelevanceToolSelector):
        return
    if not isinstance(selector.semantic, EmbeddingToolRanker):
        print("  (no embedding model: tools ranked by keywords only)")
        return
    if not await selector.semantic.prepare(registry.tools()):
        print("  (embedding model unavailable: tools ranked by keywords only)")


# -- running and reporting -------------------------------------------------------


@dataclass(slots=True)
class Outcome:
    """One measured request."""

    case: Case
    report: TurnReport
    correct: bool
    first_token_s: float | None
    total_s: float

    @property
    def tools_called(self) -> list[str]:
        """Names of the tools the turn called, in order."""
        return [call.name for call in self.report.calls]


@dataclass(slots=True)
class RunResult:
    """Every request for one model and think setting."""

    label: str
    outcomes: list[Outcome] = field(default_factory=list)


async def silent(sentence: str) -> None:
    """Sentence sink that drops everything; the answer is read from the report."""
    del sentence


async def measure(agent: Agent, llm: TimedLLM, memory: MemoryManager, case: Case) -> Outcome:
    """Run one request and check its answer against ground truth."""
    asked_at = datetime.now().astimezone()
    llm.start_turn()
    report = await agent.respond(case.request, silent)
    total = time.perf_counter() - llm.turn_started
    correct = await case.check(Probe(answer=report.reply, memory=memory, asked_at=asked_at))
    return Outcome(case, report, correct, llm.first_token, total)


async def run_model(spec: str, args: argparse.Namespace) -> RunResult:
    """Warm one model up, then run every case through a fresh agent."""
    model, think = parse_spec(spec)
    result = RunResult(label=f"{model} (think {think_label(think)})")
    with tempfile.TemporaryDirectory(prefix="jarvis-compare-") as data_dir:
        async with AsyncExitStack() as stack:
            settings = comparison_settings(Path(data_dir), args, think)
            agent, llm, memory = await build_agent(settings, model, stack)
            await agent.respond(WARM_UP_REQUEST, silent)
            for case in CASES:
                outcome = await measure(agent, llm, memory, case)
                result.outcomes.append(outcome)
                print_outcome(outcome)
    return result


def parse_spec(spec: str) -> tuple[str, bool | None]:
    """``qwen3:8b@on`` -> ("qwen3:8b", True); no suffix means thinking off."""
    model, _, flag = spec.partition("@")
    if flag and flag not in THINK_FLAGS:
        raise SystemExit(f"unknown think flag '{flag}' in {spec}; use @off, @on or @default")
    return model, THINK_FLAGS[flag or "off"]


def think_label(think: bool | None) -> str:
    """How a think setting is written in the report."""
    return {False: "off", True: "on", None: "default"}[think]


def seconds(value: float | None) -> str:
    """A duration for the table, or a dash when there is none."""
    return "-" if value is None else f"{value:.1f}"


def print_outcome(outcome: Outcome) -> None:
    """One progress line (and the start of the answer) per request."""
    tools = ", ".join(outcome.tools_called) or "-"
    mark = "yes" if outcome.correct else "no"
    print(
        f"  {outcome.case.label:<18} {tools:<28} correct={mark:<3} "
        f"ttft={seconds(outcome.first_token_s):>5} s total={outcome.total_s:5.1f} s"
    )
    for call in outcome.report.calls:
        print(f"    {call.name}({call.arguments}) ok={call.ok}")
    print(f"    -> {outcome.report.reply[:160]}")


def markdown(results: Sequence[RunResult]) -> str:
    """The comparison as a Markdown table (one row per model and request) plus totals."""
    rows = [
        "| Model | Request | Tool called | Right tool | Correct | TTFT (s) | Total (s) |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for result in results:
        for outcome in result.outcomes:
            called = set(outcome.tools_called)
            right = called & outcome.case.tools if outcome.case.tools else not called
            rows.append(
                f"| {result.label} | {outcome.case.label} | "
                f"{', '.join(outcome.tools_called) or '-'} | {'yes' if right else 'no'} | "
                f"{'yes' if outcome.correct else 'no'} | {seconds(outcome.first_token_s)} | "
                f"{outcome.total_s:.1f} |"
            )
    rows += [
        "",
        "| Model | Correct | Median TTFT (s) | Total time (s) |",
        "| --- | --- | --- | --- |",
    ]
    rows += [totals_row(result) for result in results]
    return "\n".join(rows)


def totals_row(result: RunResult) -> str:
    """Correct answers, median time to first token and total time of a run."""
    correct = sum(outcome.correct for outcome in result.outcomes)
    firsts = [o.first_token_s for o in result.outcomes if o.first_token_s is not None]
    median = statistics.median(firsts) if firsts else None
    total = sum(outcome.total_s for outcome in result.outcomes)
    return (
        f"| {result.label} | {correct}/{len(result.outcomes)} | {seconds(median)} | {total:.1f} |"
    )


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    """Command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("models", nargs="+", help="Ollama model tags, optionally @off/@on/@default")
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument(
        "--embedding-model", default="bge-m3", help="tool-selection embedding model, or 'none'"
    )
    parser.add_argument(
        "--repeat", type=int, default=1, help="rounds over all models (answers vary run to run)"
    )
    return parser.parse_args(argv)


async def main(argv: Sequence[str]) -> int:
    """Run every model in turn and print the Markdown report."""
    args = parse_args(argv)
    structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(30))
    results = []
    for round_number in range(1, args.repeat + 1):
        for spec in args.models:
            print(f"\n== {spec} (round {round_number}/{args.repeat})")
            result = await run_model(spec, args)
            if args.repeat > 1:
                result.label += f" #{round_number}"
            results.append(result)
    print("\n" + markdown(results))
    if args.repeat > 1:
        print("\n" + summary(results))
    return 0


def summary(results: Sequence[RunResult]) -> str:
    """All rounds of each configuration pooled: correct answers, median TTFT, mean total."""
    pooled: dict[str, list[Outcome]] = {}
    rounds: dict[str, int] = {}
    for result in results:
        base = result.label.rsplit(" #", 1)[0]
        pooled.setdefault(base, []).extend(result.outcomes)
        rounds[base] = rounds.get(base, 0) + 1
    rows = [
        "| Model | Correct (all rounds) | Median TTFT (s) | Mean total per round (s) |",
        "| --- | --- | --- | --- |",
    ]
    for base, outcomes in pooled.items():
        firsts = [o.first_token_s for o in outcomes if o.first_token_s is not None]
        median = statistics.median(firsts) if firsts else None
        per_round = sum(o.total_s for o in outcomes) / rounds[base]
        correct = sum(o.correct for o in outcomes)
        rows.append(f"| {base} | {correct}/{len(outcomes)} | {seconds(median)} | {per_round:.1f} |")
    return "\n".join(rows)


if __name__ == "__main__":
    sys.exit(asyncio.run(main(sys.argv[1:])))


# Design notes:
# - The agent is assembled here rather than through ``build_application``:
#   the comparison needs the sandboxed registry, a declining confirmation
#   asker and the stub OS layer, and must never start the overlay, tray,
#   proactive scheduler or real integrations. The pieces that decide the
#   result (Agent, selector, persona, OllamaClient) are the production ones.
# - Refusing stand-ins keep the tool list identical to a real session, so
#   the tool-selection pressure the model faces is the real one; a call to
#   one shows up as a wrong tool, which is what it would be.
# - Time to first token is measured on the first non-blank text delta of
#   the turn, after any tool round, because that is when speech can start.
# - Checks are deliberately lenient about phrasing (any number close to the
#   truth, either HH:MM or HH.MM) and strict about facts.
