"""YAML-defined routines: a named list of tool calls Jarvis runs as one unit.

A routine file says *when* ("07:30 on weekdays", the phrase "good morning",
or only when asked) and *what* (steps, each a tool name, its arguments and a
sentence template to speak with the result). :func:`load_routines` turns a
directory of such files into validated :class:`Routine` models with precise
error messages, and :class:`RoutineRunner` executes them through two
Protocols, :class:`ToolInvoker` and :class:`Speaker`, so this module never
imports the agent or a tool.
"""

from __future__ import annotations

import asyncio
import string
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import time, tzinfo
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any, Literal, Protocol

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from jarvis.proactive.scheduler import JobCallback, Scheduler
from jarvis.utils.errors import JarvisError
from jarvis.utils.logging import get_logger

log = get_logger(__name__)

WEEKDAYS: Mapping[str, int] = {
    "mon": 0,
    "tue": 1,
    "wed": 2,
    "thu": 3,
    "fri": 4,
    "sat": 5,
    "sun": 6,
}
"""YAML day names to ``datetime.weekday`` numbers."""

WeekdayName = Literal["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
ALL_DAYS: tuple[WeekdayName, ...] = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")
ErrorPolicy = Literal["continue", "abort"]
SPEAK_PLACEHOLDERS: frozenset[str] = frozenset({"result", "tool"})
"""Names a ``speak`` template may reference."""

ROUTINE_SUFFIXES = (".yaml", ".yml")


class RoutineError(JarvisError):
    """A routine file is missing, unreadable or invalid.

    Attributes:
        path: The file concerned.
        problems: One line per issue, each naming the field at fault.
    """

    def __init__(self, path: Path, problems: list[str]) -> None:
        self.path = path
        self.problems = problems
        details = "\n  - ".join(problems)
        super().__init__(f"invalid routine {path}:\n  - {details}")


class UnknownToolError(JarvisError):
    """Raised by a :class:`ToolInvoker` when no tool has the requested name.

    Attributes:
        tool: The name that was asked for.
    """

    def __init__(self, tool: str) -> None:
        self.tool = tool
        super().__init__(f"unknown tool '{tool}'")


# ---------------------------------------------------------------------------
# Routine file model
# ---------------------------------------------------------------------------


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class TimeTrigger(_StrictModel):
    """Run at a wall-clock time on the listed days."""

    kind: Literal["time"]
    at: time
    days: tuple[WeekdayName, ...] = ALL_DAYS

    @field_validator("at", mode="before")
    @classmethod
    def _reject_unquoted_time(cls, value: object) -> object:
        """Explain the YAML 1.1 trap: an unquoted ``07:30`` parses as the integer 450."""
        if isinstance(value, int):
            raise ValueError('quote the time, e.g. at: "07:30" (YAML reads 07:30 as a number)')
        return value

    @field_validator("days")
    @classmethod
    def _require_days(cls, value: tuple[WeekdayName, ...]) -> tuple[WeekdayName, ...]:
        if not value:
            raise ValueError("list at least one day, e.g. [mon, tue, wed, thu, fri]")
        return value

    @property
    def weekdays(self) -> frozenset[int]:
        """``datetime.weekday`` numbers for the scheduler."""
        return frozenset(WEEKDAYS[day] for day in self.days)


class PhraseTrigger(_StrictModel):
    """Run when the user says ``phrase`` (matched case-insensitively)."""

    kind: Literal["phrase"]
    phrase: str = Field(min_length=1)

    @field_validator("phrase")
    @classmethod
    def _normalise(cls, value: str) -> str:
        return _normalise_phrase(value)


class ManualTrigger(_StrictModel):
    """Run only when asked by name."""

    kind: Literal["manual"]


RoutineTrigger = Annotated[TimeTrigger | PhraseTrigger | ManualTrigger, Field(discriminator="kind")]


class Step(_StrictModel):
    """One tool invocation inside a routine.

    Attributes:
        tool: Plugin name as registered by the agent's tool registry.
        args: Keyword arguments validated by the tool itself at run time.
        speak: Template spoken after the step; ``{result}`` is the tool's
            text output. ``None`` keeps the step silent.
        timeout_seconds: Per-step budget; the step is reported as timed out
            and the routine's error policy decides what happens next.
        on_error: Overrides the routine-level policy for this step.
    """

    tool: str = Field(min_length=1)
    args: dict[str, Any] = Field(default_factory=dict)
    speak: str | None = None
    timeout_seconds: float = Field(default=30.0, gt=0)
    on_error: ErrorPolicy | None = None

    @field_validator("speak")
    @classmethod
    def _check_placeholders(cls, value: str | None) -> str | None:
        if value is None:
            return None
        unknown = _placeholders(value) - SPEAK_PLACEHOLDERS
        if unknown:
            allowed = ", ".join(sorted(SPEAK_PLACEHOLDERS))
            raise ValueError(f"unknown placeholder {sorted(unknown)}; allowed: {allowed}")
        return value


class Routine(_StrictModel):
    """A validated routine file.

    Attributes:
        name: Unique within the routines directory.
        description: Free text for the dashboard.
        trigger: When the routine starts.
        steps: Executed in order.
        max_duration_seconds: Whole-routine budget.
        on_error: Default policy when a step fails.
        intro: Spoken before the first step.
        outro: Spoken after the last step when nothing aborted.
    """

    name: str = Field(min_length=1)
    description: str = ""
    trigger: RoutineTrigger
    steps: tuple[Step, ...] = Field(min_length=1)
    max_duration_seconds: float = Field(default=300.0, gt=0)
    on_error: ErrorPolicy = "continue"
    intro: str | None = None
    outro: str | None = None

    def policy_for(self, step: Step) -> ErrorPolicy:
        """The error policy that applies to ``step``."""
        return step.on_error or self.on_error


def _placeholders(template: str) -> set[str]:
    try:
        parsed = list(string.Formatter().parse(template))
    except ValueError as error:
        raise ValueError(f"malformed template: {error}") from error
    return {name for _, name, _, _ in parsed if name}


def _normalise_phrase(text: str) -> str:
    letters = "".join(ch if ch.isalnum() or ch.isspace() else " " for ch in text.lower())
    return " ".join(letters.split())


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_routine(path: Path) -> Routine:
    """Parse and validate one routine file.

    Raises:
        RoutineError: With one line per problem, naming the field and, for
            syntax errors, the line and column.
    """
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise RoutineError(path, [f"cannot read file: {error.strerror or error}"]) from error
    except yaml.YAMLError as error:
        raise RoutineError(path, [f"YAML syntax: {_describe_yaml_error(error)}"]) from error
    if not isinstance(raw, dict):
        raise RoutineError(path, ["top level must be a mapping with name, trigger and steps"])
    try:
        return Routine.model_validate(raw)
    except ValidationError as error:
        raise RoutineError(path, _describe_validation_error(error)) from error


def load_routines(directory: Path) -> list[Routine]:
    """Load every ``*.yaml``/``*.yml`` file in ``directory``, sorted by name.

    A missing directory is not an error (a fresh install has none); it is
    logged and yields an empty list.

    Raises:
        RoutineError: On the first invalid file, or when two files share a
            routine name.
    """
    if not directory.is_dir():
        log.warning("proactive.routines.missing_dir", directory=str(directory))
        return []
    files = sorted(p for p in directory.iterdir() if p.suffix.lower() in ROUTINE_SUFFIXES)
    routines: dict[str, Path] = {}
    loaded: list[Routine] = []
    for file in files:
        routine = load_routine(file)
        if routine.name in routines:
            first = routines[routine.name]
            raise RoutineError(file, [f"name '{routine.name}' already used by {first.name}"])
        routines[routine.name] = file
        loaded.append(routine)
    log.info("proactive.routines.loaded", count=len(loaded), directory=str(directory))
    return loaded


def _describe_yaml_error(error: yaml.YAMLError) -> str:
    mark = getattr(error, "problem_mark", None)
    problem = getattr(error, "problem", None) or str(error)
    if mark is None:
        return problem
    return f"{problem} (line {mark.line + 1}, column {mark.column + 1})"


def _describe_validation_error(error: ValidationError) -> list[str]:
    lines = []
    for item in error.errors():
        location = ".".join(str(part) for part in item["loc"]) or "<root>"
        lines.append(f"{location}: {item['msg']}")
    return lines


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------


class ToolInvoker(Protocol):
    """Runs a named tool; the agent's registry implements this."""

    async def invoke(self, name: str, args: Mapping[str, Any]) -> str:
        """Run tool ``name`` with ``args`` and return its text output.

        Raises:
            UnknownToolError: If no tool has that name.
        """


class Speaker(Protocol):
    """Voices text; the orchestrator implements this over TTS."""

    async def say(self, text: str) -> None:
        """Speak ``text`` and return when done."""


class StepStatus(StrEnum):
    """Outcome of one step."""

    OK = "ok"
    FAILED = "failed"
    TIMED_OUT = "timed_out"
    UNKNOWN_TOOL = "unknown_tool"


@dataclass(frozen=True, slots=True)
class StepResult:
    """What happened to one step."""

    tool: str
    status: StepStatus
    output: str | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        """True if the tool ran to completion."""
        return self.status is StepStatus.OK


@dataclass(slots=True)
class RoutineReport:
    """Outcome of a whole routine, for logs and the dashboard."""

    name: str
    steps: list[StepResult] = field(default_factory=list)
    aborted: bool = False
    timed_out: bool = False

    @property
    def ok(self) -> bool:
        """True if every step succeeded and nothing cut the routine short."""
        return not self.aborted and not self.timed_out and all(s.ok for s in self.steps)


class RoutineRunner:
    """Executes routines step by step, speaking results and problems."""

    def __init__(self, invoker: ToolInvoker, speaker: Speaker) -> None:
        self._invoker = invoker
        self._speaker = speaker

    async def run(self, routine: Routine) -> RoutineReport:
        """Run ``routine`` within its duration budget and report what happened.

        Never raises for tool problems: those are spoken, logged and
        recorded in the report. Cancellation propagates.
        """
        report = RoutineReport(routine.name)
        log.info("proactive.routine.start", routine=routine.name, steps=len(routine.steps))
        if routine.intro:
            await self._speaker.say(routine.intro)
        try:
            async with asyncio.timeout(routine.max_duration_seconds):
                await self._run_steps(routine, report)
        except TimeoutError:
            report.timed_out = True
            log.warning("proactive.routine.timed_out", routine=routine.name)
            await self._speaker.say(f"{routine.name} ran out of time.")
        if routine.outro and report.ok:
            await self._speaker.say(routine.outro)
        log.info("proactive.routine.done", routine=routine.name, ok=report.ok)
        return report

    async def _run_steps(self, routine: Routine, report: RoutineReport) -> None:
        for step in routine.steps:
            result = await self._run_step(step)
            report.steps.append(result)
            if result.ok:
                await self._speak_result(step, result)
                continue
            await self._speaker.say(_failure_sentence(result))
            if routine.policy_for(step) == "abort":
                report.aborted = True
                log.warning("proactive.routine.aborted", routine=routine.name, tool=step.tool)
                return

    async def _run_step(self, step: Step) -> StepResult:
        try:
            async with asyncio.timeout(step.timeout_seconds):
                output = await self._invoker.invoke(step.tool, step.args)
        except UnknownToolError as error:
            log.warning("proactive.step.unknown_tool", tool=step.tool)
            return StepResult(step.tool, StepStatus.UNKNOWN_TOOL, error=str(error))
        except TimeoutError:
            log.warning("proactive.step.timed_out", tool=step.tool, limit_s=step.timeout_seconds)
            return StepResult(step.tool, StepStatus.TIMED_OUT, error="timed out")
        except Exception as error:
            log.exception("proactive.step.failed", tool=step.tool)
            return StepResult(step.tool, StepStatus.FAILED, error=repr(error))
        return StepResult(step.tool, StepStatus.OK, output=output)

    async def _speak_result(self, step: Step, result: StepResult) -> None:
        if step.speak is None:
            return
        await self._speaker.say(step.speak.format(result=result.output, tool=step.tool))


def _failure_sentence(result: StepResult) -> str:
    if result.status is StepStatus.UNKNOWN_TOOL:
        return f"I don't have a tool called {result.tool}, skipping it."
    if result.status is StepStatus.TIMED_OUT:
        return f"{result.tool} took too long, skipping it."
    return f"{result.tool} failed, skipping it."


# ---------------------------------------------------------------------------
# Wiring routines to the scheduler and to speech
# ---------------------------------------------------------------------------


class RoutineRegistry:
    """Looks routines up by name or phrase and schedules the timed ones."""

    def __init__(self, routines: Iterable[Routine] = ()) -> None:
        self._routines: dict[str, Routine] = {r.name: r for r in routines}

    @property
    def routines(self) -> list[Routine]:
        """All known routines."""
        return list(self._routines.values())

    def by_name(self, name: str) -> Routine | None:
        """Exact name lookup."""
        return self._routines.get(name)

    def by_phrase(self, utterance: str) -> Routine | None:
        """The phrase-triggered routine whose phrase appears in ``utterance``.

        Matching is on whole words after lower-casing and stripping
        punctuation, so "Good morning, Jarvis!" matches "good morning".
        """
        padded = f" {_normalise_phrase(utterance)} "
        for routine in self._routines.values():
            trigger = routine.trigger
            if isinstance(trigger, PhraseTrigger) and f" {trigger.phrase} " in padded:
                return routine
        return None

    def schedule(
        self, scheduler: Scheduler, runner: RoutineRunner, *, tz: tzinfo | None = None
    ) -> dict[str, str]:
        """Register every time-triggered routine as a daily job.

        Returns:
            Routine name to scheduler job id.
        """
        job_ids: dict[str, str] = {}
        for routine in self._routines.values():
            trigger = routine.trigger
            if not isinstance(trigger, TimeTrigger):
                continue
            job_ids[routine.name] = scheduler.add_daily(
                f"routine:{routine.name}",
                trigger.at,
                _runner_callback(runner, routine),
                weekdays=trigger.weekdays,
                tz=tz,
            )
        return job_ids


def _runner_callback(runner: RoutineRunner, routine: Routine) -> JobCallback:
    async def run() -> None:
        await runner.run(routine)

    return run


# Design notes:
# - YAML plus pydantic rather than a Python DSL: the user edits routines by
#   hand, and extra="forbid" turns a typo like "tol:" into a one-line error
#   with the field path instead of a silently ignored step.
# - The unquoted-time validator exists because YAML 1.1 (PyYAML) parses
#   07:30 as the sexagesimal integer 450; the message says what to type.
# - The runner reports rather than raises: a routine is conversation, and a
#   missing weather tool should produce "skipping it" out loud, not a
#   traceback in the morning. Per-step timeouts nest inside the routine
#   budget; asyncio.timeout handles the nesting correctly.
# - ToolInvoker/Speaker are deliberately tiny so the agent and the
#   orchestrator can satisfy them with a two-line adapter and the
#   tests here need no agent at all.
# - Assumption: a step without "speak" is intentionally silent (side-effect
#   tools such as "set volume"); the runner does not read raw tool output.
