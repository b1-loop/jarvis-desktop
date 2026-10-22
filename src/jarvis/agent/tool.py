"""``Tool`` base class, ``ToolResult`` and the ``ToolContext`` tools run in.

A tool is one capability the LLM can invoke: a name, a description, a
pydantic input model and an async ``run``. Tools never import concrete
services; everything they need (OS facade, memory, settings, the event bus,
a clock, a way to ask the user) arrives through :class:`ToolContext`, so a
tool is unit-testable with fakes and cannot reach around the confirmation
policy. The LLM-facing schema is derived from the pydantic model so the
contract the model sees and the validation the input gets are one thing.
"""

from __future__ import annotations

import inspect
import json
import re
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any, ClassVar, Final, Protocol, cast

from pydantic import BaseModel, ConfigDict, JsonValue, ValidationError

from jarvis.config import Settings
from jarvis.core.events import EventBus, ToolCallFinished, ToolCallStarted
from jarvis.integrations.services import IntegrationServices
from jarvis.utils.errors import JarvisError
from jarvis.utils.logging import get_logger
from jarvis.utils.timing import Clock, SystemClock
from jarvis.vision.screen import ScreenReaderProtocol

if TYPE_CHECKING:
    from datetime import datetime

    from jarvis.memory.models import Note, Reminder, Todo
    from jarvis.os_integration.base import Platform

log = get_logger(__name__)

MAX_SUMMARY_CHARS: Final = 120
"""Length cap for the argument summary carried in events and logs."""

_NAME_PATTERN: Final = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_MS_PER_SECOND: Final = 1000.0


class ToolCategory(StrEnum):
    """Coarse grouping used by the confirmation policy and the dashboard."""

    INFORMATION = "information"
    SYSTEM = "system"
    DEVELOPER = "developer"
    FILES = "files"
    COMMUNICATION = "communication"
    PRODUCTIVITY = "productivity"
    MEDIA = "media"
    SMART_HOME = "smart_home"
    WEB = "web"


class ToolError(JarvisError):
    """Base class for tool failures.

    Attributes:
        tool_name: The tool (or, for definition errors, the class) involved.
    """

    def __init__(self, tool_name: str, message: str) -> None:
        self.tool_name = tool_name
        super().__init__(f"{tool_name}: {message}")


class ToolDefinitionError(ToolError):
    """A ``Tool`` subclass is missing or mis-declares a required attribute."""


class ToolInputError(ToolError):
    """The arguments supplied for a tool failed pydantic validation."""


class ToolInput(BaseModel):
    """Base for tool input models; unknown fields are rejected, not ignored.

    An LLM that invents an argument should get a precise validation error
    rather than silently having it dropped.
    """

    model_config = ConfigDict(extra="forbid")


@dataclass(frozen=True, slots=True)
class ToolResult:
    """What a tool hands back to the agent.

    Attributes:
        ok: Whether the tool did what was asked.
        spoken: One short sentence for TTS; the user hears this.
        detail: Fuller text for the LLM's next turn and the dashboard.
        data: Structured, JSON-serialisable payload for chaining and the UI.
    """

    ok: bool
    spoken: str
    detail: str = ""
    data: dict[str, JsonValue] | None = None

    @classmethod
    def success(
        cls, spoken: str, *, detail: str = "", data: dict[str, JsonValue] | None = None
    ) -> ToolResult:
        """Build a successful result."""
        return cls(ok=True, spoken=spoken, detail=detail or spoken, data=data)

    @classmethod
    def failure(
        cls, spoken: str, *, detail: str = "", data: dict[str, JsonValue] | None = None
    ) -> ToolResult:
        """Build a failed result; ``detail`` should say precisely why."""
        return cls(ok=False, spoken=spoken, detail=detail or spoken, data=data)


class MemoryFacade(Protocol):
    """The part of ``jarvis.memory.MemoryManager`` that tools may use.

    Notes, todos and reminders only: recall and fact storage belong to the
    agent loop, which decides what goes into the prompt. ``MemoryManager``
    satisfies this structurally, so tools never import the memory package.
    """

    async def add_note(self, content: str, title: str = "") -> Note:
        """Store a note and index it for recall."""

    async def search_notes(self, query: str, limit: int = 10) -> list[Note]:
        """Full-text search over notes, best match first."""

    async def add_todo(self, text: str) -> Todo:
        """Add an open todo."""

    async def list_todos(self, *, include_done: bool = False) -> list[Todo]:
        """Open todos, or all of them."""

    async def complete_todo(self, todo_id: int) -> Todo:
        """Tick a todo off; raises ``NotFoundError`` for an unknown id."""

    async def add_reminder(self, text: str, due_at: datetime) -> Reminder:
        """Schedule a reminder."""

    async def list_reminders(self, *, include_fired: bool = False) -> list[Reminder]:
        """Pending reminders by due time, or all of them."""


class TrackedProcess(Protocol):
    """A spawned child process as the job tracker needs to see it."""

    @property
    def pid(self) -> int:
        """Operating-system process id."""

    async def wait(self) -> int:
        """Wait for the process to exit and return its exit code."""


class BackgroundJobTracker(Protocol):
    """Watches long-running processes a tool started and left running.

    The implementation (``jarvis.tools._jobs.BackgroundJobs``) also satisfies
    ``jarvis.proactive.triggers.CommandWatcher``, so a dev server or build
    that exits is announced by the proactive layer.
    """

    def track(self, command: str, process: TrackedProcess) -> str:
        """Start watching ``process``; returns a job id."""


class HudAppearance(Protocol):
    """The running HUD's palette, as the ``set_hud_colour`` tool changes it.

    Implemented by ``ui.hud_settings.SettingsController``, so a spoken
    change is validated, saved and applied exactly like one picked on the
    HUD's settings page.
    """

    async def set_colour(self, theme: str, accent: str | None = None) -> None:
        """Switch to ``theme``; with ``accent`` (``#rrggbb``) set the custom accent first.

        Raises:
            JarvisError: If a value is rejected.
        """


ConfirmationAsker = Callable[[str], Awaitable[bool]]
"""Asks the user a yes/no question (spoken or typed) and returns their answer."""

IntegrationProviders = IntegrationServices
"""Alias kept for the composition root (``core/bootstrap.py``), which builds the
bundle under this name; ``IntegrationServices`` has the same five provider
fields plus web search, transcripts, the web client and the draft book."""


@dataclass(frozen=True, slots=True)
class ToolContext:
    """Everything a tool may use, injected per conversation turn.

    Attributes:
        settings: Application settings (read-only by convention).
        bus: Event bus for progress events.
        ask_confirmation: How to ask the user a yes/no question. Tools
            normally never call this themselves; the policy does.
        clock: Time source; a fake in tests.
        platform: OS facade, ``None`` in text-only runs without a backend.
        memory: Notes, todos and reminders, ``None`` when memory is disabled.
        integrations: External services (email, calendar, web ...), ``None``
            when the composition root wired none; integration tools then
            answer "not configured" instead of failing.
        vision: Screen understanding, ``None`` when it is not wired in.
        jobs: Tracker for detached processes such as dev servers, ``None``
            when nothing consumes their completion.
        hud: The HUD's palette, ``None`` without a HUD (text-only mode).
    """

    settings: Settings
    bus: EventBus
    ask_confirmation: ConfirmationAsker
    clock: Clock = field(default_factory=SystemClock)
    platform: Platform | None = None
    memory: MemoryFacade | None = None
    integrations: IntegrationServices | None = None
    vision: ScreenReaderProtocol | None = None
    jobs: BackgroundJobTracker | None = None
    hud: HudAppearance | None = None


class Tool[InputT: BaseModel](ABC):
    """One capability the agent can call.

    Subclasses declare ``name``, ``description`` and ``Input`` as class
    attributes and implement ``run``. ``requires_confirmation`` marks tools
    that change the world; ``needs_confirmation`` can refine that per call.
    ``keywords`` lists words a user says for this tool that appear in neither
    its name nor its description ("clock" for ``current_time``); the
    per-turn tool selection (``agent/tool_selection.py``) matches on them.
    Discovery instantiates tools with no arguments, so constructor
    parameters must all have defaults.
    """

    name: ClassVar[str]
    description: ClassVar[str]
    Input: ClassVar[type[BaseModel]]
    category: ClassVar[ToolCategory] = ToolCategory.INFORMATION
    requires_confirmation: ClassVar[bool] = False
    keywords: ClassVar[tuple[str, ...]] = ()

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        if getattr(cls.run, "__isabstractmethod__", False):
            return  # still abstract; the concrete subclass is checked instead
        _validate_definition(cls)

    @property
    def spoken_name(self) -> str:
        """The name as it is said aloud: ``shell_command`` -> ``shell command``."""
        return self.name.replace("_", " ")

    @abstractmethod
    async def run(self, params: InputT, context: ToolContext) -> ToolResult:
        """Do the work. Raise ``JarvisError`` for expected failures."""

    def needs_confirmation(self, params: InputT, context: ToolContext) -> bool:
        """Whether this particular call must be confirmed by the user.

        Defaults to the class-level flag; override for tools whose risk
        depends on the arguments (the shell tool looks at the command).
        """
        del params, context
        return self.requires_confirmation

    def confirmation_prompt(self, params: InputT) -> str:
        """The yes/no question put to the user before a confirmed call."""
        return f"Do you want me to run {self.spoken_name} with {self.summarize(params)}?"

    def summarize(self, params: InputT) -> str:
        """Short rendering of the arguments for events, logs and prompts."""
        text = json.dumps(params.model_dump(mode="json"), default=str, sort_keys=True)
        if len(text) <= MAX_SUMMARY_CHARS:
            return text
        return text[: MAX_SUMMARY_CHARS - 1] + "…"

    def parse(self, raw: Mapping[str, object]) -> InputT:
        """Validate raw arguments (as the LLM sent them) into ``Input``.

        Raises:
            ToolInputError: With one line per offending field.
        """
        try:
            return cast(InputT, self.Input.model_validate(dict(raw)))
        except ValidationError as error:
            raise ToolInputError(self.name, _describe_validation_error(error)) from error

    def invalid_input_result(self, error: ToolInputError) -> ToolResult:
        """The failure result returned when ``parse`` rejects the arguments."""
        return ToolResult.failure(
            f"I couldn't run {self.spoken_name}: the request was missing something.",
            detail=str(error),
        )

    async def call(self, raw: Mapping[str, object], context: ToolContext) -> ToolResult:
        """Validate and execute, without consulting the confirmation policy.

        For tests and trusted internal callers. The agent loop goes through
        ``ConfirmationPolicy.execute_with_policy`` instead.
        """
        try:
            params = self.parse(raw)
        except ToolInputError as error:
            return self.invalid_input_result(error)
        return await self.execute(params, context)

    async def execute(self, params: InputT, context: ToolContext) -> ToolResult:
        """Run with validated input, publishing tool-call events and timing.

        Never raises for tool failures: ``JarvisError`` and unexpected
        exceptions become ``ok=False`` results (the latter with a logged
        traceback) so one broken tool cannot end the conversation.
        """
        summary = self.summarize(params)
        context.bus.publish(ToolCallStarted(name=self.name, args_summary=summary))
        started = context.clock.monotonic()
        result = await self._run_guarded(params, context)
        duration_ms = (context.clock.monotonic() - started) * _MS_PER_SECOND
        context.bus.publish(
            ToolCallFinished(
                name=self.name,
                args_summary=summary,
                duration_ms=duration_ms,
                ok=result.ok,
                error=None if result.ok else result.detail,
            )
        )
        log.info("tool.finished", tool=self.name, ok=result.ok, duration_ms=round(duration_ms, 1))
        return result

    async def _run_guarded(self, params: InputT, context: ToolContext) -> ToolResult:
        try:
            return await self.run(params, context)
        except JarvisError as error:
            log.warning("tool.failed", tool=self.name, error=str(error))
            return ToolResult.failure(f"I couldn't complete {self.spoken_name}.", detail=str(error))
        except Exception as error:
            log.exception("tool.crashed", tool=self.name)
            return ToolResult.failure(
                f"Something went wrong while running {self.spoken_name}.",
                detail=f"{type(error).__name__}: {error}",
            )

    @classmethod
    def to_llm_schema(cls) -> dict[str, Any]:
        """The tool definition in the shape Anthropic-style tool calling expects."""
        return {
            "name": cls.name,
            "description": cls.description,
            "input_schema": cls.Input.model_json_schema(),
        }


def _validate_definition(cls: type[Tool[Any]]) -> None:
    missing = [attr for attr in ("name", "description", "Input") if not hasattr(cls, attr)]
    if missing:
        raise ToolDefinitionError(cls.__name__, f"missing class attributes: {', '.join(missing)}")
    if not isinstance(cls.name, str) or not _NAME_PATTERN.fullmatch(cls.name):
        raise ToolDefinitionError(
            cls.__name__, f"name {cls.name!r} must match {_NAME_PATTERN.pattern}"
        )
    if not isinstance(cls.description, str) or not cls.description.strip():
        raise ToolDefinitionError(cls.__name__, "description must be a non-empty string")
    if not (inspect.isclass(cls.Input) and issubclass(cls.Input, BaseModel)):
        raise ToolDefinitionError(cls.__name__, "Input must be a pydantic BaseModel subclass")


def _describe_validation_error(error: ValidationError) -> str:
    lines = []
    for item in error.errors():
        location = ".".join(str(part) for part in item["loc"]) or "input"
        lines.append(f"{location}: {item['msg']}")
    return "; ".join(lines)


# Design notes:
# - Tool is generic in its input model so ``run`` is typed with the concrete
#   model in every plugin, while ``Input`` stays a plain ClassVar because
#   ClassVar cannot carry a type variable. The two must agree; ``parse``
#   casts and the test for each tool exercises the pairing.
# - Definition errors are raised from __init_subclass__, i.e. at import
#   time. The registry turns them into recorded discovery failures, so a
#   broken plugin shows up on the dashboard instead of crashing startup.
# - ``execute`` converts exceptions into failed results. That is deliberate
#   graceful degradation, not swallowing: expected failures are logged at
#   WARNING with their message and unexpected ones at ERROR with a traceback.
# - Services are interfaces: the ``Platform`` ABC and Protocols that the
#   concrete services satisfy structurally. They are imported for type
#   checking only, so importing a tool never drags in the native OS layer
#   or SQLite. Tools depend on shapes, never on modules.
# - ``memory`` exposes notes, todos and reminders only. The step-6
#   placeholder had recall/remember, which no tool used and which never
#   matched MemoryManager; recall belongs to the agent loop.
