import asyncio
from collections.abc import Callable

import pytest

from jarvis.agent.tool import (
    MAX_SUMMARY_CHARS,
    Tool,
    ToolCategory,
    ToolContext,
    ToolDefinitionError,
    ToolInput,
    ToolInputError,
    ToolResult,
)
from jarvis.config import Settings
from jarvis.core.events import EventBus, ToolCallFinished, ToolCallStarted
from jarvis.utils.errors import JarvisError
from tests.agent.fakes import EchoInput, EchoTool, FlakyTool, make_context
from tests.fakes import FakeClock

WAIT = 1.0


@pytest.fixture
def settings(make_settings: Callable[..., Settings]) -> Settings:
    return make_settings()


# --- ToolResult -------------------------------------------------------------


def test_result_factories_default_detail_to_spoken() -> None:
    success = ToolResult.success("Done.")
    failure = ToolResult.failure("Nope.", detail="because")

    assert (success.ok, success.spoken, success.detail, success.data) == (
        True,
        "Done.",
        "Done.",
        None,
    )
    assert (failure.ok, failure.spoken, failure.detail) == (False, "Nope.", "because")


# --- definition checks ------------------------------------------------------


def test_missing_class_attributes_fail_at_definition_time() -> None:
    with pytest.raises(ToolDefinitionError, match="missing class attributes: name, description"):

        class Incomplete(Tool[EchoInput]):
            Input = EchoInput

            async def run(self, params: EchoInput, context: ToolContext) -> ToolResult:
                return ToolResult.success("x")


@pytest.mark.parametrize("bad_name", ["Echo", "with space", "", "9lives", "a-b"])
def test_tool_names_must_be_snake_case_identifiers(bad_name: str) -> None:
    with pytest.raises(ToolDefinitionError, match="must match"):

        class BadName(Tool[EchoInput]):
            name = bad_name
            description = "x"
            Input = EchoInput

            async def run(self, params: EchoInput, context: ToolContext) -> ToolResult:
                return ToolResult.success("x")


def test_description_must_be_non_empty() -> None:
    with pytest.raises(ToolDefinitionError, match="description"):

        class Blank(Tool[EchoInput]):
            name = "blank"
            description = "   "
            Input = EchoInput

            async def run(self, params: EchoInput, context: ToolContext) -> ToolResult:
                return ToolResult.success("x")


def test_input_must_be_a_pydantic_model() -> None:
    with pytest.raises(ToolDefinitionError, match="pydantic"):

        class BadInput(Tool[EchoInput]):
            name = "bad_input"
            description = "x"
            Input = dict  # type: ignore[assignment]

            async def run(self, params: EchoInput, context: ToolContext) -> ToolResult:
                return ToolResult.success("x")


def test_intermediate_abstract_subclasses_are_not_checked() -> None:
    class Base(Tool[EchoInput]):
        category = ToolCategory.WEB

    class Concrete(Base):
        name = "concrete"
        description = "x"
        Input = EchoInput

        async def run(self, params: EchoInput, context: ToolContext) -> ToolResult:
            return ToolResult.success("x")

    assert Concrete.category is ToolCategory.WEB
    assert Concrete().spoken_name == "concrete"


def test_definition_errors_are_jarvis_errors() -> None:
    error = ToolDefinitionError("X", "broken")

    assert isinstance(error, JarvisError)
    assert error.tool_name == "X"


# --- schema and parsing -----------------------------------------------------


def test_llm_schema_has_the_anthropic_shape() -> None:
    schema = EchoTool.to_llm_schema()

    assert schema["name"] == "echo"
    assert schema["description"] == "Repeat a piece of text."
    assert schema["input_schema"]["type"] == "object"
    assert set(schema["input_schema"]["properties"]) == {"text", "repeat"}
    assert schema["input_schema"]["required"] == ["text"]
    assert schema["input_schema"]["additionalProperties"] is False


def test_parse_returns_the_validated_model() -> None:
    params = EchoTool().parse({"text": "hi", "repeat": "2"})

    assert isinstance(params, EchoInput)
    assert params.repeat == 2


@pytest.mark.parametrize(
    ("raw", "fragment"),
    [
        ({}, "text: Field required"),
        ({"text": "hi", "repeat": 9}, "repeat: Input should be less than or equal to 3"),
        ({"text": "hi", "bogus": 1}, "bogus: Extra inputs are not permitted"),
    ],
)
def test_parse_reports_the_offending_field(raw: dict[str, object], fragment: str) -> None:
    with pytest.raises(ToolInputError) as info:
        EchoTool().parse(raw)

    assert fragment in str(info.value)
    assert info.value.tool_name == "echo"


def test_tool_input_base_forbids_extra_fields() -> None:
    class Model(ToolInput):
        a: int = 1

    assert Model.model_config["extra"] == "forbid"


def test_summary_is_deterministic_and_truncated() -> None:
    tool = EchoTool()

    assert tool.summarize(EchoInput(text="hi")) == '{"repeat": 1, "text": "hi"}'
    long = tool.summarize(EchoInput(text="x" * 500))
    assert len(long) == MAX_SUMMARY_CHARS
    assert long.endswith("…")


def test_default_confirmation_hooks(settings: Settings) -> None:
    tool = EchoTool()
    params = EchoInput(text="hi")

    assert tool.needs_confirmation(params, make_context(settings)) is False
    assert tool.confirmation_prompt(params) == (
        'Do you want me to run echo with {"repeat": 1, "text": "hi"}?'
    )


# --- execution --------------------------------------------------------------


async def test_execute_runs_the_tool_and_publishes_timed_events(settings: Settings) -> None:
    bus = EventBus()
    clock = FakeClock()
    context = make_context(settings, bus=bus, clock=clock)
    started = bus.subscribe(ToolCallStarted)
    finished = bus.subscribe(ToolCallFinished)
    tool = EchoTool(on_run=lambda _params: clock.advance(0.25))

    result = await tool.execute(EchoInput(text="hi", repeat=2), context)

    assert result == ToolResult.success("hi hi", data={"repeat": 2})
    start_event = await asyncio.wait_for(anext(started), WAIT)
    end_event = await asyncio.wait_for(anext(finished), WAIT)
    assert start_event.name == end_event.name == "echo"
    assert start_event.args_summary == end_event.args_summary == '{"repeat": 2, "text": "hi"}'
    assert end_event.duration_ms == pytest.approx(250.0)
    assert end_event.ok is True
    assert end_event.error is None


async def test_call_validates_then_executes(settings: Settings) -> None:
    result = await EchoTool().call({"text": "yo"}, make_context(settings))

    assert result.ok
    assert result.spoken == "yo"


async def test_call_turns_invalid_input_into_a_failed_result(settings: Settings) -> None:
    bus = EventBus()
    finished = bus.subscribe(ToolCallFinished)

    result = await EchoTool().call({}, make_context(settings, bus=bus))

    assert not result.ok
    assert "text: Field required" in result.detail
    assert result.spoken == "I couldn't run echo: the request was missing something."
    assert bus.published == 0
    finished.close()


async def test_jarvis_errors_become_failed_results_with_the_message(settings: Settings) -> None:
    bus = EventBus()
    finished = bus.subscribe(ToolCallFinished)

    result = await FlakyTool().call({"mode": "jarvis"}, make_context(settings, bus=bus))

    assert not result.ok
    assert result.detail == "provider unavailable"
    assert result.spoken == "I couldn't complete flaky."
    event = await asyncio.wait_for(anext(finished), WAIT)
    assert event.ok is False
    assert event.error == "provider unavailable"


async def test_unexpected_exceptions_become_failed_results_with_the_type(
    settings: Settings,
) -> None:
    result = await FlakyTool().call({"mode": "crash"}, make_context(settings))

    assert not result.ok
    assert result.detail == "ValueError: unexpected"
    assert "flaky" in result.spoken


async def test_cancellation_is_not_converted_into_a_result(settings: Settings) -> None:
    class Hanging(Tool[ToolInput]):
        name = "hanging"
        description = "never returns"
        Input = ToolInput

        async def run(self, params: ToolInput, context: ToolContext) -> ToolResult:
            await asyncio.sleep(WAIT * 10)
            return ToolResult.success("late")

    task = asyncio.create_task(Hanging().call({}, make_context(settings)))
    await asyncio.sleep(0)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task


def test_context_defaults(settings: Settings) -> None:
    async def never(prompt: str) -> bool:
        return False

    context = ToolContext(settings=settings, bus=EventBus(), ask_confirmation=never)

    assert context.platform is None
    assert context.memory is None
    assert context.clock.now().tzinfo is not None
