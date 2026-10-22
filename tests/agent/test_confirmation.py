import asyncio
from collections.abc import Callable

import pytest

from jarvis.agent.confirmation import (
    DECLINED_SPOKEN,
    ConfirmationPolicy,
    NotNeeded,
    Required,
)
from jarvis.agent.tool import ToolCategory, ToolContext, ToolResult
from jarvis.config import Settings
from jarvis.core.events import (
    ConfirmationRequested,
    ConfirmationResolved,
    Event,
    ToolCallFinished,
    ToolCallStarted,
)
from tests.agent.fakes import (
    DestructiveInput,
    DestructiveTool,
    EchoInput,
    EchoTool,
    FakeConfirmation,
    SendInput,
    SendTool,
    make_context,
)

WAIT = 1.0


@pytest.fixture
def settings(make_settings: Callable[..., Settings]) -> Settings:
    return make_settings()


@pytest.fixture
def context(settings: Settings) -> ToolContext:
    return make_context(settings)


# --- decide -----------------------------------------------------------------


def test_safe_tool_needs_no_confirmation(context: ToolContext) -> None:
    decision = ConfirmationPolicy().decide(EchoTool(), EchoInput(text="hi"), context)

    assert isinstance(decision, NotNeeded)
    assert decision.reason == "no rule requires confirmation"


def test_requires_confirmation_flag_always_asks(context: ToolContext) -> None:
    decision = ConfirmationPolicy().decide(DestructiveTool(), DestructiveInput(), context)

    assert decision == Required("Delete everything?", "the tool requires confirmation")


def test_always_confirm_categories_ask_even_without_the_flag(context: ToolContext) -> None:
    policy = ConfirmationPolicy(always_confirm_categories=[ToolCategory.COMMUNICATION])

    decision = policy.decide(SendTool(), SendInput(to="bob"), context)

    assert isinstance(decision, Required)
    assert decision.reason == "the communication category always confirms"
    assert decision.prompt == 'Do you want me to run send with {"to": "bob"}?'


def test_category_rule_does_not_affect_other_categories(context: ToolContext) -> None:
    policy = ConfirmationPolicy(always_confirm_categories=[ToolCategory.COMMUNICATION])

    assert isinstance(policy.decide(EchoTool(), EchoInput(text="hi"), context), NotNeeded)


def test_trusted_mode_never_asks(context: ToolContext) -> None:
    policy = ConfirmationPolicy(
        confirm_destructive=False, always_confirm_categories=[ToolCategory.FILES]
    )

    decision = policy.decide(DestructiveTool(), DestructiveInput(), context)

    assert policy.trusted_mode
    assert decision == NotNeeded("trusted mode is on")


def test_from_settings_reads_the_trusted_mode_flag(
    make_settings: Callable[..., Settings],
) -> None:
    assert not ConfirmationPolicy.from_settings(make_settings()).trusted_mode
    assert ConfirmationPolicy.from_settings(make_settings(confirm_destructive=False)).trusted_mode
    policy = ConfirmationPolicy.from_settings(
        make_settings(), always_confirm_categories=[ToolCategory.SMART_HOME]
    )
    assert policy._always_confirm == frozenset({ToolCategory.SMART_HOME})


# --- execute_with_policy ----------------------------------------------------


async def test_safe_tool_runs_without_asking(settings: Settings) -> None:
    asker = FakeConfirmation(answers=[])
    context = make_context(settings, confirmation=asker)

    result = await ConfirmationPolicy().execute_with_policy(EchoTool(), {"text": "hi"}, context)

    assert result.ok
    assert result.spoken == "hi"
    assert asker.prompts == []


async def test_approved_call_runs_and_publishes_both_events(settings: Settings) -> None:
    asker = FakeConfirmation(answers=[True])
    context = make_context(settings, confirmation=asker)
    events = context.bus.subscribe(Event)
    tool = DestructiveTool()

    result = await ConfirmationPolicy().execute_with_policy(tool, {"target": "logs"}, context)

    assert result.ok
    assert tool.calls == [DestructiveInput(target="logs")]
    assert asker.prompts == ["Delete logs?"]
    kinds = [type(await asyncio.wait_for(anext(events), WAIT)) for _ in range(4)]
    assert kinds == [ConfirmationRequested, ConfirmationResolved, ToolCallStarted, ToolCallFinished]


async def test_declined_call_does_not_run(settings: Settings) -> None:
    asker = FakeConfirmation(answers=[False])
    context = make_context(settings, confirmation=asker)
    resolved = context.bus.subscribe(ConfirmationResolved)
    tool = DestructiveTool()

    result = await ConfirmationPolicy().execute_with_policy(tool, {}, context)

    assert not result.ok
    assert result.spoken == DECLINED_SPOKEN
    assert "declined" in result.detail
    assert tool.calls == []
    event = await asyncio.wait_for(anext(resolved), WAIT)
    assert (event.tool_name, event.approved) == ("destroy", False)
    assert context.bus.published == 2  # requested + resolved, no tool-call events


async def test_invalid_input_is_reported_before_asking(settings: Settings) -> None:
    asker = FakeConfirmation(answers=[])
    context = make_context(settings, confirmation=asker)

    result = await ConfirmationPolicy().execute_with_policy(
        DestructiveTool(), {"target": 1, "extra": True}, context
    )

    assert not result.ok
    assert "extra: Extra inputs are not permitted" in result.detail
    assert asker.prompts == []
    assert context.bus.published == 0


async def test_trusted_mode_runs_destructive_tools_silently(settings: Settings) -> None:
    asker = FakeConfirmation(answers=[])
    context = make_context(settings, confirmation=asker)
    tool = DestructiveTool()

    result = await ConfirmationPolicy(confirm_destructive=False).execute_with_policy(
        tool, {}, context
    )

    assert result.ok
    assert len(tool.calls) == 1
    assert asker.prompts == []


async def test_tool_failures_pass_through_after_approval(settings: Settings) -> None:
    class ExplodingDestructive(DestructiveTool):
        async def run(self, params: DestructiveInput, context: ToolContext) -> ToolResult:
            raise RuntimeError("disk on fire")

    context = make_context(settings, confirmation=FakeConfirmation(answers=[True]))

    result = await ConfirmationPolicy().execute_with_policy(ExplodingDestructive(), {}, context)

    assert not result.ok
    assert result.detail == "RuntimeError: disk on fire"
