import asyncio
from collections.abc import Callable
from typing import Any

import pytest

from jarvis.agent.chaining import (
    MAX_RESULT_CHARS,
    CallOutcome,
    ToolCallRecord,
    ToolChain,
    TurnReport,
    call_key,
    serialized,
    to_llm_result,
)
from jarvis.agent.confirmation import ConfirmationPolicy
from jarvis.agent.registry import ToolRegistry
from jarvis.agent.tool import Tool, ToolResult
from jarvis.config import Settings
from jarvis.core.events import Event, EventBus, ToolCallFinished, ToolCallStarted
from tests.agent.fakes import DestructiveTool, EchoTool, FakeConfirmation, make_context
from tests.agent.llm_fakes import (
    AddNoteTool,
    CountingTool,
    CreateEventTool,
    FakeMemory,
    FreeSlotTool,
    LookupContactTool,
    RendezvousTool,
    ScriptedLLM,
    SendInviteTool,
    SentenceCollector,
    call,
    make_agent,
    text_round,
    tool_round,
)
from tests.fakes import FakeClock


def registry_of(*tools: Tool[Any]) -> ToolRegistry:
    registry = ToolRegistry()
    for tool in tools:
        registry.register(tool)
    return registry


def chain_for(
    make_settings: Callable[..., Settings],
    *tools: Tool[Any],
    confirmation: FakeConfirmation | None = None,
    bus: EventBus | None = None,
    **limits: int,
) -> ToolChain:
    settings = make_settings()
    context = make_context(settings, confirmation=confirmation, bus=bus)
    return ToolChain(
        registry_of(*tools),
        ConfirmationPolicy.from_settings(settings),
        context,
        clock=FakeClock(),
        **limits,
    )


async def test_results_come_back_in_request_order(make_settings: Callable[..., Settings]) -> None:
    chain = chain_for(make_settings, EchoTool())

    message = await chain.run_round(
        [call("a", "echo", text="one"), call("b", "echo", text="two")], round_index=1
    )

    assert [r.tool_call_id for r in message.results] == ["a", "b"]
    assert [r.content.split("\n")[0] for r in message.results] == ["one", "two"]
    assert not any(r.is_error for r in message.results)


async def test_independent_calls_run_in_parallel(make_settings: Callable[..., Settings]) -> None:
    tool = RendezvousTool(expected=2)
    chain = chain_for(make_settings, tool)

    # Sequential execution would deadlock: each call waits for the other.
    message = await asyncio.wait_for(
        chain.run_round(
            [call("a", "rendezvous", label="a"), call("b", "rendezvous", label="b")],
            round_index=1,
        ),
        timeout=2,
    )

    assert tool.peak == 2
    assert all(not r.is_error for r in message.results)


async def test_parallelism_is_bounded(make_settings: Callable[..., Settings]) -> None:
    tool = CountingTool()
    chain = chain_for(make_settings, tool, max_parallel=1)

    await chain.run_round([call(str(i), "counting", label=str(i)) for i in range(3)], round_index=1)

    assert tool.runs == 3
    assert tool.peak == 1


async def test_identical_calls_in_one_round_run_once(
    make_settings: Callable[..., Settings],
) -> None:
    tool = CountingTool()
    chain = chain_for(make_settings, tool)

    message = await chain.run_round(
        [call("a", "counting", label="x"), call("b", "counting", label="x")], round_index=1
    )

    assert tool.runs == 1
    assert [r.tool_call_id for r in message.results] == ["a", "b"]
    assert message.results[0].content == message.results[1].content
    assert [r.outcome for r in chain.records] == [CallOutcome.EXECUTED, CallOutcome.DUPLICATE]


async def test_identical_call_in_a_later_round_reuses_the_result(
    make_settings: Callable[..., Settings],
) -> None:
    tool = DestructiveTool()
    confirmation = FakeConfirmation(answers=[True])
    chain = chain_for(make_settings, tool, confirmation=confirmation)

    await chain.run_round([call("a", "destroy", target="tmp")], round_index=1)
    second = await chain.run_round([call("b", "destroy", target="tmp")], round_index=2)

    assert len(tool.calls) == 1
    assert len(confirmation.prompts) == 1
    assert second.results[0].tool_call_id == "b"
    assert chain.records[-1].outcome is CallOutcome.DUPLICATE
    assert chain.records[-1].duration_ms == 0.0


async def test_call_budget_refuses_extra_calls(make_settings: Callable[..., Settings]) -> None:
    tool = CountingTool()
    chain = chain_for(make_settings, tool, max_calls=2)

    message = await chain.run_round(
        [call(str(i), "counting", label=str(i)) for i in range(3)], round_index=1
    )

    assert tool.runs == 2
    assert message.results[2].is_error
    assert "budget" in message.results[2].content
    assert chain.call_budget_hit
    assert chain.records[2].outcome is CallOutcome.OVER_BUDGET


async def test_duplicates_do_not_consume_budget(make_settings: Callable[..., Settings]) -> None:
    tool = CountingTool()
    chain = chain_for(make_settings, tool, max_calls=1)

    message = await chain.run_round([call("a", "counting"), call("b", "counting")], round_index=1)

    assert not any(r.is_error for r in message.results)
    assert not chain.call_budget_hit


async def test_unknown_tool_becomes_an_error_result(
    make_settings: Callable[..., Settings],
) -> None:
    chain = chain_for(make_settings, EchoTool())

    message = await chain.run_round([call("a", "teleport", to="mars")], round_index=1)

    assert message.results[0].is_error
    assert "teleport" in message.results[0].content
    assert chain.records[0].outcome is CallOutcome.UNKNOWN_TOOL


async def test_every_call_goes_through_the_confirmation_policy(
    make_settings: Callable[..., Settings],
) -> None:
    tool = DestructiveTool()
    confirmation = FakeConfirmation(answers=[False])
    chain = chain_for(make_settings, tool, confirmation=confirmation)

    message = await chain.run_round([call("a", "destroy", target="photos")], round_index=1)

    assert confirmation.prompts == ["Delete photos?"]
    assert tool.calls == []
    assert message.results[0].is_error
    assert "declined" in message.results[0].content


async def test_parallel_confirmations_are_asked_one_at_a_time(
    make_settings: Callable[..., Settings],
) -> None:
    active = 0
    peak = 0

    async def ask(prompt: str) -> bool:
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0)
        active -= 1
        return True

    settings = make_settings()
    context = make_context(settings)
    context = type(context)(settings=settings, bus=context.bus, ask_confirmation=ask)
    tool = DestructiveTool()
    chain = ToolChain(registry_of(tool), ConfirmationPolicy.from_settings(settings), context)

    await chain.run_round(
        [call("a", "destroy", target="a"), call("b", "destroy", target="b")], round_index=1
    )

    assert len(tool.calls) == 2
    assert peak == 1


async def test_executed_tools_publish_their_events(
    make_settings: Callable[..., Settings],
) -> None:
    bus = EventBus()
    events: list[Event] = []
    bus.subscribe_callback(Event, events.append)
    chain = chain_for(make_settings, EchoTool(), bus=bus)

    await chain.run_round([call("a", "echo", text="hi"), call("b", "nope")], round_index=1)

    assert [type(e) for e in events] == [ToolCallStarted, ToolCallFinished]


async def test_decline_answers_every_call_without_running(
    make_settings: Callable[..., Settings],
) -> None:
    tool = CountingTool()
    chain = chain_for(make_settings, tool)

    message = chain.decline([call("a", "counting")], round_index=3, detail="limit reached")

    assert tool.runs == 0
    assert message.results[0].is_error
    assert message.results[0].content == "limit reached"
    assert chain.records[0].round == 3


async def test_serialized_passes_answers_through() -> None:
    confirmation = FakeConfirmation(answers=[True, False])
    ask = serialized(confirmation)

    assert await ask("one?") is True
    assert await ask("two?") is False
    assert confirmation.prompts == ["one?", "two?"]


def test_call_key_ignores_argument_order() -> None:
    first = call("a", "echo", text="x", repeat=2)
    second = call("b", "echo", repeat=2, text="x")

    assert call_key(first) == call_key(second)
    assert call_key(first) != call_key(call("c", "echo", text="y", repeat=2))


def test_llm_result_includes_data_and_is_capped() -> None:
    result = ToolResult.success("ok", detail="done", data={"id": 7})
    rendered = to_llm_result(call("a", "echo"), result)

    assert rendered.content == 'done\nData: {"id": 7}'
    assert not rendered.is_error

    huge = to_llm_result(call("a", "echo"), ToolResult.failure("no", detail="x" * 10_000))
    assert len(huge.content) == MAX_RESULT_CHARS
    assert huge.is_error


def record(name: str, outcome: CallOutcome, *, ok: bool = True, ms: float = 1.0) -> ToolCallRecord:
    return ToolCallRecord(
        round=1, name=name, arguments={}, outcome=outcome, ok=ok, duration_ms=ms, spoken=name
    )


def test_turn_report_summarises_the_calls() -> None:
    report = TurnReport(
        user_text="q",
        reply="a",
        calls=(
            record("calendar", CallOutcome.EXECUTED, ms=10.0),
            record("calendar", CallOutcome.DUPLICATE, ms=0.0),
            record("mail", CallOutcome.EXECUTED, ok=False, ms=5.0),
            record("ghost", CallOutcome.UNKNOWN_TOOL, ok=False, ms=0.0),
        ),
    )

    assert report.tools_used == ("calendar", "mail")
    assert [c.name for c in report.failures] == ["mail", "ghost"]
    assert report.tool_ms == pytest.approx(15.0)
    assert report.memory_payload()[0] == {
        "name": "calendar",
        "arguments": {},
        "outcome": "executed",
        "ok": True,
    }


async def test_one_request_chains_several_tools_across_rounds(
    make_settings: Callable[..., Settings],
) -> None:
    """ "Book Anna on Thursday, send her an invite and add the agenda to my notes." """
    create, invite, notes = CreateEventTool(), SendInviteTool(), AddNoteTool()
    llm = ScriptedLLM(
        rounds=[
            tool_round(
                call("c1", "lookup_contact", name="Anna"),
                call("c2", "find_free_slot", day="Thursday"),
                text="Let me check Thursday.",
            ),
            tool_round(
                call("c3", "create_event", title="Sync with Anna", day="Thursday", start="10:00")
            ),
            tool_round(
                call("c4", "send_invite", event_id="evt-1", email="anna@example.com"),
                call("c5", "add_note", text="Agenda: roadmap, hiring"),
            ),
            text_round(
                "Booked Anna for Thursday at ten, ", "sent the invite and saved the agenda."
            ),
        ]
    )
    confirmation = FakeConfirmation(answers=[True, True])
    memory = FakeMemory()
    sink = SentenceCollector()
    agent = make_agent(
        make_settings(),
        llm,
        LookupContactTool(),
        FreeSlotTool(),
        create,
        invite,
        notes,
        confirmation=confirmation,
        memory=memory,
    )

    report = await agent.respond("Book Anna on Thursday and send her an invite", sink)

    assert report.tools_used == (
        "lookup_contact",
        "find_free_slot",
        "create_event",
        "send_invite",
        "add_note",
    )
    assert [c.round for c in report.calls] == [1, 1, 2, 3, 3]
    assert report.failures == ()
    assert report.llm_rounds == 4
    assert len(confirmation.prompts) == 2
    assert create.created[0].start == "10:00"
    assert invite.sent[0].email == "anna@example.com"
    assert notes.notes == ["Agenda: roadmap, hiring"]
    assert sink.sentences == [
        "Let me check Thursday.",
        "Booked Anna for Thursday at ten, sent the invite and saved the agenda.",
    ]
    assert [m["name"] for m in memory.turns[0][2]] == list(report.tools_used)
