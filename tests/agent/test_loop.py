import asyncio
from collections.abc import Callable

import pytest

from jarvis.agent.loop import (
    APOLOGY,
    CLOUD_HINT,
    NOTHING_DONE,
    NOTHING_TO_SAY,
    ROUND_LIMIT_DETAIL,
    SUMMARY_REQUEST,
    AgentLimits,
)
from jarvis.config import Settings
from jarvis.core.events import AssistantSaid, Event, EventBus, LatencySample
from jarvis.llm.base import (
    AssistantMessage,
    LLMError,
    SystemMessage,
    TextDelta,
    ToolResultMessage,
    UserMessage,
)
from tests.agent.fakes import DestructiveTool, EchoTool, FakeConfirmation
from tests.agent.llm_fakes import (
    Block,
    FakeMemory,
    ScriptedLLM,
    SentenceCollector,
    call,
    make_agent,
    text_round,
    tool_round,
)

MakeSettings = Callable[..., Settings]


def recorded(bus: EventBus) -> list[Event]:
    events: list[Event] = []
    bus.subscribe_callback(Event, events.append)
    return events


async def test_streams_complete_sentences_as_they_arrive(make_settings: MakeSettings) -> None:
    llm = ScriptedLLM(
        rounds=[text_round("Good evening, sir. ", "All systems are ", "running fine.")]
    )
    sink = SentenceCollector()

    report = await make_agent(make_settings(), llm).respond("status?", sink)

    assert sink.sentences == ["Good evening, sir.", "All systems are running fine."]
    assert report.reply == "Good evening, sir. All systems are running fine."
    assert report.calls == ()
    assert report.llm_rounds == 1
    assert report.error is None


async def test_first_request_has_persona_memory_question_and_tools(
    make_settings: MakeSettings,
) -> None:
    llm = ScriptedLLM(rounds=[text_round("Hello there, Bozhidar.")])
    memory = FakeMemory(context="User name: Bozhidar")
    agent = make_agent(make_settings(assistant_name="Friday"), llm, EchoTool(), memory=memory)

    await agent.respond("hi", SentenceCollector())

    system, user = llm.requests[0]
    assert isinstance(system, SystemMessage)
    assert "You are Friday" in system.text
    assert "User name: Bozhidar" in system.text
    assert user == UserMessage("hi")
    assert [tool.name for tool in llm.tools_seen[0]] == ["echo"]
    assert memory.recalled == ["hi"]


async def test_turn_is_recorded_in_memory(make_settings: MakeSettings) -> None:
    llm = ScriptedLLM(
        rounds=[tool_round(call("t1", "echo", text="hi")), text_round("I echoed it for you.")]
    )
    memory = FakeMemory()

    await make_agent(make_settings(), llm, EchoTool(), memory=memory).respond(
        "echo hi", SentenceCollector()
    )

    user_text, reply, tool_calls = memory.turns[0]
    assert (user_text, reply) == ("echo hi", "I echoed it for you.")
    assert tool_calls == [
        {"name": "echo", "arguments": {"text": "hi"}, "outcome": "executed", "ok": True}
    ]


async def test_tool_results_are_fed_back_to_the_model(make_settings: MakeSettings) -> None:
    llm = ScriptedLLM(
        rounds=[tool_round(call("t1", "echo", text="ping")), text_round("The echo said ping.")]
    )

    await make_agent(make_settings(), llm, EchoTool()).respond("echo ping", SentenceCollector())

    second = llm.requests[1]
    assert isinstance(second[-2], AssistantMessage)
    assert second[-2].tool_calls[0].id == "t1"
    results = second[-1]
    assert isinstance(results, ToolResultMessage)
    assert results.results[0].tool_call_id == "t1"
    assert results.results[0].content.startswith("ping")


async def test_destructive_tool_is_confirmed_before_running(make_settings: MakeSettings) -> None:
    tool = DestructiveTool()
    confirmation = FakeConfirmation(answers=[True])
    llm = ScriptedLLM(
        rounds=[tool_round(call("t1", "destroy", target="logs")), text_round("Deleted the logs.")]
    )

    agent = make_agent(make_settings(), llm, tool, confirmation=confirmation)
    report = await agent.respond("delete the logs", SentenceCollector())

    assert confirmation.prompts == ["Delete logs?"]
    assert len(tool.calls) == 1
    assert report.failures == ()


async def test_declined_tool_is_reported_to_the_model(make_settings: MakeSettings) -> None:
    tool = DestructiveTool()
    llm = ScriptedLLM(
        rounds=[
            tool_round(call("t1", "destroy", target="logs")),
            text_round("Okay, I left the logs alone."),
        ]
    )

    agent = make_agent(make_settings(), llm, tool, confirmation=FakeConfirmation(answers=[False]))
    report = await agent.respond("delete the logs", SentenceCollector())

    assert tool.calls == []
    result = llm.requests[1][-1]
    assert isinstance(result, ToolResultMessage)
    assert result.results[0].is_error
    assert report.failures[0].name == "destroy"


async def test_silent_ending_after_tools_asks_once_for_a_summary(
    make_settings: MakeSettings,
) -> None:
    llm = ScriptedLLM(
        rounds=[
            tool_round(call("t1", "echo", text="hi")),
            text_round(),
            text_round("I echoed hi."),
        ]
    )
    sink = SentenceCollector()

    report = await make_agent(make_settings(), llm, EchoTool()).respond("echo hi", sink)

    summary_request = llm.requests[2]
    assert summary_request[-1] == UserMessage(SUMMARY_REQUEST)
    assert isinstance(summary_request[-2], ToolResultMessage)  # empty assistant turn dropped
    assert sink.sentences == ["I echoed hi."]
    assert report.llm_rounds == 3


async def test_text_before_tools_does_not_count_as_the_summary(
    make_settings: MakeSettings,
) -> None:
    llm = ScriptedLLM(
        rounds=[
            tool_round(call("t1", "echo", text="hi"), text="One moment please."),
            text_round(),
            text_round("Done, I echoed it."),
        ]
    )
    sink = SentenceCollector()

    await make_agent(make_settings(), llm, EchoTool()).respond("echo hi", sink)

    assert sink.sentences == ["One moment please.", "Done, I echoed it."]


async def test_fallback_summary_uses_the_tools_own_words(make_settings: MakeSettings) -> None:
    llm = ScriptedLLM(
        rounds=[tool_round(call("t1", "echo", text="hi there")), text_round(), text_round()]
    )
    sink = SentenceCollector()

    await make_agent(make_settings(), llm, EchoTool()).respond("echo", sink)

    assert sink.sentences == ["hi there"]


async def test_fallback_when_every_tool_failed(make_settings: MakeSettings) -> None:
    llm = ScriptedLLM(rounds=[tool_round(call("t1", "ghost")), text_round(), text_round()])
    sink = SentenceCollector()

    await make_agent(make_settings(), llm).respond("do magic", sink)

    assert sink.sentences == [NOTHING_DONE]


async def test_summary_round_tool_calls_are_ignored(make_settings: MakeSettings) -> None:
    tool = EchoTool()
    runs: list[str] = []
    tool.on_run = lambda params: runs.append(params.text)
    llm = ScriptedLLM(
        rounds=[
            tool_round(call("t1", "echo", text="a")),
            text_round(),
            tool_round(call("t2", "echo", text="b"), text="I echoed a."),
        ]
    )

    await make_agent(make_settings(), llm, tool).respond("echo a", SentenceCollector())

    assert runs == ["a"]


async def test_no_text_and_no_tools_gets_an_honest_fallback(make_settings: MakeSettings) -> None:
    sink = SentenceCollector()

    await make_agent(make_settings(), ScriptedLLM(rounds=[text_round()])).respond("hmm", sink)

    assert sink.sentences == [NOTHING_TO_SAY]


async def test_round_limit_stops_the_chain_and_still_summarises(
    make_settings: MakeSettings,
) -> None:
    llm = ScriptedLLM(
        rounds=[
            tool_round(call("t1", "echo", text="1")),
            tool_round(call("t2", "echo", text="2")),
            text_round("I got as far as the first echo."),
        ]
    )
    agent = make_agent(make_settings(), llm, EchoTool(), limits=AgentLimits(max_tool_rounds=1))

    report = await agent.respond("echo forever", SentenceCollector())

    assert report.round_limit_hit
    declined = llm.requests[2][-2]
    assert isinstance(declined, ToolResultMessage)
    assert declined.results[0].content == ROUND_LIMIT_DETAIL
    assert llm.requests[2][-1] == UserMessage(SUMMARY_REQUEST)
    assert report.reply == "I got as far as the first echo."


async def test_llm_failure_is_spoken_with_the_local_mode_hint(make_settings: MakeSettings) -> None:
    llm = ScriptedLLM(rounds=[LLMError("could not reach the Anthropic API")])
    sink = SentenceCollector()
    memory = FakeMemory()

    agent = make_agent(make_settings(), llm, memory=memory, cloud_provider=True)
    report = await agent.respond("hello", sink)

    assert sink.sentences == [APOLOGY, "Could not reach the Anthropic API.", CLOUD_HINT]
    assert report.error == "could not reach the Anthropic API"
    assert memory.turns[0][1] == report.reply


async def test_local_provider_failure_has_no_cloud_hint(make_settings: MakeSettings) -> None:
    llm = ScriptedLLM(rounds=[LLMError("Ollama is not running.")])
    sink = SentenceCollector()

    await make_agent(make_settings(), llm).respond("hello", sink)

    assert sink.sentences == [APOLOGY, "Ollama is not running."]


async def test_failure_mid_stream_keeps_what_was_said(make_settings: MakeSettings) -> None:
    llm = ScriptedLLM(rounds=[[TextDelta("Let me look that up. "), LLMError("connection reset")]])
    sink = SentenceCollector()

    await make_agent(make_settings(), llm).respond("weather?", sink)

    assert sink.sentences[:2] == ["Let me look that up.", APOLOGY]
    assert llm.closed == 1


async def test_stream_without_a_final_event_is_an_llm_error(make_settings: MakeSettings) -> None:
    llm = ScriptedLLM(rounds=[[TextDelta("Half an answer")]])
    sink = SentenceCollector()

    report = await make_agent(make_settings(), llm).respond("hi", sink)

    assert sink.sentences[0] == "Half an answer"
    assert report.error is not None
    assert "without a final event" in report.error


async def test_memory_failures_do_not_silence_the_turn(make_settings: MakeSettings) -> None:
    memory = FakeMemory(fail_recall=True, fail_record=True)
    llm = ScriptedLLM(rounds=[text_round("Still here, sir.")])
    sink = SentenceCollector()

    await make_agent(make_settings(), llm, memory=memory).respond("hello", sink)

    assert sink.sentences == ["Still here, sir."]
    system = llm.requests[0][0]
    assert isinstance(system, SystemMessage)
    assert "remember" not in system.text


async def test_events_report_the_answer_and_latencies(make_settings: MakeSettings) -> None:
    bus = EventBus()
    events = recorded(bus)
    llm = ScriptedLLM(rounds=[text_round("Right away.")])

    await make_agent(make_settings(), llm, bus=bus).respond("go", SentenceCollector())

    said = [e for e in events if isinstance(e, AssistantSaid)]
    stages = [e.stage for e in events if isinstance(e, LatencySample)]
    assert [e.text for e in said] == ["Right away."]
    assert stages == ["agent.first_sentence", "agent.llm_round", "agent.turn"]


async def test_cancelling_the_turn_stops_streaming_and_skips_memory(
    make_settings: MakeSettings,
) -> None:
    llm = ScriptedLLM(rounds=[[TextDelta("Starting now. "), Block()]])
    memory = FakeMemory()
    sink = SentenceCollector()
    agent = make_agent(make_settings(), llm, memory=memory)

    task = asyncio.create_task(agent.respond("long story", sink))
    await asyncio.wait_for(llm.started.wait(), timeout=2)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert sink.sentences == ["Starting now."]
    assert llm.closed == 1
    assert memory.turns == []
