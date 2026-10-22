import asyncio
import contextlib
import threading

import pytest

from jarvis.core.events import EventBus, StateChanged
from jarvis.core.orchestrator import (
    FAREWELL,
    HELP_NOTICE,
    USER_PROMPT,
    Orchestrator,
    TerminalConfirmation,
    TerminalInput,
    TerminalOutput,
    parse_yes_no,
)
from jarvis.core.state_machine import AssistantState, StateMachine
from jarvis.utils.errors import JarvisError
from tests.core.fakes import FakeAgent, RecordingOutput, ScriptedInput

IDLE, LISTENING, THINKING, SPEAKING = AssistantState


def build(
    lines: list[str | None], agent: FakeAgent | None = None
) -> tuple[Orchestrator, FakeAgent, RecordingOutput, list[StateChanged]]:
    bus = EventBus()
    transitions: list[StateChanged] = []
    bus.subscribe_callback(StateChanged, transitions.append)
    agent = agent or FakeAgent()
    output = RecordingOutput()
    orchestrator = Orchestrator(
        agent=agent, state=StateMachine(bus), source=ScriptedInput(lines), sink=output
    )
    return orchestrator, agent, output, transitions


def path(transitions: list[StateChanged]) -> list[AssistantState]:
    return [transitions[0].old, *(t.new for t in transitions)] if transitions else []


async def test_a_turn_walks_the_state_machine_and_streams_the_answer() -> None:
    orchestrator, agent, output, transitions = build(["what time is it?", "exit"])

    await orchestrator.run()

    assert agent.requests == ["what time is it?"]
    assert output.said == ["Certainly, sir.", FAREWELL]
    assert output.notices == [HELP_NOTICE]
    assert path(transitions) == [IDLE, LISTENING, THINKING, SPEAKING, IDLE]
    assert orchestrator.state is IDLE


async def test_a_silent_turn_goes_from_thinking_back_to_idle() -> None:
    orchestrator, _, _, transitions = build([], FakeAgent(replies=[]))

    report = await orchestrator.handle_text("hello")

    assert report is not None
    assert path(transitions) == [IDLE, LISTENING, THINKING, IDLE]


@pytest.mark.parametrize("command", ["exit", "QUIT", "  bye  ", "Goodbye."])
async def test_exit_commands_end_the_session(command: str) -> None:
    orchestrator, agent, output, _ = build([command, "never read"])

    await orchestrator.run()

    assert agent.requests == []
    assert output.said == [FAREWELL]


class ListeningInput(ScriptedInput):
    """A voice-like source: it enters LISTENING itself before returning a line."""

    def __init__(self, lines: list[str | None], state: StateMachine) -> None:
        super().__init__(lines)
        self.state = state

    async def read_line(self, prompt: str) -> str | None:
        self.state.transition(LISTENING, "wake word")
        return await super().read_line(prompt)


async def test_a_source_that_already_listens_is_not_moved_to_listening_again() -> None:
    bus = EventBus()
    transitions: list[StateChanged] = []
    bus.subscribe_callback(StateChanged, transitions.append)
    state = StateMachine(bus)
    output = RecordingOutput()
    orchestrator = Orchestrator(
        agent=FakeAgent(),
        state=state,
        source=ListeningInput(["What time is it?", "Goodbye."], state),
        sink=output,
        greeting="Say hey Jarvis.",
    )

    await orchestrator.run()

    assert path(transitions) == [IDLE, LISTENING, THINKING, SPEAKING, IDLE, LISTENING, IDLE]
    assert transitions[-1].reason == "exit requested"
    assert output.notices == ["Say hey Jarvis."]
    assert output.flushes == 2  # once for the answer, once for the farewell


async def test_an_interrupted_delivery_still_ends_the_turn() -> None:
    class Interrupted(RecordingOutput):
        async def flush(self) -> bool:
            await super().flush()
            return False

    bus = EventBus()
    orchestrator = Orchestrator(
        agent=FakeAgent(), state=StateMachine(bus), source=ScriptedInput([]), sink=Interrupted()
    )

    assert await orchestrator.handle_text("hello") is not None
    assert orchestrator.state is IDLE


async def test_blank_lines_are_skipped_and_end_of_input_stops() -> None:
    orchestrator, agent, _, _ = build(["", "   ", "hi"])

    await orchestrator.run()

    assert agent.requests == ["hi"]


async def test_prompts_use_the_user_prompt() -> None:
    source = ScriptedInput(["exit"])
    orchestrator = Orchestrator(
        agent=FakeAgent(), state=StateMachine(EventBus()), source=source, sink=RecordingOutput()
    )

    await orchestrator.run()

    assert source.prompts == [USER_PROMPT]


async def test_cancel_current_turn_abandons_it_and_returns_to_idle() -> None:
    orchestrator, agent, _, transitions = build([], FakeAgent(block=True))

    turn = asyncio.create_task(orchestrator.handle_text("tell me a long story"))
    await asyncio.wait_for(agent.started.wait(), timeout=2)
    assert orchestrator.turn_in_progress

    assert orchestrator.cancel_current_turn() is True
    assert await turn is None
    assert orchestrator.state is IDLE
    assert transitions[-1].reason == "answer finished"
    assert not orchestrator.turn_in_progress


async def test_cancel_without_a_turn_is_a_no_op() -> None:
    orchestrator, _, _, _ = build([])

    assert orchestrator.cancel_current_turn() is False


async def test_cancelling_the_orchestrator_itself_propagates() -> None:
    orchestrator, agent, _, _ = build([], FakeAgent(block=True))

    outer = asyncio.create_task(orchestrator.handle_text("hello"))
    await asyncio.wait_for(agent.started.wait(), timeout=2)
    outer.cancel()

    with pytest.raises(asyncio.CancelledError):
        await outer
    assert orchestrator.state is IDLE


async def test_jarvis_errors_are_reported_and_the_session_continues() -> None:
    agent = FakeAgent(error=JarvisError("calendar exploded"))
    orchestrator, _, output, _ = build(["first", "exit"], agent)

    await orchestrator.run()

    assert any("calendar exploded" in notice for notice in output.notices)
    assert orchestrator.state is IDLE
    assert output.said == [FAREWELL]


async def test_terminal_input_reads_lines_off_the_event_loop() -> None:
    lines = iter(["hello\r\n", "exit\n", ""])
    written: list[str] = []
    source = TerminalInput(readline=lambda: next(lines), write=written.append)

    assert await source.read_line("you> ") == "hello"
    assert await source.read_line("you> ") == "exit"
    assert await source.read_line("you> ") is None
    assert written == ["you> "] * 3


async def test_terminal_input_treats_a_closed_stdin_as_end_of_input() -> None:
    def broken() -> str:
        raise ValueError("I/O operation on closed file")

    source = TerminalInput(readline=broken, write=lambda _: None)

    assert await source.read_line("> ") is None


def test_terminal_input_reader_stops_quietly_after_the_loop_closes() -> None:
    release = threading.Event()

    def slow() -> str:
        release.wait(timeout=2)
        return "late line\n"

    source = TerminalInput(readline=slow, write=lambda _: None)

    async def start_reading() -> None:
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(source.read_line("> "), timeout=0.01)

    asyncio.run(start_reading())
    release.set()
    readers = [t for t in threading.enumerate() if t.name == "jarvis-stdin"]
    for reader in readers:
        reader.join(timeout=2)
    assert not any(reader.is_alive() for reader in readers)


async def test_terminal_output_labels_the_speaker_and_notices() -> None:
    written: list[str] = []
    output = TerminalOutput("Jarvis", write=written.append)

    await output.say("At your service.")
    await output.notice("Voice mode arrives later")

    assert written == ["Jarvis> At your service.\n", "[Voice mode arrives later]\n"]


@pytest.mark.parametrize(
    ("answer", "expected"),
    [("y", True), ("Yes!", True), ("okay", True), ("n", False), ("No.", False), ("maybe", None)],
)
def test_parse_yes_no(answer: str, expected: bool | None) -> None:
    assert parse_yes_no(answer) is expected


async def test_confirmation_asks_again_on_unclear_answers() -> None:
    source = ScriptedInput(["maybe", "yes"])
    output = RecordingOutput()

    approved = await TerminalConfirmation(source, output)("Delete logs?")

    assert approved is True
    assert source.prompts == ["Delete logs? [y/n] "] * 2
    assert output.notices == ["Please answer yes or no."]


async def test_confirmation_defaults_to_no() -> None:
    unclear = TerminalConfirmation(ScriptedInput(["eh", "hm", "what"]), RecordingOutput())
    ended = TerminalConfirmation(ScriptedInput([]), RecordingOutput())

    assert await unclear("Send it?") is False
    assert await ended("Send it?") is False
