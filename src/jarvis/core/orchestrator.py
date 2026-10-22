"""Owns the conversation turn lifecycle and the assistant state machine.

The orchestrator reads a request from an :class:`InputSource`, drives the
state machine (``IDLE -> LISTENING -> THINKING -> SPEAKING -> IDLE``), runs
the agent's turn as its own task, and streams every sentence to an
:class:`OutputSink`. Text mode plugs in :class:`TerminalInput` and
:class:`TerminalOutput`; the voice pipeline plugs in a microphone/STT
source and a TTS sink (``core/voice_io.py``) and reuses everything else,
including :meth:`Orchestrator.cancel_current_turn` for barge-in
(``core/interruption.py``).
"""

from __future__ import annotations

import asyncio
import sys
import threading
from collections.abc import Callable
from typing import Final, Protocol

from jarvis.agent.chaining import TurnReport
from jarvis.agent.loop import SentenceSink
from jarvis.core.state_machine import AssistantState, StateMachine
from jarvis.utils.errors import JarvisError
from jarvis.utils.logging import get_logger

log = get_logger(__name__)

EXIT_COMMANDS: Final = frozenset({"exit", "quit", "bye", "goodbye"})
"""Typed alone, these end the session instead of going to the model."""

USER_PROMPT: Final = "you> "
FAREWELL: Final = "Goodbye."
HELP_NOTICE: Final = "Type a request and press Enter. Type 'exit' or press Ctrl+C to quit."
CONFIRM_SUFFIX: Final = " [y/n] "
DEFAULT_CONFIRM_ATTEMPTS: Final = 3
_YES: Final = frozenset({"y", "yes", "yeah", "yep", "sure", "ok", "okay", "do it"})
_NO: Final = frozenset({"n", "no", "nope", "don't", "do not", "cancel", "stop"})

LineReader = Callable[[], str]
"""Blocking read of one line; ``""`` means end of input (``io.TextIOBase.readline``)."""

Writer = Callable[[str], None]
"""Writes text to the terminal without adding a newline."""


class InputSource(Protocol):
    """Where requests (and confirmation answers) come from."""

    async def read_line(self, prompt: str) -> str | None:
        """Show ``prompt`` and return the next line, or ``None`` at end of input."""


class OutputSink(Protocol):
    """Where the assistant's words and system notices go."""

    async def say(self, sentence: str) -> None:
        """Deliver one sentence of the assistant's answer.

        May return before the sentence has been delivered (speech is
        synthesised and played in the background); :meth:`flush` waits.
        """

    async def notice(self, text: str) -> None:
        """Show a system message that is not the assistant speaking."""

    async def flush(self) -> bool:
        """Wait until everything passed to :meth:`say` has been delivered.

        Returns:
            True if it was delivered in full, False if the user interrupted.
        """


class TurnRunner(Protocol):
    """The agent as the orchestrator sees it."""

    async def respond(self, user_text: str, sink: SentenceSink) -> TurnReport:
        """Answer one request, streaming sentences into ``sink``."""


def _stdout_write(text: str) -> None:
    sys.stdout.write(text)
    sys.stdout.flush()


class TerminalInput:
    """Reads stdin lines on a daemon thread so the event loop never blocks.

    A plain ``asyncio.to_thread(input)`` would leave an executor thread stuck
    in ``input()`` at shutdown, and ``asyncio.run`` waits for executor
    threads before returning, so Ctrl+C or "exit" would hang until the user
    pressed Enter again. A daemon thread dies with the process.

    Args:
        readline: Blocking line reader; ``sys.stdin.readline`` by default.
        write: Where the prompt is printed.
    """

    def __init__(self, *, readline: LineReader | None = None, write: Writer | None = None) -> None:
        # Bound now, not per call: the reader must keep reading the stream
        # that was stdin when the session started.
        self._readline = readline or sys.stdin.readline
        self._write = write or _stdout_write
        self._lines: asyncio.Queue[str | None] = asyncio.Queue()
        self._thread: threading.Thread | None = None

    async def read_line(self, prompt: str) -> str | None:
        """Print ``prompt`` and wait for the next line without blocking the loop."""
        self._write(prompt)
        if self._thread is None:
            self._start_reader(asyncio.get_running_loop())
        return await self._lines.get()

    def _start_reader(self, loop: asyncio.AbstractEventLoop) -> None:
        self._thread = threading.Thread(
            target=self._pump, args=(loop,), name="jarvis-stdin", daemon=True
        )
        self._thread.start()

    def _pump(self, loop: asyncio.AbstractEventLoop) -> None:
        """Reader thread: forward each line to the loop; ``None`` marks the end."""
        while True:
            try:
                raw = self._readline()
            except (OSError, ValueError) as error:  # stdin closed underneath us
                log.warning("terminal.stdin.closed", error=repr(error))
                raw = ""
            line = raw.rstrip("\r\n") if raw else None
            try:
                loop.call_soon_threadsafe(self._lines.put_nowait, line)
            except RuntimeError:
                return  # the loop has shut down; nobody is listening any more
            if line is None:
                return


class TerminalOutput:
    """Prints the assistant's sentences as they stream, one per line.

    Args:
        speaker: Label in front of each sentence, normally the assistant's name.
        write: Where text goes; ``sys.stdout`` by default.
    """

    def __init__(self, speaker: str, *, write: Writer | None = None) -> None:
        self._speaker = speaker
        self._write = write or _stdout_write

    async def say(self, sentence: str) -> None:
        """Print one sentence of the answer."""
        self._write(f"{self._speaker}> {sentence}\n")

    async def notice(self, text: str) -> None:
        """Print a system message in brackets so it is not mistaken for the assistant."""
        self._write(f"[{text}]\n")

    async def flush(self) -> bool:
        """Printing is synchronous, so everything is already delivered."""
        return True


def parse_yes_no(answer: str) -> bool | None:
    """``"yes"`` -> True, ``"no"`` -> False, anything unclear -> None."""
    normalised = answer.strip().lower().rstrip(".!")
    if normalised in _YES:
        return True
    if normalised in _NO:
        return False
    return None


class TerminalConfirmation:
    """``ask_confirmation`` for text mode: a typed y/n question.

    Unclear answers are asked again a few times; end of input or repeated
    unclear answers count as "no", because the safe default for a
    destructive action is not to do it.
    """

    def __init__(
        self, source: InputSource, sink: OutputSink, *, attempts: int = DEFAULT_CONFIRM_ATTEMPTS
    ) -> None:
        self._source = source
        self._sink = sink
        self._attempts = attempts

    async def __call__(self, prompt: str) -> bool:
        """Ask ``prompt`` until the user answers yes or no."""
        for _ in range(self._attempts):
            answer = await self._source.read_line(prompt + CONFIRM_SUFFIX)
            if answer is None:
                return False
            verdict = parse_yes_no(answer)
            if verdict is not None:
                return verdict
            await self._sink.notice("Please answer yes or no.")
        log.info("confirmation.unclear", attempts=self._attempts)
        return False


class Orchestrator:
    """Runs conversation turns and keeps the state machine truthful.

    Args:
        agent: Answers one request at a time.
        state: The assistant state machine; must start in IDLE.
        source: Where requests come from.
        sink: Where answers go.
        greeting: Notice shown when the session starts (how to talk to Jarvis).
    """

    def __init__(
        self,
        *,
        agent: TurnRunner,
        state: StateMachine,
        source: InputSource,
        sink: OutputSink,
        greeting: str = HELP_NOTICE,
    ) -> None:
        self._agent = agent
        self._state = state
        self._source = source
        self._sink = sink
        self._greeting = greeting
        self._current: asyncio.Task[TurnReport] | None = None

    @property
    def state(self) -> AssistantState:
        """What the assistant is doing right now."""
        return self._state.state

    @property
    def turn_in_progress(self) -> bool:
        """True while a turn task is running."""
        return self._current is not None and not self._current.done()

    async def run(self) -> None:
        """Converse until "exit", end of input or cancellation (Ctrl+C)."""
        log.info("orchestrator.started")
        await self._sink.notice(self._greeting)
        while (text := await self._next_request()) is not None:
            await self.handle_text(text)
        await self._sink.say(FAREWELL)
        await self._sink.flush()
        log.info("orchestrator.stopped")

    async def handle_text(self, text: str) -> TurnReport | None:
        """Run one turn for ``text``; ``None`` if it was cancelled or failed.

        A voice source has already moved the machine to LISTENING while it
        captured the request; typed input passes through LISTENING here.
        """
        if self._state.state is not AssistantState.LISTENING:
            self._state.transition(AssistantState.LISTENING, "typed input")
        self._state.transition(AssistantState.THINKING, "request received")
        self._current = asyncio.create_task(self._respond(text), name="turn")
        try:
            return await self._current
        except asyncio.CancelledError:
            if _caller_is_cancelled():
                raise
            log.info("orchestrator.turn.cancelled")
            return None
        except JarvisError as error:
            log.exception("orchestrator.turn.failed")
            await self._sink.notice(f"That request failed: {error}")
            return None
        finally:
            self._current = None
            self._return_to_idle()

    def cancel_current_turn(self) -> bool:
        """Abandon the running turn (barge-in); True if there was one to cancel."""
        if self._current is None or self._current.done():
            return False
        log.info("orchestrator.turn.cancel_requested")
        self._current.cancel()
        return True

    async def _respond(self, text: str) -> TurnReport:
        """The turn task: answer, then wait until the answer has been delivered.

        Waiting for the sink inside the task keeps SPEAKING truthful (the
        machine returns to IDLE only once the last word was played) and lets
        barge-in cancel a turn that is only playing its final sentence.
        """
        report = await self._agent.respond(text, self._speak)
        if not await self._sink.flush():
            log.info("orchestrator.turn.output_interrupted")
        return report

    async def _next_request(self) -> str | None:
        while (line := await self._source.read_line(USER_PROMPT)) is not None:
            text = line.strip()
            if text.lower().rstrip(".!?") in EXIT_COMMANDS:
                self._leave_listening("exit requested")
                return None
            if text:
                return text
        return None

    def _leave_listening(self, reason: str) -> None:
        if self._state.state is AssistantState.LISTENING:
            self._state.transition(AssistantState.IDLE, reason)

    async def _speak(self, sentence: str) -> None:
        if self._state.state is AssistantState.THINKING:
            self._state.transition(AssistantState.SPEAKING, "answer ready")
        await self._sink.say(sentence)

    def _return_to_idle(self) -> None:
        if self._state.state is AssistantState.SPEAKING:
            self._state.transition(AssistantState.IDLE, "answer finished")
        elif self._state.state is AssistantState.THINKING:
            self._state.transition(AssistantState.IDLE, "nothing to say")


def _caller_is_cancelled() -> bool:
    """Whether the task awaiting the turn is itself being cancelled (shutdown)."""
    task = asyncio.current_task()
    return task is not None and task.cancelling() > 0


# Design notes:
# - Text mode passes through LISTENING on the way to THINKING because the
#   transition table (core/state_machine.py) has no IDLE -> THINKING edge,
#   and that is right: a typed line is the text equivalent of "the user
#   finished speaking". Observers see the same sequence in both modes.
# - The turn runs as its own task so ``cancel_current_turn`` can stop it
#   without stopping the orchestrator. Cancellation of the orchestrator
#   itself (Ctrl+C) is told apart with ``Task.cancelling()`` and re-raised;
#   only a turn cancellation is absorbed, as the expected barge-in outcome.
# - A JarvisError escaping the agent is reported and the session goes on;
#   anything else is a bug and propagates so it is seen, not hidden.
# - Confirmation reads from the same InputSource as requests. That is safe
#   because turns are sequential: while a turn runs, nobody else reads.
# - Unclear confirmation answers default to "no" after a few tries: the
#   policy only asks for actions where doing the wrong thing is worse than
#   doing nothing.
# - Exit words are compared without trailing punctuation because speech
#   recognisers write "Goodbye." where a typist writes "goodbye".
# - OutputSink.flush is part of the turn, not an afterthought: a voice sink
#   returns from say() at once and plays in the background (sentence
#   pipelining, ADR-0019), so without the flush the machine would report
#   IDLE while Jarvis was still talking.
