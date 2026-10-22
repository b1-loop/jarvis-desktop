"""One conversation from two inputs: voice turns and lines typed in the HUD.

:class:`CombinedInput` is the orchestrator's ``InputSource`` in voice mode
(ADR-0024). It waits for whichever comes first, the voice source's next
request (wake word or hotkey, then speech) or a line typed in the HUD's text
field, so both feed the same orchestrator, memory and history:

* **Typed while idle.** The pending voice read is cancelled (it was only
  waiting for the wake word, or capturing speech the user has abandoned in
  favour of typing) and the typed line becomes the next turn.
* **Typed while busy.** Typing while Jarvis thinks or speaks is the typed
  equivalent of barge-in: the running turn is cancelled, speech is
  silenced, and the typed request is handled next.
* **Typed during a confirmation.** When a tool is waiting for a yes/no, the
  typed line is the answer (:class:`CombinedAnswers`), not a new request.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections import deque
from collections.abc import Callable
from typing import Protocol

from jarvis.core.orchestrator import InputSource, Writer
from jarvis.utils.logging import get_logger

log = get_logger(__name__)


class TurnControl(Protocol):
    """The orchestrator as typed barge-in sees it."""

    @property
    def turn_in_progress(self) -> bool:
        """True while a turn (thinking or speaking) runs."""

    def cancel_current_turn(self) -> bool:
        """Abandon the running turn; True if there was one."""


class AnswerListener(Protocol):
    """Captures one spoken reply (``VoiceInput.listen_for_answer``)."""

    async def listen_for_answer(self) -> str:
        """The transcript of the reply, empty if nothing was heard."""


class CombinedInput:
    """``InputSource`` over the voice source and the HUD's text field.

    Args:
        voice: The voice source (``VoiceInput``).
        silence: Stops speech output now (``Speaker.stop``); called before a
            typed line that interrupted a turn is handed to the orchestrator.
        echo: Prints typed lines like spoken ones (``you> ...``), if given.
    """

    def __init__(
        self,
        voice: InputSource,
        *,
        silence: Callable[[], None] | None = None,
        echo: Writer | None = None,
    ) -> None:
        self._voice = voice
        self._silence = silence
        self._echo = echo
        self._typed: deque[str] = deque()
        self._typed_ready = asyncio.Event()
        self._voice_read: asyncio.Task[str | None] | None = None
        self._turns: TurnControl | None = None
        self._answer: asyncio.Future[str] | None = None
        self._interrupted = False

    @property
    def voice(self) -> InputSource:
        """The voice source this input wraps."""
        return self._voice

    def attach(self, turns: TurnControl) -> None:
        """Connect the orchestrator once it exists (it is built after the sources)."""
        self._turns = turns

    def submit(self, text: str) -> None:
        """A line typed in the HUD (already trimmed, never empty).

        Answers a pending confirmation; otherwise queues the line and, when a
        turn is running, cancels it so the line is handled next.
        """
        if self._answer is not None and not self._answer.done():
            log.info("combined_input.typed_answer")
            self._answer.set_result(text)
            return
        self._typed.append(text)
        self._typed_ready.set()
        if self._turns is not None and self._turns.turn_in_progress:
            log.info("combined_input.typed_barge_in")
            self._interrupted = True
            self._turns.cancel_current_turn()

    async def read_line(self, prompt: str) -> str | None:
        """The next request, typed or spoken, whichever arrives first."""
        if self._typed:
            await self._cancel_voice()
            return self._take_typed()
        voice = self._voice_read or asyncio.create_task(
            self._voice.read_line(prompt), name="combined-voice-read"
        )
        self._voice_read = voice
        typed = asyncio.create_task(self._typed_ready.wait(), name="combined-typed-wait")
        try:
            await asyncio.wait({voice, typed}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            typed.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await typed
        if voice.done():
            # A finished voice turn wins a tie; a typed line waits its turn.
            self._voice_read = None
            return voice.result()
        await self._cancel_voice()
        return self._take_typed()

    async def wait_for_typed_answer(self) -> str:
        """Resolve with the next typed line; used by :class:`CombinedAnswers`."""
        future: asyncio.Future[str] = asyncio.get_running_loop().create_future()
        self._answer = future
        try:
            return await future
        finally:
            self._answer = None

    async def aclose(self) -> None:
        """Cancel a pending voice read (shutdown)."""
        await self._cancel_voice()

    def _take_typed(self) -> str:
        text = self._typed.popleft()
        if not self._typed:
            self._typed_ready.clear()
        if self._interrupted:
            # The interrupted turn has finished unwinding by now (the
            # orchestrator only reads after it), so stopping the speaker
            # cannot be mistaken for the user talking over a live answer.
            self._interrupted = False
            if self._silence is not None:
                self._silence()
        if self._echo is not None:
            self._echo(f"you (typed)> {text}\n")
        log.info("combined_input.typed", chars=len(text))
        return text

    async def _cancel_voice(self) -> None:
        voice, self._voice_read = self._voice_read, None
        if voice is None or voice.done():
            return
        voice.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await voice


class CombinedAnswers:
    """``AnswerListener`` for confirmations: a spoken or a typed yes/no, first wins.

    Args:
        voice: Listens for a spoken answer.
        typed: The combined input whose next typed line counts as the answer.
    """

    def __init__(self, voice: AnswerListener, typed: CombinedInput) -> None:
        self._voice = voice
        self._typed = typed

    async def listen_for_answer(self) -> str:
        """The first answer, spoken or typed; the other wait is cancelled."""
        spoken = asyncio.create_task(self._voice.listen_for_answer(), name="answer-voice")
        typed = asyncio.create_task(self._typed.wait_for_typed_answer(), name="answer-typed")
        done, pending = await asyncio.wait({spoken, typed}, return_when=asyncio.FIRST_COMPLETED)
        for task in pending:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        winner = typed if typed in done else spoken
        return winner.result()


# Design notes:
# - The voice read is one long-lived task, reused across calls while it is
#   still waiting, and cancelled only when a typed line wins. Cancelling is
#   safe at every await of VoiceInput: the trigger wait has no side
#   effects, record_utterance switches listening off in a finally, and a
#   transcription in progress is simply discarded. The state machine may
#   be left in LISTENING, which the orchestrator accepts for the next turn.
# - Typed barge-in cancels the turn immediately but silences the speaker
#   only when the typed line is taken, after the turn has unwound to IDLE.
#   Stopping playback while the machine still says SPEAKING would make the
#   voice barge-in controller treat it as the user talking and start a
#   voice capture that nobody asked for.
# - Typed lines wait in a deque with an Event, not an asyncio.Queue: the
#   read only waits for "something was typed" and takes the line once it
#   has decided, so a line that ties with a finished voice turn is never
#   consumed and re-queued out of order; it is simply the next request.
