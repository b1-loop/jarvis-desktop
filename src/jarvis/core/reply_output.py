"""Speak replies, or only show them: the HUD's "speak replies" switch (ADR-0024).

:class:`SwitchableOutput` is the voice front end's ``OutputSink``. It
forwards each answer either to the speaking sink (``VoiceOutput``: TTS and
the speaker) or to a silent one (:class:`TextReplies`: the terminal/log and
the overlay's reply line). The choice is made per answer, at its first
sentence, so flipping the switch mid-answer never splits one reply between
voice and text; the expanded HUD shows every reply as text either way
(``AssistantSaid`` on the bus).
"""

from __future__ import annotations

from collections.abc import Callable

from jarvis.core.orchestrator import OutputSink
from jarvis.utils.logging import get_logger

log = get_logger(__name__)


class TextReplies:
    """``OutputSink`` that never speaks: prints, and shows the line in the overlay.

    Args:
        terminal: Prints sentences and notices (the log in the windowless app).
        on_sentence: Shows a sentence in the overlay's reply row.
    """

    def __init__(self, terminal: OutputSink, on_sentence: Callable[[str], None]) -> None:
        self._terminal = terminal
        self._on_sentence = on_sentence

    async def say(self, sentence: str) -> None:
        """Print the sentence and show it."""
        await self._terminal.say(sentence)
        self._on_sentence(sentence)

    async def notice(self, text: str) -> None:
        """Print a system message."""
        await self._terminal.notice(text)

    async def flush(self) -> bool:
        """Text is delivered as soon as it is printed."""
        return True


class SwitchableOutput:
    """Routes each answer to the speaking or the silent sink.

    Args:
        speaking: Where answers go while replies are spoken.
        silent: Where they go otherwise.
        speak: Initial state (``Settings.speak_replies``).
    """

    def __init__(self, speaking: OutputSink, silent: OutputSink, *, speak: bool = True) -> None:
        self._speaking = speaking
        self._silent = silent
        self._speak = speak
        self._answer: OutputSink | None = None

    @property
    def speak(self) -> bool:
        """Whether the next answer will be spoken."""
        return self._speak

    def set_speak(self, speak: bool) -> None:
        """Speak (True) or only show (False) answers from the next one on."""
        self._speak = speak
        log.info("reply_output.speak", speak=speak)

    async def say(self, sentence: str) -> None:
        """Send ``sentence`` to the sink chosen at the answer's first sentence."""
        if self._answer is None:
            self._answer = self._speaking if self._speak else self._silent
        await self._answer.say(sentence)

    async def notice(self, text: str) -> None:
        """System messages are never spoken; both sinks print them."""
        await self._silent.notice(text)

    async def flush(self) -> bool:
        """Wait for the current answer's sink and close the answer."""
        answer, self._answer = self._answer, None
        return await (answer or self._speaking).flush()


# Design notes:
# - An answer is the span between the first say() and the flush() that the
#   orchestrator always ends a turn with, which is why the choice of sink
#   is latched there and released by flush().
# - flush() without any say() still goes to the speaking sink: it reports
#   whether earlier speech was interrupted, which the orchestrator logs.
