"""Helpers for turning provider streams into pipeline-ready pieces.

Two problems are the same for every provider: text arrives in fragments but
TTS wants whole sentences, and tool arguments arrive as JSON fragments that
only mean something once the block is complete. Both are solved here, once,
and tested without any network.
"""

from __future__ import annotations

import json
import re
from typing import Any, Final

from jarvis.llm.base import LLMError, ToolCall

# The curly quotes are deliberate: models emit them, and a sentence that ends
# in one must still split.
_SENTENCE_END: Final = re.compile(r"[.!?…][\"'”’)\]]*\s+|\n+")  # noqa: RUF001
"""Terminal punctuation (plus closing quotes) followed by whitespace, or a line break."""

DEFAULT_MIN_SENTENCE_CHARS: Final = 12
"""Shorter candidates are merged with the next sentence; see the design notes."""


class SentenceSplitter:
    """Accumulates text deltas and releases complete sentences.

    The pipeline feeds every :class:`~jarvis.llm.base.TextDelta` through
    ``push`` and hands each returned sentence to TTS immediately, so speech
    starts while the model is still writing. ``flush`` returns whatever is
    left when the turn ends.
    """

    def __init__(self, *, min_chars: int = DEFAULT_MIN_SENTENCE_CHARS) -> None:
        self._min_chars = min_chars
        self._buffer = ""

    @property
    def pending(self) -> str:
        """Text received but not yet released as a sentence."""
        return self._buffer

    def push(self, delta: str) -> list[str]:
        """Add ``delta`` and return every sentence it completed, in order."""
        self._buffer += delta
        sentences: list[str] = []
        while (boundary := self._next_boundary()) is not None:
            sentence = self._buffer[:boundary].strip()
            self._buffer = self._buffer[boundary:]
            if sentence:
                sentences.append(sentence)
        return sentences

    def flush(self) -> str | None:
        """Return the unfinished tail (or ``None`` if empty) and reset."""
        tail, self._buffer = self._buffer.strip(), ""
        return tail or None

    def _next_boundary(self) -> int | None:
        search_from = 0
        while (match := _SENTENCE_END.search(self._buffer, search_from)) is not None:
            if len(self._buffer[: match.start()].strip()) >= self._min_chars:
                return match.end()
            search_from = match.end()
        return None


class ToolCallAccumulator:
    """Collects streamed tool-call fragments, one per content-block index.

    Providers stream a tool call as a start (id and name), any number of
    JSON argument fragments and a stop; this class turns that into one
    :class:`~jarvis.llm.base.ToolCall` when the block closes.
    """

    def __init__(self) -> None:
        self._pending: dict[int, _PendingCall] = {}

    def start(self, index: int, call_id: str, name: str) -> None:
        """Begin collecting the block at ``index``."""
        self._pending[index] = _PendingCall(call_id=call_id, name=name)

    def append(self, index: int, partial_json: str) -> None:
        """Add an argument fragment to the block at ``index``.

        Fragments for an index that was never started are ignored: they can
        only come from block types this accumulator does not track.
        """
        pending = self._pending.get(index)
        if pending is not None:
            pending.fragments.append(partial_json)

    def is_tracking(self, index: int) -> bool:
        """True if ``index`` is a tool-call block being collected."""
        return index in self._pending

    def finish(self, index: int) -> ToolCall:
        """Close the block at ``index`` and parse its arguments.

        Raises:
            KeyError: If ``index`` was never started.
            LLMError: If the arguments are not a JSON object.
        """
        pending = self._pending.pop(index)
        return ToolCall(id=pending.call_id, name=pending.name, arguments=pending.parse())


class _PendingCall:
    def __init__(self, *, call_id: str, name: str) -> None:
        self.call_id = call_id
        self.name = name
        self.fragments: list[str] = []

    def parse(self) -> dict[str, Any]:
        raw = "".join(self.fragments).strip()
        if not raw:
            return {}
        try:
            arguments = json.loads(raw)
        except json.JSONDecodeError as error:
            raise LLMError(
                f"tool call '{self.name}' ({self.call_id}) sent arguments that are not valid JSON"
            ) from error
        if not isinstance(arguments, dict):
            raise LLMError(f"tool call '{self.name}' ({self.call_id}) sent a non-object argument")
        return arguments


# Design notes:
# - A boundary needs whitespace after the punctuation, so "3.14" and
#   "jarvis.local" never split; a newline is always a boundary because list
#   items and paragraphs read better as separate utterances.
# - min_chars merges fragments such as "Yes." or "e.g." with what follows.
#   An occasional split after an abbreviation costs a short pause; holding
#   every sentence back to second-guess punctuation would cost latency on
#   every single sentence, which is the wrong trade for a voice assistant.
# - Arguments must be an object because every tool validates them with a
#   pydantic model; an empty argument block is a valid call with no fields.
