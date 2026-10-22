"""Per-stage latency of the voice path, published as ``LatencySample`` events.

Four stage names make up the voice latency budget (ARCHITECTURE.md
section 3); the dashboard and the step-14 benchmark read them from the bus:

* ``voice.wake_to_listening``: trigger received -> earcon played, LISTENING.
* ``voice.speech_end_to_transcript``: end of speech reported -> transcript.
* ``voice.transcript_to_first_token``: transcript -> first event from the LLM.
* ``voice.first_sentence_to_audio``: first sentence of an answer -> its
  first audio handed to the device.

The first and last two are measured where they happen (voice input and
output). The LLM stage spans the agent, which knows nothing about voice, so
:class:`FirstTokenProbe` wraps the ``LLMClient`` and reports the first
event after :meth:`TurnTimeline.transcript_ready`.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import Final

from jarvis.core.events import EventBus, LatencySample
from jarvis.llm.base import LLMClient, LLMEvent, Message, ToolDefinition
from jarvis.utils.logging import get_logger
from jarvis.utils.timing import Clock, Stopwatch, SystemClock

log = get_logger(__name__)

WAKE_TO_LISTENING: Final = "voice.wake_to_listening"
SPEECH_END_TO_TRANSCRIPT: Final = "voice.speech_end_to_transcript"
TRANSCRIPT_TO_LLM_START: Final = "voice.transcript_to_first_token"
FIRST_SENTENCE_TO_AUDIO: Final = "voice.first_sentence_to_audio"

VOICE_STAGES: Final = (
    WAKE_TO_LISTENING,
    SPEECH_END_TO_TRANSCRIPT,
    TRANSCRIPT_TO_LLM_START,
    FIRST_SENTENCE_TO_AUDIO,
)
"""Every stage this module names, in pipeline order."""


def publish_sample(bus: EventBus, stage: str, ms: float) -> None:
    """Log and publish one sample; shared so every voice stage reports alike."""
    log.info("latency.sample", stage=stage, ms=round(ms, 1))
    bus.publish(LatencySample(stage=stage, ms=ms))


class TurnTimeline:
    """Remembers when the current request's transcript was ready.

    Args:
        bus: Where samples are published.
        clock: Monotonic time source.
    """

    def __init__(self, bus: EventBus, *, clock: Clock | None = None) -> None:
        self._bus = bus
        self._clock = clock or SystemClock()
        self._since_transcript: Stopwatch | None = None

    def transcript_ready(self) -> None:
        """Start the transcript -> first token clock."""
        self._since_transcript = Stopwatch(self._clock)

    def first_token(self) -> None:
        """Report the first model output after a transcript; later calls are no-ops."""
        if self._since_transcript is None:
            return
        publish_sample(self._bus, TRANSCRIPT_TO_LLM_START, self._since_transcript.stop())
        self._since_transcript = None


class FirstTokenProbe(LLMClient):
    """An ``LLMClient`` decorator that tells the timeline when output starts.

    Args:
        inner: The real client; every call is forwarded unchanged.
        timeline: Receives :meth:`TurnTimeline.first_token`.
    """

    def __init__(self, inner: LLMClient, timeline: TurnTimeline) -> None:
        self._inner = inner
        self._timeline = timeline

    @property
    def model(self) -> str:
        """The wrapped client's model."""
        return self._inner.model

    @property
    def inner(self) -> LLMClient:
        """The wrapped client, for logs and tests."""
        return self._inner

    async def stream(
        self, messages: Sequence[Message], tools: Sequence[ToolDefinition] = ()
    ) -> AsyncIterator[LLMEvent]:
        """Forward the inner stream, marking its first event."""
        events = self._inner.stream(messages, tools)
        try:
            async for event in events:
                self._timeline.first_token()
                yield event
        finally:
            # Closing this generator must close the provider's too, or a
            # cancelled turn would leave its HTTP stream open.
            aclose = getattr(events, "aclose", None)
            if aclose is not None:
                await aclose()


# Design notes:
# - "First token" is the first LLMEvent of any kind (text or tool call):
#   both mean the model has started answering, and a tool-first answer is
#   just as much a response as text.
# - Only the first stream after a transcript reports, so tool rounds and
#   summary requests within one turn do not produce extra samples.
# - The end-of-speech stage is measured from the moment Python learns of
#   SPEECH_ENDED. The segmenter's silence hangover (silenceEndMs, 700 ms by
#   default) happens before that and is a tuning knob, not processing time.
