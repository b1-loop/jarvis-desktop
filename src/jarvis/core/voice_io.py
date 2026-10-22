"""Speech in and speech out: the voice ``InputSource`` and ``OutputSink``.

:class:`VoiceInput` waits for a trigger (wake word or hotkey), plays an
earcon, moves the state machine to LISTENING, records one utterance and
transcribes it. :class:`VoiceOutput` speaks the agent's sentences as they
stream: each sentence is synthesised by the TTS provider as soon as the
agent produces it, concurrently with the sentences before it and while
they play, and :meth:`VoiceOutput.flush` reports
whether the user interrupted. :class:`VoiceConfirmation` asks a yes/no
question out loud. The orchestrator drives all three exactly as it drives
the terminal in text mode (ADR-0014, ADR-0019).
"""

from __future__ import annotations

import asyncio
import math
from collections.abc import AsyncIterator, Awaitable, Callable, Coroutine
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Final, Protocol, runtime_checkable

import numpy as np

from jarvis.audio.engine import Samples, Utterance
from jarvis.core.events import EventBus, TranscriptReady, WakeWordHeard
from jarvis.core.orchestrator import DEFAULT_CONFIRM_ATTEMPTS, OutputSink, Writer, parse_yes_no
from jarvis.core.state_machine import AssistantState, StateMachine
from jarvis.core.voice_latency import (
    FIRST_SENTENCE_TO_AUDIO,
    SPEECH_END_TO_TRANSCRIPT,
    WAKE_TO_LISTENING,
    TurnTimeline,
    publish_sample,
)
from jarvis.core.voice_triggers import TriggerHit, TurnTrigger
from jarvis.stt.base import STT, TranscriptionError
from jarvis.tts.base import TTS, SynthesisError
from jarvis.utils.logging import get_logger
from jarvis.utils.timing import Clock, Stopwatch, SystemClock

log = get_logger(__name__)

DIDNT_CATCH: Final = "Sorry, I didn't catch that."
PLEASE_ANSWER: Final = "Please answer yes or no."
TTS_FAILED_NOTICE: Final = "Speech output failed ({error}); answers are printed here instead"
DEFAULT_LISTEN_TIMEOUT: Final = 6.0
DEFAULT_MIN_TRANSCRIPT_CHARS: Final = 2
DEFAULT_LOOKAHEAD: Final = 2
"""Sentences queued, and synthesised, behind the one playing; bounds work lost to a barge-in."""
CHUNK_POLL_SECONDS: Final = 0.02
"""How often a playing sentence checks for a native stop while waiting for audio."""

EARCON_SECONDS: Final = 0.08
EARCON_HZ: Final = 880.0
EARCON_AMPLITUDE: Final = 0.2

Announcer = Callable[[str], Awaitable[None]]
"""Speaks a short system phrase in full ("Sorry, I didn't catch that.")."""

Cue = Callable[[], Awaitable[None]]
"""Plays the "I'm listening" earcon."""


def make_earcon(
    sample_rate: int,
    *,
    seconds: float = EARCON_SECONDS,
    frequency_hz: float = EARCON_HZ,
    amplitude: float = EARCON_AMPLITUDE,
) -> Samples:
    """A short sine blip with a Hann envelope, generated instead of shipped as a file.

    The envelope starts and ends at zero, so the tone has no click.
    """
    count = max(1, round(sample_rate * seconds))
    t = np.arange(count, dtype=np.float32) / np.float32(sample_rate)
    tone = np.sin(np.float32(2.0 * math.pi * frequency_hz) * t)
    return (tone * np.hanning(count).astype(np.float32) * np.float32(amplitude)).astype(np.float32)


class SpeechPlayer(Protocol):
    """The slice of ``audio.engine.Speaker`` the voice path uses."""

    @property
    def stop_count(self) -> int:
        """Stops so far; a change during playback means "interrupted"."""

    async def play(self, chunks: AsyncIterator[Samples]) -> bool:
        """Play ``chunks``; False if a stop interrupted them."""

    def stop(self) -> None:
        """Silence playback now."""


async def _one_chunk(samples: Samples) -> AsyncIterator[Samples]:
    yield samples


class Earcon:
    """Plays a generated tone through the speaker and waits for it to finish.

    Waiting matters: listening starts after the tone, so it is not captured.
    """

    def __init__(self, speaker: SpeechPlayer, samples: Samples) -> None:
        self._speaker = speaker
        self._samples = samples

    async def __call__(self) -> None:
        """Play the tone once."""
        await self._speaker.play(_one_chunk(self._samples))


class CaptureEngine(Protocol):
    """The slice of ``audio.engine.AudioEngine`` the voice input uses."""

    def set_listening(self, listening: bool) -> None:
        """Start or stop retaining audio."""

    def discard_pending_events(self) -> int:
        """Drop stale events."""

    async def record_utterance(
        self, *, start_timeout: float | None = None, speech_in_progress: bool = False
    ) -> Utterance:
        """Record one utterance."""


class VoiceInput:
    """``InputSource`` over the microphone: trigger, earcon, capture, STT.

    Args:
        engine: Captures audio and reports speech start/end.
        stt: Transcribes; providers run inference off the event loop.
        trigger: Wake word, hotkey, or both (:class:`~jarvis.core.voice_triggers.AnyTrigger`).
        state: The assistant state machine; this source enters LISTENING.
        bus: Receives ``WakeWordHeard``, ``TranscriptReady`` and latency samples.
        timeline: Starts the transcript -> first token clock.
        earcon: Played after a trigger, before listening.
        announce: Speaks "didn't catch that".
        clock: Monotonic time for latency.
        listen_timeout: Seconds to wait for speech after a trigger.
        min_chars: Fewer letters/digits than this count as nothing heard.
        echo: Prints what was understood (``you> ...``), if given.
    """

    def __init__(
        self,
        *,
        engine: CaptureEngine,
        stt: STT,
        trigger: TurnTrigger,
        state: StateMachine,
        bus: EventBus,
        timeline: TurnTimeline,
        earcon: Cue,
        announce: Announcer,
        clock: Clock | None = None,
        listen_timeout: float = DEFAULT_LISTEN_TIMEOUT,
        min_chars: int = DEFAULT_MIN_TRANSCRIPT_CHARS,
        echo: Writer | None = None,
    ) -> None:
        self._engine = engine
        self._stt = stt
        self._trigger = trigger
        self._state = state
        self._bus = bus
        self._timeline = timeline
        self._earcon = earcon
        self._announce = announce
        self._clock = clock or SystemClock()
        self._listen_timeout = listen_timeout
        self._min_chars = min_chars
        self._echo = echo
        self._barge_in = False
        self._triggered: TriggerHit | None = None
        self.muted = False
        """While True, triggers are ignored (the tray's "Mute microphone")."""

    async def read_line(self, prompt: str) -> str | None:
        """Wait for the next spoken request; never returns ``None`` (end is Ctrl+C).

        ``prompt`` is the terminal's ``you>`` and has no spoken equivalent.
        """
        del prompt
        while (text := await self._next_request()) is None:
            pass
        return text

    def begin_barge_in_capture(self) -> None:
        """The user is talking over Jarvis: keep their audio, skip the trigger next time."""
        self._engine.discard_pending_events()
        self._engine.set_listening(True)
        self._barge_in = True

    def begin_triggered_capture(self, hit: TriggerHit) -> None:
        """The wake word or hotkey interrupted Jarvis: the next request starts with the earcon.

        Unlike :meth:`begin_barge_in_capture` nothing is retained yet: the
        user said the wake phrase (or pressed the key) and will speak the
        request after it, exactly as when starting a turn from idle.
        """
        self._engine.discard_pending_events()
        self._triggered = hit

    async def listen_for_answer(self) -> str:
        """Capture one short reply without a trigger (confirmation questions)."""
        self._engine.discard_pending_events()
        utterance = await self._engine.record_utterance(start_timeout=self._listen_timeout)
        return await self._transcribe(utterance)

    async def _next_request(self) -> str | None:
        barge_in, self._barge_in = self._barge_in, False
        triggered, self._triggered = self._triggered, None
        if triggered is not None:
            await self._acknowledge(triggered, enter_listening=False)
        elif not barge_in:
            await self._wait_for_trigger()
        utterance = await self._engine.record_utterance(
            start_timeout=None if barge_in else self._listen_timeout,
            speech_in_progress=barge_in,
        )
        text = await self._transcribe(utterance)
        if self._usable(text):
            self._timeline.transcript_ready()
            if self._echo is not None:
                self._echo(f"you> {text}\n")
            return text
        self._state.transition(AssistantState.IDLE, "nothing heard")
        if utterance.speech_detected:
            await self._announce(DIDNT_CATCH)
        return None

    async def _wait_for_trigger(self) -> None:
        self._engine.discard_pending_events()
        while (hit := await self._trigger.wait()) and self.muted:
            log.info("voice.trigger.ignored_while_muted", source=hit.source)
        await self._acknowledge(hit, enter_listening=True)

    async def _acknowledge(self, hit: TriggerHit, *, enter_listening: bool) -> None:
        """Earcon and LISTENING; after a barge-in the controller already entered LISTENING."""
        watch = Stopwatch(self._clock)
        log.info("voice.triggered", source=hit.source, score=round(hit.score, 3))
        self._bus.publish(WakeWordHeard(score=hit.score))
        await self._earcon()
        if enter_listening:
            self._state.transition(AssistantState.LISTENING, hit.source)
        publish_sample(self._bus, WAKE_TO_LISTENING, watch.elapsed_ms)

    async def _transcribe(self, utterance: Utterance) -> str:
        if not utterance.speech_detected:
            return ""
        watch = Stopwatch(self._clock)
        try:
            transcript = await self._stt.transcribe(utterance.samples)
        except TranscriptionError:
            log.exception("voice.stt.failed")
            return ""
        latency_ms = watch.stop()
        publish_sample(self._bus, SPEECH_END_TO_TRANSCRIPT, latency_ms)
        self._bus.publish(TranscriptReady(text=transcript.text, latency_ms=latency_ms))
        return transcript.text.strip()

    def _usable(self, text: str) -> bool:
        return sum(character.isalnum() for character in text) >= self._min_chars


class PlaybackObserver(Protocol):
    """Told when an answer becomes audible, ends, or is cut off (barge-in)."""

    def playback_started(self) -> None:
        """The first audio of an answer reached the device."""

    def playback_finished(self) -> None:
        """The answer was played in full."""

    def playback_interrupted(self) -> None:
        """Playback was stopped under the speaker."""


@runtime_checkable
class SentenceObserver(Protocol):
    """Optionally told the text of each sentence as it starts (the self-wake guard)."""

    def sentence_started(self, text: str) -> None:
        """``text`` is about to be heard."""


@dataclass(frozen=True, slots=True)
class _EndOfSentence:
    error: SynthesisError | None


@dataclass(slots=True)
class _Sentence:
    text: str
    chunks: asyncio.Queue[Samples | _EndOfSentence]
    error: SynthesisError | None = None


@dataclass(frozen=True, slots=True)
class _Flush:
    done: asyncio.Future[bool]


class VoiceOutput:
    """``OutputSink`` that speaks: TTS per sentence, pipelined into the speaker.

    Two worker tasks connected by a bounded queue: the synthesiser turns
    sentences into chunk queues, starting one synthesis task per sentence
    at once, and the player feeds the queues to the speaker in order. So
    sentences *n + 1* and *n + 2* are synthesised concurrently with each
    other and with *n*, and a slow network round trip for one sentence no
    longer delays the ones after it. Every sentence is also printed
    (``Jarvis> ...``), which keeps the terminal a transcript and is the
    TTS-failure fallback.

    Args:
        tts: The speech provider; its ``SynthesisError`` degrades to text.
        speaker: Plays chunks and reports interruption.
        bus: Receives the first-sentence -> audio latency sample.
        terminal: Prints sentences and notices.
        observer: Usually the barge-in controller.
        on_sentence: Called as each sentence starts (the overlay's response line).
        clock: Monotonic time for latency.
        lookahead: Sentences queued, and synthesised, behind the one playing.
    """

    def __init__(
        self,
        *,
        tts: TTS,
        speaker: SpeechPlayer,
        bus: EventBus,
        terminal: OutputSink,
        observer: PlaybackObserver | None = None,
        on_sentence: Callable[[str], None] | None = None,
        clock: Clock | None = None,
        lookahead: int = DEFAULT_LOOKAHEAD,
    ) -> None:
        self._tts = tts
        self._speaker = speaker
        self._bus = bus
        self._terminal = terminal
        self._observer = observer
        self._on_sentence = on_sentence
        self._clock = clock or SystemClock()
        self._lookahead = lookahead
        self._sentences: asyncio.Queue[str | _Flush] = asyncio.Queue()
        self._utterances: asyncio.Queue[_Sentence | _Flush] = asyncio.Queue(lookahead)
        self._pending: set[asyncio.Future[bool]] = set()
        self._synthesiser: asyncio.Task[None] | None = None
        self._syntheses: set[asyncio.Task[None]] = set()
        self._player: asyncio.Task[None] | None = None
        self._answer_watch: Stopwatch | None = None
        self._answer_open = False
        self._audible = False
        self._tts_failed = False
        self._interruptions = 0
        self._failure: BaseException | None = None

    @property
    def interruptions(self) -> int:
        """How many times playback was cut off so far."""
        return self._interruptions

    def set_observer(self, observer: PlaybackObserver) -> None:
        """Attach the barge-in controller (built after this sink)."""
        self._observer = observer

    async def say(self, sentence: str) -> None:
        """Queue ``sentence`` for synthesis and return at once."""
        self._ensure_workers()
        if not self._answer_open:
            self._answer_open = True
            self._answer_watch = Stopwatch(self._clock)
        await self._sentences.put(sentence)

    async def notice(self, text: str) -> None:
        """System messages are printed, never spoken."""
        await self._terminal.notice(text)

    async def flush(self) -> bool:
        """Wait until everything said so far has played; False if interrupted."""
        self._ensure_workers()
        done: asyncio.Future[bool] = asyncio.get_running_loop().create_future()
        self._pending.add(done)
        await self._sentences.put(_Flush(done))
        return await done

    async def aclose(self) -> None:
        """Stop the workers and fall silent; pending flushes report False."""
        for task in (self._synthesiser, self._player, *self._cancel_syntheses()):
            if task is not None:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
        self._synthesiser = self._player = None
        self._resolve_pending(False)
        self._speaker.stop()

    def _ensure_workers(self) -> None:
        if self._failure is not None:
            raise RuntimeError("the voice output worker crashed") from self._failure
        if self._synthesiser is None:
            self._synthesiser = self._spawn(self._synthesise_forever(), "voice-synthesiser")
        if self._player is None:
            self._player = self._spawn(self._play_forever(), "voice-player")

    def _spawn(self, work: Coroutine[object, object, None], name: str) -> asyncio.Task[None]:
        task = asyncio.create_task(work, name=name)
        task.add_done_callback(self._on_worker_done)
        return task

    def _on_worker_done(self, task: asyncio.Task[None]) -> None:
        if task.cancelled() or (error := task.exception()) is None:
            return
        log.error("voice.output.worker_failed", worker=task.get_name(), error=repr(error))
        self._failure = error
        for done in self._pending:
            if not done.done():
                done.set_exception(error)
        self._pending.clear()

    async def _synthesise_forever(self) -> None:
        while True:
            item = await self._sentences.get()
            if isinstance(item, _Flush):
                await self._utterances.put(item)
                continue
            sentence = _Sentence(item, asyncio.Queue())
            await self._utterances.put(sentence)
            self._start_synthesis(sentence)

    def _start_synthesis(self, sentence: _Sentence) -> None:
        """Synthesise ``sentence`` in its own task, concurrently with earlier ones."""
        task = asyncio.create_task(self._synthesise(sentence), name="voice-synthesis")
        self._syntheses.add(task)
        task.add_done_callback(self._syntheses.discard)
        task.add_done_callback(self._on_worker_done)

    def _cancel_syntheses(self) -> list[asyncio.Task[None]]:
        """Cancel every running synthesis; returns them so a caller can await them."""
        tasks = list(self._syntheses)
        for task in tasks:
            task.cancel()
        self._syntheses.clear()
        return tasks

    async def _synthesise(self, sentence: _Sentence) -> None:
        error: SynthesisError | None = None
        try:
            async for chunk in self._tts.synthesize(sentence.text):
                sentence.chunks.put_nowait(chunk.samples)
        except SynthesisError as failure:
            error = failure  # reported by the player, in playback order
        finally:
            sentence.chunks.put_nowait(_EndOfSentence(error))

    async def _play_forever(self) -> None:
        while True:
            item = await self._utterances.get()
            if isinstance(item, _Flush):
                self._finish_answer(item)
            else:
                await self._play(item)

    async def _play(self, sentence: _Sentence) -> None:
        await self._terminal.say(sentence.text)
        if self._on_sentence is not None:
            self._on_sentence(sentence.text)
        observer: object = self._observer  # the sentence hook is optional
        if isinstance(observer, SentenceObserver):
            observer.sentence_started(sentence.text)
        stops_before = self._speaker.stop_count
        completed = await self._speaker.play(self._chunks(sentence, stops_before))
        if sentence.error is not None:
            await self._report_failure(sentence.error)
        if not completed:
            self._interrupted()

    async def _chunks(self, sentence: _Sentence, stops_before: int) -> AsyncIterator[Samples]:
        while (samples := await self._next_chunk(sentence, stops_before)) is not None:
            self._audio_started()
            yield samples

    async def _next_chunk(self, sentence: _Sentence, stops_before: int) -> Samples | None:
        """The next chunk, or None at the end of the sentence or once playback was stopped."""
        while True:
            try:
                async with asyncio.timeout(CHUNK_POLL_SECONDS):
                    item = await sentence.chunks.get()
            except TimeoutError:
                if self._speaker.stop_count != stops_before:
                    return None
                continue
            if isinstance(item, _EndOfSentence):
                sentence.error = item.error
                return None
            return item

    def _audio_started(self) -> None:
        if self._audible:
            return
        self._audible = True
        if self._answer_watch is not None:
            publish_sample(self._bus, FIRST_SENTENCE_TO_AUDIO, self._answer_watch.stop())
            self._answer_watch = None
        if self._observer is not None:
            self._observer.playback_started()

    def _finish_answer(self, flush: _Flush) -> None:
        if self._audible and self._observer is not None:
            self._observer.playback_finished()
        self._audible = self._answer_open = False
        self._answer_watch = None
        self._pending.discard(flush.done)
        if not flush.done.done():
            flush.done.set_result(True)

    async def _report_failure(self, error: SynthesisError) -> None:
        if self._tts_failed:
            log.debug("voice.tts.failed_again", error=str(error))
            return
        self._tts_failed = True
        log.warning("voice.tts.degraded", error=str(error))
        await self._terminal.notice(TTS_FAILED_NOTICE.format(error=error))

    def _interrupted(self) -> None:
        """Drop everything not yet played and tell the observer."""
        self._interruptions += 1
        if self._synthesiser is not None:
            self._synthesiser.cancel()
            self._synthesiser = None
        self._cancel_syntheses()
        self._sentences = asyncio.Queue()
        self._utterances = asyncio.Queue(self._lookahead)
        self._resolve_pending(False)
        self._audible = self._answer_open = False
        self._answer_watch = None
        log.info("voice.output.interrupted", count=self._interruptions)
        if self._observer is not None:
            self._observer.playback_interrupted()

    def _resolve_pending(self, delivered: bool) -> None:
        for done in self._pending:
            if not done.done():
                done.set_result(delivered)
        self._pending.clear()


class SystemVoice:
    """Speaks short system phrases in full with barge-in held off.

    Used for "Sorry, I didn't catch that." and confirmation questions: the
    user is expected to talk right after (or over) them, and that must not
    count as interrupting an answer.

    Args:
        output: Where the phrase is spoken.
        hold_barge_in: Context manager factory that keeps barge-in disarmed.
    """

    def __init__(
        self, output: OutputSink, hold_barge_in: Callable[[], AbstractContextManager[None]]
    ) -> None:
        self._output = output
        self._hold = hold_barge_in

    async def __call__(self, text: str) -> None:
        """Say ``text`` and wait until it has been played."""
        with self._hold():
            await self._output.say(text)
            await self._output.flush()


class AnswerListener(Protocol):
    """Captures one spoken reply (``VoiceInput.listen_for_answer``)."""

    async def listen_for_answer(self) -> str:
        """The transcript of the reply, empty if nothing was heard."""


def parse_spoken_yes_no(answer: str) -> bool | None:
    """Like ``parse_yes_no`` but also accepts "Yes, please." and "No thanks."."""
    verdict = parse_yes_no(answer)
    if verdict is not None or not answer.strip():
        return verdict
    return parse_yes_no(answer.split(maxsplit=1)[0].strip(",.!?"))


class VoiceConfirmation:
    """``ask_confirmation`` for voice mode: the question is spoken, the answer heard.

    Unclear or missing answers are asked again a few times and then count
    as "no", the same safe default as the terminal version.
    """

    def __init__(
        self,
        *,
        listener: AnswerListener,
        speak: Announcer,
        attempts: int = DEFAULT_CONFIRM_ATTEMPTS,
    ) -> None:
        self._listener = listener
        self._speak = speak
        self._attempts = attempts

    async def __call__(self, prompt: str) -> bool:
        """Ask ``prompt`` until the user says yes or no."""
        await self._speak(prompt)
        for _ in range(self._attempts):
            verdict = parse_spoken_yes_no(await self._listener.listen_for_answer())
            if verdict is not None:
                return verdict
            await self._speak(PLEASE_ANSWER)
        log.info("voice.confirmation.unclear", attempts=self._attempts)
        return False


# Design notes:
# - Sentence pipelining with a bounded hand-off queue (``lookahead``): the
#   synthesiser runs ahead of the player by at most two sentences, little
#   enough that a barge-in throws away seconds, not paragraphs, of work.
# - One synthesis task per sentence rather than one after the other
#   (ADR-0030): edge-tts's time to first byte is a service round trip of
#   0.6 s at the median and up to several seconds, and a sentence that
#   waited for the previous one's whole download inherited that delay as a
#   gap in the speech. Measured with three Swedish sentences, the third was
#   complete after 2.7 s sequentially and 0.9 s concurrently. The queue
#   bound keeps at most lookahead + 1 requests in flight.
# - Interruption is detected twice over: Speaker.play compares the native
#   stop counter per chunk, and _next_chunk checks it while waiting for a
#   slow TTS chunk, so a barge-in is noticed within ~20 ms even when the
#   network is the bottleneck. The audio itself already stopped natively.
# - After an interruption the queues are replaced rather than drained and
#   the synthesiser task is cancelled (closing the provider's stream); the
#   old synthesiser can only write into the abandoned queues, so no stale
#   audio ever reaches the new answer.
# - TTS failure is a SynthesisError value carried to the player, so the
#   notice appears in playback order and is printed once; the sentence was
#   already printed, which is the degradation the spec asks for. The next
#   sentence tries TTS again, because edge-tts failures are usually a
#   transient network problem.
# - Any other exception in a worker is a bug: it is logged, re-raised from
#   the pending flush() calls and from every later say()/flush(), so it
#   surfaces (and ends the session) instead of leaving it hanging silently.
# - VoiceInput.read_line loops on "nothing heard" rather than returning an
#   empty line: the orchestrator would skip it anyway, and the state
#   machine has already gone back to IDLE.
