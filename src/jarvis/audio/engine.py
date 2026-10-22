"""Asyncio facade over the native audio engine (``jarvis_audio``).

The native engine runs its own threads and exposes a poll-based API so no
native thread ever needs the GIL. This module turns that into asyncio
primitives: an async stream of events, ``capture_utterance`` for the
orchestrator and a ``Speaker`` that streams TTS chunks and knows when
barge-in cut it short. Native objects are injected through Protocols, so
this module is unit-tested with fakes and the real ``.pyd`` is exercised only
by ``native``-marked tests.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, cast

import numpy as np
from numpy.typing import NDArray

from jarvis import _native
from jarvis.audio.models import AudioModelPaths
from jarvis.utils.errors import ConfigError, JarvisError
from jarvis.utils.logging import get_logger

SAMPLE_RATE: int = 16_000
"""Capture rate of the native engine; event positions count samples at this rate."""

DEFAULT_SILENCE_END_MS: int = 700
"""The native segmenter's default silence hangover before ``SPEECH_ENDED``."""

FOLLOW_ON_SPEECH_MS: int = 300
"""Speech after the wake phrase shorter than this means the user only said
the wake phrase and is waiting for the earcon, not that they asked something."""

Samples = NDArray[np.float32]
"""Mono float32 PCM in [-1, 1]."""

log = get_logger(__name__)


class AudioModelError(JarvisError):
    """The native engine rejected a model file (corrupt, wrong format, wrong version)."""


class AudioEventKind(StrEnum):
    """Mirror of the native ``AudioEventType`` enum, by member name."""

    WAKE_WORD_DETECTED = "wake_word_detected"
    SPEECH_STARTED = "speech_started"
    SPEECH_ENDED = "speech_ended"
    SAMPLES_DROPPED = "samples_dropped"


@dataclass(frozen=True, slots=True)
class AudioEvent:
    """One event from the engine.

    Attributes:
        kind: What happened.
        value: Detector score, VAD probability or dropped-sample count.
        sample_position: Engine clock in samples, for latency measurement.
    """

    kind: AudioEventKind
    value: float
    sample_position: int


@dataclass(frozen=True, slots=True)
class Utterance:
    """What :meth:`AudioEngine.record_utterance` captured.

    Attributes:
        samples: Captured audio (pre-roll included), float32 at 16 kHz.
        speech_detected: Whether the VAD reported speech at all. False means
            the user said nothing before the start timeout.
        truncated: Whether the maximum utterance length cut the speech off.
    """

    samples: Samples
    speech_detected: bool
    truncated: bool = False


class NativeEventType(Protocol):
    """The pybind11 enum value; only its name is needed."""

    @property
    def name(self) -> str:
        """Enum member name, e.g. ``"SPEECH_STARTED"``."""


class NativeEvent(Protocol):
    """Shape of ``jarvis_audio.AudioEvent``."""

    @property
    def type(self) -> NativeEventType:
        """Which event occurred."""

    @property
    def value(self) -> float:
        """Score, probability or dropped-sample count."""

    @property
    def sample_position(self) -> int:
        """Engine clock in samples."""


class NativePlayback(Protocol):
    """Shape of ``jarvis_audio.Playback``."""

    def enqueue(self, samples: Samples) -> None:
        """Queue mono float32 samples at ``sample_rate``."""

    def stop(self) -> None:
        """Discard the queue and fall silent immediately."""

    def mark_end_of_stream(self) -> None:
        """Declare that no more chunks follow for this utterance."""

    def is_playing(self) -> bool:
        """True while samples are queued or more are expected."""

    def queued_samples(self) -> int:
        """Samples not yet rendered."""

    def stop_count(self) -> int:
        """How many times ``stop`` has run, including native barge-in."""

    def output_level(self) -> float:
        """RMS of the most recently rendered block, for the overlay waveform."""

    def sample_rate(self) -> int:
        """Rate of the output device."""


class NativeEngine(Protocol):
    """Shape of ``jarvis_audio.AudioEngine``."""

    def start(self) -> None:
        """Open the device and start the worker thread."""

    def stop(self) -> None:
        """Stop the worker and close the device."""

    def running(self) -> bool:
        """True between ``start`` and ``stop``."""

    def set_listening(self, listening: bool) -> None:
        """Retain audio for STT (and pause wake-word detection) while True."""

    def listening(self) -> bool:
        """Current listening state."""

    def take_captured(self) -> Samples:
        """Return and clear the retained audio."""

    def poll_events(self) -> list[NativeEvent]:
        """Drain the events reported since the previous poll."""

    def set_barge_in_target(self, playback: NativePlayback | None) -> None:
        """Stop ``playback`` natively when speech starts; ``None`` disables."""

    def set_wake_word_threshold(self, threshold: float) -> None:
        """Change the wake-word score threshold from the next chunk on."""

    def input_level(self) -> float:
        """RMS of the latest frame."""

    def dropped_samples(self) -> int:
        """Total samples lost to ring-buffer overflow."""


class AudioEngine:
    """Asyncio view of the native capture engine.

    Owns a polling task that drains native events into an ``asyncio.Queue``.
    There is a single consumer by design: the orchestrator. Events that a
    waiter is not interested in are logged and discarded rather than queued
    forever.
    """

    def __init__(
        self,
        native: NativeEngine,
        *,
        poll_interval: float = 0.01,
        max_utterance_seconds: float = 15.0,
        silence_end_ms: int = DEFAULT_SILENCE_END_MS,
    ) -> None:
        self._native = native
        self._poll_interval = poll_interval
        self._max_utterance_seconds = max_utterance_seconds
        self._silence_end_samples = silence_end_ms * SAMPLE_RATE // 1000
        self._events: asyncio.Queue[AudioEvent] = asyncio.Queue()
        self._poller: asyncio.Task[None] | None = None
        # Tracked from every event as it is polled, independent of the queue,
        # because the queue is cleared before each capture.
        self._in_speech = False
        self._wake_in_speech = False  # the wake phrase was part of the current speech
        self._wake_seen_before_end = False  # ... of the speech that ended last
        self._last_wake_position = 0

    async def __aenter__(self) -> AudioEngine:
        await self.start()
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.stop()

    @property
    def running(self) -> bool:
        """True between ``start`` and ``stop``."""
        return self._poller is not None

    @property
    def listening(self) -> bool:
        """True while captured audio is being retained for STT."""
        return self._native.listening()

    @property
    def input_level(self) -> float:
        """RMS of the latest frame, for the overlay waveform."""
        return self._native.input_level()

    async def start(self) -> None:
        """Open the device (off the event loop) and begin polling for events."""
        if self._poller is not None:
            return
        await asyncio.to_thread(self._native.start)
        self._poller = asyncio.create_task(self._poll_forever(), name="audio-engine-poller")
        log.info("audio.engine.started")

    async def stop(self) -> None:
        """Stop polling and close the device. Idempotent."""
        if self._poller is None:
            return
        self._poller.cancel()
        try:
            await self._poller
        except asyncio.CancelledError:
            pass
        except Exception as error:
            # Already reported with a traceback when it happened; note it
            # again here so a stop() caller sees why no events arrived.
            log.warning("audio.poller.was_dead", error=repr(error))
        self._poller = None
        await asyncio.to_thread(self._native.stop)
        log.info("audio.engine.stopped", dropped_samples=self._native.dropped_samples())

    async def events(self) -> AsyncIterator[AudioEvent]:
        """Every event, in order, for as long as the caller iterates."""
        while True:
            yield await self._events.get()

    async def next_event(self, *kinds: AudioEventKind) -> AudioEvent:
        """Wait for the next event of one of ``kinds`` (any kind if none given).

        Wrap the call in ``asyncio.timeout`` to bound the wait.
        """
        while True:
            event = await self._events.get()
            if not kinds or event.kind in kinds:
                return event
            log.debug("audio.event.skipped", kind=event.kind.value)

    def set_listening(self, listening: bool) -> None:
        """Start or stop retaining audio for STT; see the native docs."""
        self._native.set_listening(listening)

    def discard_pending_events(self) -> int:
        """Drop every event already queued and return how many there were.

        Events queue up while nobody waits for them (the assistant thinking
        or speaking). A stale ``SPEECH_ENDED`` would end the next capture
        before it began, and a stale ``WAKE_WORD_DETECTED`` (Jarvis saying
        its own name) would open a turn nobody asked for, so the voice front
        end clears the queue before it starts waiting.
        """
        discarded = 0
        while not self._events.empty():
            self._events.get_nowait()
            discarded += 1
        if discarded:
            log.debug("audio.events.discarded", count=discarded)
        return discarded

    def take_captured(self) -> Samples:
        """Return and clear the audio retained while listening."""
        return self._native.take_captured()

    def enable_barge_in(self, playback: NativePlayback | None) -> None:
        """Make speech onset stop ``playback`` natively; ``None`` disables it."""
        self._native.set_barge_in_target(playback)

    def set_wake_word_threshold(self, threshold: float) -> None:
        """Apply a new wake-word threshold live (the HUD's settings slider).

        The native pipeline clamps it to a usable probability, so a slider
        at either end can never disable or trip the wake word permanently.
        """
        self._native.set_wake_word_threshold(float(threshold))

    async def capture_utterance(self) -> Samples:
        """Record until the user stops talking.

        Listening is switched on, which seeds the capture with the pre-roll
        buffer, and off again once the engine reports the end of speech or
        ``max_utterance_seconds`` elapse. Whatever was captured is returned
        either way, so a user who talks for too long is transcribed rather
        than ignored.
        """
        return (await self.record_utterance(speech_in_progress=True)).samples

    async def record_utterance(
        self, *, start_timeout: float | None = None, speech_in_progress: bool = False
    ) -> Utterance:
        """Record one utterance and report whether it contained speech.

        Args:
            start_timeout: Seconds to wait for speech to begin; ``None``
                waits as long as the utterance limit allows.
            speech_in_progress: The user is already talking (barge-in), so
                only the end of speech is awaited.
        """
        self.set_listening(True)
        try:
            if speech_in_progress:
                heard, truncated = True, not await self._speech_ends()
            elif self._in_speech:
                heard, truncated = await self._continue_after_wake_phrase(start_timeout)
            else:
                heard = await self._speech_starts(start_timeout)
                truncated = heard and not await self._speech_ends()
        finally:
            self.set_listening(False)
        return Utterance(self.take_captured(), speech_detected=heard, truncated=truncated)

    @property
    def speech_active(self) -> bool:
        """True between the VAD's ``SPEECH_STARTED`` and ``SPEECH_ENDED``."""
        return self._in_speech

    async def _continue_after_wake_phrase(self, start_timeout: float | None) -> tuple[bool, bool]:
        """Capture when the user was already talking as listening began.

        "Hey Jarvis, what time is it" in one breath: the VAD reported the start
        of that speech before the wake word fired, and the event was cleared
        with the rest of the queue, so only the end is awaited. If the speech
        ends right after the wake phrase, the user is waiting for the earcon;
        then the real request is awaited as usual.

        Returns:
            ``(heard, truncated)`` as in :class:`Utterance`.
        """
        ended = await self._wait_for_end()
        if ended is None:
            return True, True
        if not self._wake_in_speech_of(ended):
            return True, False
        heard = await self._speech_starts(start_timeout)
        return heard, heard and not await self._speech_ends()

    def _wake_in_speech_of(self, ended: AudioEvent) -> bool:
        """True when the speech that just ended was only the wake phrase.

        A hotkey pressed mid-sentence has no wake phrase in that speech, so the
        whole speech is the request.
        """
        if not self._wake_seen_before_end:
            return False
        spoken_after_wake = ended.sample_position - self._last_wake_position
        follow_on = spoken_after_wake - self._silence_end_samples
        return follow_on < FOLLOW_ON_SPEECH_MS * SAMPLE_RATE // 1000

    async def _wait_for_end(self) -> AudioEvent | None:
        try:
            async with asyncio.timeout(self._max_utterance_seconds):
                return await self.next_event(AudioEventKind.SPEECH_ENDED)
        except TimeoutError:
            log.warning("audio.capture.timeout", limit_s=self._max_utterance_seconds)
            return None

    async def _speech_starts(self, limit_s: float | None) -> bool:
        limit = self._max_utterance_seconds if limit_s is None else limit_s
        try:
            async with asyncio.timeout(limit):
                await self.next_event(AudioEventKind.SPEECH_STARTED)
        except TimeoutError:
            log.info("audio.capture.no_speech", waited_s=limit)
            return False
        return True

    async def _speech_ends(self) -> bool:
        return await self._wait_for_end() is not None

    def _track(self, event: AudioEvent) -> None:
        if event.kind is AudioEventKind.SPEECH_STARTED:
            self._in_speech = True
            self._wake_in_speech = False
        elif event.kind is AudioEventKind.SPEECH_ENDED:
            self._wake_seen_before_end = self._wake_in_speech
            self._in_speech = False
            self._wake_in_speech = False
        elif event.kind is AudioEventKind.WAKE_WORD_DETECTED:
            self._last_wake_position = event.sample_position
            self._wake_in_speech = self._in_speech

    async def _poll_forever(self) -> None:
        try:
            while True:
                for native_event in self._native.poll_events():
                    event = _convert(native_event)
                    self._track(event)
                    await self._events.put(event)
                await asyncio.sleep(self._poll_interval)
        except Exception:
            # A dead poller would make every waiter hang silently; make the
            # cause visible before propagating it into the task.
            log.exception("audio.poller.crashed")
            raise


def _convert(native_event: NativeEvent) -> AudioEvent:
    event = AudioEvent(
        kind=AudioEventKind[native_event.type.name],
        value=float(native_event.value),
        sample_position=int(native_event.sample_position),
    )
    if event.kind is AudioEventKind.SAMPLES_DROPPED:
        log.warning("audio.samples_dropped", count=int(event.value))
    return event


class Speaker:
    """Streams synthesised audio to native playback and reports interruption."""

    def __init__(self, native: NativePlayback, *, poll_interval: float = 0.02) -> None:
        self._native = native
        self._poll_interval = poll_interval

    @property
    def native(self) -> NativePlayback:
        """The underlying playback, to hand to ``AudioEngine.enable_barge_in``."""
        return self._native

    @property
    def sample_rate(self) -> int:
        """Rate the chunks passed to ``play`` must have."""
        return self._native.sample_rate()

    @property
    def playing(self) -> bool:
        """True while audio is queued or more chunks are expected."""
        return self._native.is_playing()

    @property
    def output_level(self) -> float:
        """RMS of what is being played right now, for the overlay waveform."""
        return self._native.output_level()

    @property
    def stop_count(self) -> int:
        """Stops so far (Python or native barge-in); a change means "interrupted"."""
        return self._native.stop_count()

    async def play(self, chunks: AsyncIterator[Samples]) -> bool:
        """Play ``chunks`` as they arrive and wait for the last one to finish.

        Returns:
            True if everything was played, False if ``stop`` (from Python or
            from native barge-in) interrupted it.
        """
        stops_before = self._native.stop_count()
        async for chunk in chunks:
            if self._native.stop_count() != stops_before:
                break
            self._native.enqueue(chunk)
        self._native.mark_end_of_stream()
        await self._wait_until_done(stops_before)
        return self._native.stop_count() == stops_before

    def stop(self) -> None:
        """Silence playback immediately and discard what was queued."""
        self._native.stop()

    async def _wait_until_done(self, stops_before: int) -> None:
        # Polling is the only option: the native queue drains on the device
        # thread and exposes no awaitable signal. ASYNC110 wants an Event.
        while self._native.is_playing() and self._native.stop_count() == stops_before:  # noqa: ASYNC110
            await asyncio.sleep(self._poll_interval)


def create_native_engine(
    paths: AudioModelPaths,
    *,
    wake_word: bool = True,
    min_speech_ms: int | None = None,
    wake_word_threshold: float | None = None,
) -> NativeEngine:
    """Construct the real ``jarvis_audio.AudioEngine``.

    Args:
        paths: Model files; the wake-word ones are only needed with ``wake_word``.
        wake_word: Load openWakeWord; False runs the VAD only (hotkey operation).
        wake_word_threshold: openWakeWord score that counts as the wake phrase;
            ``None`` keeps the native default (0.5). Higher rejects more
            look-alike speech, which matters when the user talks in another
            language near the microphone.
        min_speech_ms: How long speech must last before the engine reports
            ``SPEECH_STARTED`` (and triggers native barge-in); ``None`` keeps
            the native default.

    Raises:
        ConfigError: If a needed model file is missing.
        AudioModelError: If the engine cannot load a model file.
        NativeUnavailableError: If the extension module is not built.
    """
    missing = paths.missing() if wake_word else tuple(p for p in paths.missing() if p == paths.vad)
    if missing:
        names = ", ".join(path.name for path in missing)
        raise ConfigError(f"audio models missing ({names}); run scripts/download_models.py")
    module = _native.load("jarvis_audio")
    wake_word_models = (
        (paths.melspectrogram, paths.embedding, paths.wake_word) if wake_word else None
    )
    config = module.EngineConfig()
    if min_speech_ms is not None:
        config.pipeline.segmenter.min_speech_ms = min_speech_ms
    if wake_word_threshold is not None:
        config.pipeline.wake_word_threshold = wake_word_threshold
    try:
        return cast(NativeEngine, module.AudioEngine(paths.vad, wake_word_models, config))
    except module.ModelError as error:
        raise AudioModelError(str(error)) from error


def create_native_playback(sample_rate: int) -> NativePlayback:
    """Construct the real ``jarvis_audio.Playback`` on the default output device."""
    module = _native.load("jarvis_audio")
    return cast(NativePlayback, module.Playback(sample_rate))


# Design notes:
# - Polling instead of native callbacks: a callback would need the GIL on the
#   worker thread and would run Python code at audio-thread cadence. A 10 ms
#   poll adds at most 10 ms to wake-word latency, far below what a user notices.
# - Speaker.play reports interruption via the native stop counter because
#   barge-in happens entirely in C++; Python only learns about it afterwards.
# - capture_utterance returns partial audio on timeout on purpose: a long
#   monologue should be transcribed, not dropped. record_utterance adds a
#   separate start timeout so a wake word followed by silence gives up after
#   a few seconds instead of the full utterance limit.
