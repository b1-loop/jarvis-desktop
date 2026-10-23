"""whisper.cpp backed ``STT`` via ``jarvis_inference`` (the LOCAL-mode provider).

The native ``WhisperTranscriber`` runs whisper.cpp on the CPU and releases
the GIL while it decodes, so wrapping each call in ``asyncio.to_thread`` is
enough to keep the event loop (and the audio engine's polling) responsive.
Like :class:`~jarvis.stt.faster_whisper.FasterWhisperSTT`, the model is
loaded lazily or by ``warm_up`` and the result is the same
:class:`~jarvis.stt.base.Transcript`, so the orchestrator cannot tell the
two providers apart.
"""

from __future__ import annotations

import asyncio
import re
import time
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Final, Protocol, cast

from jarvis.inference.native import load_inference_module, require_model_file
from jarvis.stt.base import (
    ENGINE_SAMPLE_RATE,
    STT,
    Samples,
    Transcript,
    TranscriptionError,
    TranscriptSegment,
)
from jarvis.stt.faster_whisper import WHISPER_SAMPLE_RATE, prepare_whisper_audio
from jarvis.utils.errors import JarvisError
from jarvis.utils.logging import get_logger

if TYPE_CHECKING:
    from pathlib import Path

    from jarvis.config import Settings

GREEDY_BEAM_SIZE: Final = 1
"""whisper.cpp's greedy decoder; beam search costs ~2x latency on a CPU for
a small gain on the short commands an assistant hears."""

MILLISECONDS_PER_SECOND: Final = 1000.0

_NON_SPEECH: Final = re.compile(r"^\s*[\[(][^\])]*[\])]\s*$")
"""Whole-segment annotations such as ``[BLANK_AUDIO]`` or ``(wind blowing)``."""

log = get_logger(__name__)


class NativeSegment(Protocol):
    """Shape of ``jarvis_inference.TranscriptSegment``."""

    @property
    def start_ms(self) -> int:
        """Start in milliseconds from the beginning of the audio."""

    @property
    def end_ms(self) -> int:
        """End in milliseconds from the beginning of the audio."""

    @property
    def text(self) -> str:
        """Decoded text, usually with a leading space."""


class NativeTranscript(Protocol):
    """Shape of ``jarvis_inference.Transcript``."""

    @property
    def text(self) -> str:
        """The whole transcript as whisper.cpp joined it."""

    @property
    def language(self) -> str:
        """Detected (or forced) ISO 639-1 code."""

    @property
    def segments(self) -> Sequence[NativeSegment]:
        """Timed pieces, in order."""


class NativeWhisper(Protocol):
    """Shape of ``jarvis_inference.WhisperTranscriber``."""

    def transcribe(
        self,
        samples: Samples,
        language: str | None = None,
        translate: bool = False,
        threads: int = 0,
        beam_size: int = 1,
    ) -> NativeTranscript:
        """Transcribe 16 kHz mono float32 samples (GIL released natively)."""


WhisperFactory = Callable[[], NativeWhisper]
"""Builds the transcriber on demand; the real one loads a ggml file (~100 ms-1 s)."""


class WhisperCppSTT(STT):
    """``STT`` over the native whisper.cpp wrapper, fully offline."""

    def __init__(
        self,
        factory: WhisperFactory,
        *,
        language: str | None = None,
        threads: int = 0,
        beam_size: int = GREEDY_BEAM_SIZE,
    ) -> None:
        self._factory = factory
        self._language = language
        self._threads = threads
        self._beam_size = beam_size
        self._model: NativeWhisper | None = None
        self._load_lock = asyncio.Lock()

    @classmethod
    def from_settings(cls, settings: Settings) -> WhisperCppSTT:
        """Build the real provider from ``Settings``.

        Raises:
            ConfigError: If ``jarvis_inference`` is not built (or is the stub
                build) or the model file is missing; checked here so the
                failure happens at startup, not on the first utterance.
        """
        model_path = require_model_file(settings.local_stt_model, setting="JARVIS_LOCAL_STT_MODEL")
        factory = native_whisper_factory(model_path)
        return cls(factory, language=settings.stt_language, threads=settings.local_stt_threads)

    @property
    def loaded(self) -> bool:
        """True once the model weights are in memory."""
        return self._model is not None

    async def warm_up(self) -> None:
        """Load the model now so the first utterance is not slowed by it."""
        await self._get_model()

    async def transcribe(
        self, samples: Samples, sample_rate: int = ENGINE_SAMPLE_RATE
    ) -> Transcript:
        """Transcribe ``samples``; silence yields an empty transcript.

        Raises:
            TranscriptionError: If ``samples`` is not mono, the model cannot
                be loaded, or whisper.cpp fails.
        """
        audio = prepare_whisper_audio(samples, sample_rate)
        duration = audio.size / WHISPER_SAMPLE_RATE
        if audio.size == 0:
            return Transcript(text="", language=None, segments=(), duration=0.0, confidence=None)
        model = await self._get_model()
        started = time.perf_counter()
        try:
            native = await asyncio.to_thread(self._run, model, audio)
        except Exception as error:
            raise TranscriptionError(f"whisper.cpp failed: {error}") from error
        transcript = to_transcript(native, duration)
        log.info(
            "stt.transcribed",
            provider="whisper.cpp",
            audio_s=round(duration, 2),
            elapsed_ms=round((time.perf_counter() - started) * 1000),
            language=transcript.language,
            segments=len(transcript.segments),
        )
        return transcript

    async def _get_model(self) -> NativeWhisper:
        async with self._load_lock:
            if self._model is None:
                started = time.perf_counter()
                self._model = await asyncio.to_thread(self._load)
                elapsed_ms = round((time.perf_counter() - started) * 1000)
                log.info("stt.model_loaded", provider="whisper.cpp", elapsed_ms=elapsed_ms)
            return self._model

    def _load(self) -> NativeWhisper:
        try:
            return self._factory()
        except JarvisError:
            raise
        except Exception as error:
            raise TranscriptionError(f"could not load the whisper.cpp model: {error}") from error

    def _run(self, model: NativeWhisper, audio: Samples) -> NativeTranscript:
        return model.transcribe(
            audio,
            language=self._language,
            translate=False,
            threads=self._threads,
            beam_size=self._beam_size,
        )


def to_transcript(native: NativeTranscript, duration: float) -> Transcript:
    """Convert the native result, dropping non-speech annotations.

    whisper.cpp marks silence and noise with bracketed tags such as
    ``[BLANK_AUDIO]``; keeping them would turn silence into a non-empty
    transcript and the agent would answer a question nobody asked.
    """
    segments = tuple(
        TranscriptSegment(
            start=segment.start_ms / MILLISECONDS_PER_SECOND,
            end=segment.end_ms / MILLISECONDS_PER_SECOND,
            text=segment.text.strip(),
        )
        for segment in native.segments
        if segment.text.strip() and not _NON_SPEECH.match(segment.text)
    )
    return Transcript(
        text=" ".join(segment.text for segment in segments),
        language=(native.language or None) if segments else None,
        segments=segments,
        duration=duration,
        confidence=None,
    )


def native_whisper_factory(model_path: Path) -> WhisperFactory:
    """The real ``WhisperFactory`` over ``jarvis_inference.WhisperTranscriber``.

    The module is loaded (and the stub build rejected) now, so a missing
    build is a startup ``ConfigError``; the weights load later, on demand.

    Raises:
        ConfigError: If the native module is unavailable.
    """
    module = load_inference_module()

    def factory() -> NativeWhisper:
        return cast(NativeWhisper, module.WhisperTranscriber(model_path))

    return factory


# Design notes:
# - Confidence is None: the wrapper does not expose token probabilities and
#   inventing a number would mislead the "did you say ...?" heuristic that
#   FasterWhisperSTT's confidence feeds.
# - Duration comes from the sample count rather than the last segment end,
#   so silence trimmed by the model still reports the real audio length,
#   matching faster-whisper's `info.duration`.
# - Audio preparation reuses faster_whisper.prepare_whisper_audio; both providers feed
#   the same 16 kHz Whisper front end, so one tested implementation serves.
# - The factory loads the module eagerly but the weights lazily: building
#   the provider must fail fast on a missing build, yet stay cheap so
#   text-only runs and tests never pay for weights they do not use.
