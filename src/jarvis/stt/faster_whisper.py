"""faster-whisper backed ``STT`` (the CLOUD-mode default).

faster-whisper runs Whisper through CTranslate2 on the local CPU or GPU, so
the only network use is the one-off model download. Inference is CPU-bound
and takes hundreds of milliseconds, which is why every call into the model
goes through ``asyncio.to_thread``: the event loop keeps polling the audio
engine while a transcription runs. The model is loaded lazily on first use
(or by ``warm_up``) because loading takes seconds and text-only mode never
needs it.

With ``JARVIS_STT_DEVICE=auto`` CTranslate2 picks CUDA whenever it sees a
GPU, even when the CUDA runtime libraries (cuBLAS, cuDNN) are not
installed, and then fails on load or on the first transcription. In that
case the provider falls back to the CPU (int8) once, with one warning that
names the missing library; an explicit ``cuda`` device still fails loudly.
"""

from __future__ import annotations

import asyncio
import importlib
import math
import re
import time
from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, Final, Protocol, cast

import numpy as np

from jarvis.stt.base import (
    ENGINE_SAMPLE_RATE,
    STT,
    Samples,
    Transcript,
    TranscriptionError,
    TranscriptSegment,
)
from jarvis.utils.errors import ConfigError
from jarvis.utils.logging import get_logger

if TYPE_CHECKING:
    from jarvis.config import Settings

WHISPER_SAMPLE_RATE: Final = 16_000
"""Whisper models are trained on 16 kHz audio and accept nothing else."""

DEFAULT_BEAM_SIZE: Final = 5
"""faster-whisper's own default; wider beams cost latency for little accuracy."""

AUTO_DEVICE: Final = "auto"
CPU_FALLBACK_DEVICE: Final = "cpu"
CPU_FALLBACK_COMPUTE_TYPE: Final = "int8"
"""The fastest CPU compute type CTranslate2 offers, used when CUDA cannot load."""

_GPU_LIBRARY_ERROR: Final = re.compile(
    r"cublas|cudnn|cudart|cufft|curand|nvrtc|\bcuda\b.*\b(?:library|dll|driver)\b", re.IGNORECASE
)
_LIBRARY_NAME: Final = re.compile(r"[\w.-]*(?:cublas|cudnn|cudart|cufft|curand|nvrtc)[\w.-]*", re.I)

log = get_logger(__name__)


class WhisperSegment(Protocol):
    """One segment as faster-whisper reports it."""

    @property
    def start(self) -> float:
        """Start time in seconds."""

    @property
    def end(self) -> float:
        """End time in seconds."""

    @property
    def text(self) -> str:
        """Decoded text, usually with a leading space."""

    @property
    def avg_logprob(self) -> float:
        """Mean token log-probability; ``exp`` of it approximates confidence."""


class WhisperInfo(Protocol):
    """Whole-utterance metadata as faster-whisper reports it."""

    @property
    def language(self) -> str:
        """Detected (or forced) ISO 639-1 language code."""

    @property
    def duration(self) -> float:
        """Audio length in seconds."""


class WhisperModel(Protocol):
    """The slice of ``faster_whisper.WhisperModel`` this module uses."""

    def transcribe(
        self,
        audio: Samples,
        *,
        language: str | None,
        beam_size: int,
        vad_filter: bool,
    ) -> tuple[Iterable[WhisperSegment], WhisperInfo]:
        """Return a lazy segment iterator and the utterance metadata."""


ModelFactory = Callable[[], WhisperModel]
"""Builds the model on demand; the real one downloads and loads weights."""


class FasterWhisperSTT(STT):
    """``STT`` over faster-whisper with lazy loading and off-loop inference."""

    def __init__(
        self,
        model_factory: ModelFactory,
        *,
        language: str | None = None,
        beam_size: int = DEFAULT_BEAM_SIZE,
        vad_filter: bool = False,
        cpu_fallback: ModelFactory | None = None,
    ) -> None:
        self._model_factory = model_factory
        self._cpu_fallback = cpu_fallback
        self._language = language
        self._beam_size = beam_size
        self._vad_filter = vad_filter
        self._model: WhisperModel | None = None
        self._load_lock = asyncio.Lock()

    @classmethod
    def from_settings(cls, settings: Settings, *, language: str | None = None) -> FasterWhisperSTT:
        """Build the real provider from ``Settings`` (model size, device, compute type).

        Only ``stt_device=auto`` gets a CPU fallback: "auto" means "whatever
        works", while an explicit ``cuda`` must fail loudly if CUDA cannot.
        """
        root = str(settings.models_dir / "whisper")
        factory = _faster_whisper_factory(
            settings.stt_model,
            device=settings.stt_device,
            compute_type=settings.stt_compute_type,
            download_root=root,
        )
        fallback = None
        if settings.stt_device == AUTO_DEVICE:
            fallback = _faster_whisper_factory(
                settings.stt_model,
                device=CPU_FALLBACK_DEVICE,
                compute_type=CPU_FALLBACK_COMPUTE_TYPE,
                download_root=root,
            )
        return cls(factory, language=language, cpu_fallback=fallback)

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
            TranscriptionError: If ``samples`` is not one-dimensional or the
                model fails.
        """
        audio = _prepare_audio(samples, sample_rate)
        if audio.size == 0:
            return Transcript(text="", language=None, segments=(), duration=0.0, confidence=None)
        model = await self._get_model()
        started = time.perf_counter()
        try:
            segments, info = await asyncio.to_thread(self._run, model, audio)
        except Exception as error:
            if not self._can_fall_back(error):
                raise TranscriptionError(f"faster-whisper failed: {error}") from error
            segments, info = await self._retry_on_cpu(audio, error)
        transcript = _to_transcript(segments, info)
        log.info(
            "stt.transcribed",
            audio_s=round(transcript.duration, 2),
            elapsed_ms=round((time.perf_counter() - started) * 1000),
            language=transcript.language,
            segments=len(transcript.segments),
        )
        return transcript

    async def _get_model(self) -> WhisperModel:
        async with self._load_lock:
            if self._model is None:
                started = time.perf_counter()
                self._model = await self._load()
                elapsed_ms = round((time.perf_counter() - started) * 1000)
                log.info("stt.model_loaded", elapsed_ms=elapsed_ms)
            return self._model

    async def _load(self) -> WhisperModel:
        try:
            return await asyncio.to_thread(self._model_factory)
        except Exception as error:
            if not self._can_fall_back(error):
                raise
            return await self._load_cpu_fallback(error)

    def _can_fall_back(self, error: Exception) -> bool:
        """A CPU fallback is left and ``error`` is a missing GPU library."""
        return self._cpu_fallback is not None and is_gpu_library_error(error)

    async def _load_cpu_fallback(self, error: Exception) -> WhisperModel:
        """Load the CPU model; the fallback is used up whether or not it works."""
        fallback, self._cpu_fallback = self._cpu_fallback, None
        if fallback is None:
            raise error
        log.warning(
            "stt.gpu_unavailable_using_cpu",
            missing=missing_gpu_library(error),
            error=str(error),
            compute_type=CPU_FALLBACK_COMPUTE_TYPE,
        )
        return await asyncio.to_thread(fallback)

    async def _retry_on_cpu(
        self, audio: Samples, error: Exception
    ) -> tuple[list[WhisperSegment], WhisperInfo]:
        """Swap in the CPU model after a GPU library failure and decode again."""
        try:
            async with self._load_lock:
                self._model = await self._load_cpu_fallback(error)
            return await asyncio.to_thread(self._run, self._model, audio)
        except Exception as retry_error:
            raise TranscriptionError(f"faster-whisper failed: {retry_error}") from retry_error

    def _run(self, model: WhisperModel, audio: Samples) -> tuple[list[WhisperSegment], WhisperInfo]:
        # The segment iterator is lazy: decoding happens while it is consumed,
        # so it must be drained here, inside the worker thread.
        segments, info = model.transcribe(
            audio,
            language=self._language,
            beam_size=self._beam_size,
            vad_filter=self._vad_filter,
        )
        return list(segments), info


def is_gpu_library_error(error: BaseException) -> bool:
    """Whether ``error`` says a CUDA runtime library could not be loaded.

    Matches CTranslate2's "Library cublas64_12.dll is not found or cannot be
    loaded" and the cuDNN equivalents, but not GPU errors such as "CUDA out
    of memory" that a CPU fallback would only hide.
    """
    return _GPU_LIBRARY_ERROR.search(str(error)) is not None


def missing_gpu_library(error: BaseException) -> str:
    """The library named in ``error`` (``cublas64_12.dll``), for the warning."""
    match = _LIBRARY_NAME.search(str(error))
    return match.group(0) if match else "a CUDA runtime library"


def _prepare_audio(samples: Samples, sample_rate: int) -> Samples:
    if samples.ndim != 1:
        raise TranscriptionError(f"expected mono audio, got {samples.ndim} dimensions")
    audio = np.ascontiguousarray(samples, dtype=np.float32)
    if sample_rate == WHISPER_SAMPLE_RATE:
        return audio
    return resample_linear(audio, sample_rate, WHISPER_SAMPLE_RATE)


def resample_linear(audio: Samples, source_rate: int, target_rate: int) -> Samples:
    """Resample by linear interpolation.

    Good enough for speech going into Whisper, which downsamples to a
    mel spectrogram anyway; a proper polyphase filter would only matter for
    rates far above 16 kHz.
    """
    if source_rate <= 0 or target_rate <= 0:
        raise TranscriptionError("sample rates must be positive")
    if audio.size == 0 or source_rate == target_rate:
        return audio
    target_length = max(1, round(audio.size * target_rate / source_rate))
    source_positions = np.arange(audio.size, dtype=np.float64)
    target_positions = np.linspace(0.0, audio.size - 1, target_length)
    return np.interp(target_positions, source_positions, audio).astype(np.float32)


def _to_transcript(segments: list[WhisperSegment], info: WhisperInfo) -> Transcript:
    converted = tuple(
        TranscriptSegment(
            start=float(segment.start),
            end=float(segment.end),
            text=segment.text.strip(),
            confidence=_confidence(segment.avg_logprob),
        )
        for segment in segments
        if segment.text.strip()
    )
    return Transcript(
        text=" ".join(segment.text for segment in converted),
        language=info.language if converted else None,
        segments=converted,
        duration=float(info.duration),
        confidence=_weighted_confidence(converted),
    )


def _confidence(avg_logprob: float) -> float:
    return min(1.0, max(0.0, math.exp(avg_logprob)))


def _weighted_confidence(segments: tuple[TranscriptSegment, ...]) -> float | None:
    weights = [max(segment.end - segment.start, 1e-3) for segment in segments]
    scores = [segment.confidence for segment in segments if segment.confidence is not None]
    if not scores or len(scores) != len(weights):
        return None
    return sum(w * s for w, s in zip(weights, scores, strict=True)) / sum(weights)


def _faster_whisper_factory(
    model_size: str, *, device: str, compute_type: str, download_root: str
) -> ModelFactory:
    def factory() -> WhisperModel:
        try:
            # Imported lazily by name: an optional, slow-to-import dependency
            # that text-only mode never needs.
            faster_whisper = importlib.import_module("faster_whisper")
        except ImportError as error:
            raise ConfigError(
                "faster-whisper is not installed; run 'uv sync --extra cloud' "
                "or switch to JARVIS_MODE=local once step 12 lands."
            ) from error
        log.info("stt.model_loading", model=model_size, device=device, compute_type=compute_type)
        model = faster_whisper.WhisperModel(
            model_size, device=device, compute_type=compute_type, download_root=download_root
        )
        return cast(WhisperModel, model)

    return factory


# Design notes:
# - A model factory rather than a model instance keeps construction cheap
#   and lets tests inject a scripted fake; the real factory also owns the
#   ImportError-to-ConfigError translation so the remedy is spelled out.
# - The CPU fallback is one-shot and limited to device=auto: a machine
#   with an NVIDIA GPU but no CUDA toolkit is common, and without the
#   fallback every utterance became "Sorry, I didn't catch that". It
#   covers both places the failure shows up (CTranslate2 loads cuBLAS
#   lazily, on the first transcription) and logs exactly one warning.
# - The load lock prevents two concurrent first calls from loading the
#   weights twice; after that it is uncontended.
# - Confidence is exp(avg_logprob): a well-known approximation, not a
#   calibrated probability. It is exposed so the agent can ask "did you say
#   ...?" below a threshold rather than act on a bad transcript.
