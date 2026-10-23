"""faster-whisper backed ``STT`` (the CLOUD-mode default).

faster-whisper runs Whisper through CTranslate2 on the local CPU or GPU, so
the only network use is the one-off model download. Inference is CPU-bound
and takes hundreds of milliseconds, which is why every call into the model
goes through ``asyncio.to_thread``: the event loop keeps polling the audio
engine while a transcription runs. The model is loaded lazily on first use
(or by ``warm_up``) because loading takes seconds and text-only mode never
needs it.

With ``JARVIS_STT_DEVICE=auto`` the provider tries the GPU first, with the
GPU model (``stt_model_gpu``, large-v3-turbo) and the cuBLAS/cuDNN DLLs of
the ``gpu`` extra on the DLL search path (:mod:`jarvis.stt.cuda_libraries`).
When CUDA cannot work (no NVIDIA device, or the runtime libraries missing,
which CTranslate2 reports on load or on the first transcription) it falls
back once to the CPU model (``stt_model_cpu``, small) in int8, with one
warning that names what is missing; an explicit ``cuda`` device still
fails loudly. ``stt_hotwords`` bias decoding towards names Whisper
otherwise mishears ("Jarvis", "Discord").

Decoding is tuned for short spoken commands (ADR-0030): greedy search
(``stt_beam_size`` 1), no timestamp tokens and no conditioning on earlier
windows, which transcribed the Swedish benchmark clips word for word like
beam search did, only faster. ``warm_up`` also runs one silent decode, because
the first CUDA decode after a load costs seconds of kernel set-up that
otherwise land on the user's first request.
"""

from __future__ import annotations

import asyncio
import importlib
import math
import re
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from types import ModuleType
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
from jarvis.stt.cuda_libraries import register_cuda_libraries
from jarvis.utils.errors import ConfigError
from jarvis.utils.logging import get_logger

if TYPE_CHECKING:
    from jarvis.config import Settings

WHISPER_SAMPLE_RATE: Final = 16_000
"""Whisper models are trained on 16 kHz audio and accept nothing else."""

DEFAULT_BEAM_SIZE: Final = 1
"""Greedy decoding: on short commands it matched beam 5 word for word, faster (ADR-0030)."""

WARM_UP_SECONDS: Final = 1.0
"""Length of the silent clip ``warm_up`` decodes to initialise the CUDA kernels."""

AUTO_DEVICE: Final = "auto"
AUTO_COMPUTE_TYPE: Final = "auto"
CUDA_DEVICE: Final = "cuda"
CPU_FALLBACK_DEVICE: Final = "cpu"
CPU_FALLBACK_COMPUTE_TYPE: Final = "int8"
"""The fastest CPU compute type CTranslate2 offers, used when CUDA cannot load."""
GPU_DEFAULT_COMPUTE_TYPE: Final = "int8_float16"
"""int8 weights, float16 maths: as fast as float16 on an RTX 4070 Laptop and
about 1 GB less VRAM, which matters next to a 7B Ollama model on 8 GB."""

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
        hotwords: str | None,
        without_timestamps: bool,
        condition_on_previous_text: bool,
    ) -> tuple[Iterable[WhisperSegment], WhisperInfo]:
        """Return a lazy segment iterator and the utterance metadata."""


class GpuUnavailableError(RuntimeError):
    """CTranslate2 sees no CUDA device, so the GPU model cannot load at all."""


ModelFactory = Callable[[], WhisperModel]
"""Builds the model on demand; the real one downloads and loads weights."""


@dataclass(frozen=True, slots=True)
class DecodeOptions:
    """How faster-whisper decodes one utterance.

    Attributes:
        beam_size: 1 is greedy search; wider beams cost latency on every request.
        vad_filter: faster-whisper's own Silero pass. Off because the native
            engine already cut the utterance with the same VAD model; running
            it again only adds time.
        without_timestamps: Skip timestamp tokens; the pipeline never uses
            segment times beyond the confidence weighting, which one segment
            spanning the whole clip serves just as well.
        condition_on_previous_text: Feed one window's text into the next.
            Utterances are single windows (under 30 s), so it only matters
            for long dictation, where it is also Whisper's main source of
            repetition loops.
    """

    beam_size: int = DEFAULT_BEAM_SIZE
    vad_filter: bool = False
    without_timestamps: bool = True
    condition_on_previous_text: bool = False


class FasterWhisperSTT(STT):
    """``STT`` over faster-whisper with lazy loading and off-loop inference."""

    def __init__(
        self,
        model_factory: ModelFactory,
        *,
        language: str | None = None,
        decode: DecodeOptions | None = None,
        cpu_fallback: ModelFactory | None = None,
        hotwords: str | None = None,
    ) -> None:
        self._model_factory = model_factory
        self._cpu_fallback = cpu_fallback
        self._language = language
        self._decode = decode or DecodeOptions()
        self._hotwords = hotwords.strip() if hotwords and hotwords.strip() else None
        self._model: WhisperModel | None = None
        self._load_lock = asyncio.Lock()

    @classmethod
    def from_settings(cls, settings: Settings, *, language: str | None = None) -> FasterWhisperSTT:
        """Build the real provider from ``Settings`` (model size, device, compute type).

        ``auto`` and ``cuda`` load the GPU model on CUDA; ``cpu`` (or any
        other CTranslate2 device) loads the CPU model there. Only ``auto``
        gets a CPU fallback: "auto" means "whatever works", while an
        explicit ``cuda`` must fail loudly if CUDA cannot.
        """
        root = str(settings.models_dir / "whisper")
        on_gpu = settings.stt_device in {AUTO_DEVICE, CUDA_DEVICE}
        factory = _faster_whisper_factory(
            settings.whisper_model(gpu=on_gpu),
            device=CUDA_DEVICE if on_gpu else settings.stt_device,
            compute_type=_compute_type(settings.stt_compute_type, gpu=on_gpu),
            download_root=root,
        )
        fallback = None
        if settings.stt_device == AUTO_DEVICE:
            fallback = _faster_whisper_factory(
                settings.whisper_model(gpu=False),
                device=CPU_FALLBACK_DEVICE,
                compute_type=CPU_FALLBACK_COMPUTE_TYPE,
                download_root=root,
            )
        return cls(
            factory,
            language=language,
            decode=DecodeOptions(beam_size=settings.stt_beam_size),
            cpu_fallback=fallback,
            hotwords=settings.stt_hotwords,
        )

    @property
    def loaded(self) -> bool:
        """True once the model weights are in memory."""
        return self._model is not None

    async def warm_up(self) -> None:
        """Load the model and decode a moment of silence, so the first request is fast.

        The silent decode initialises the CUDA kernels (seconds on the first
        call after a load) and is also where a missing cuBLAS shows up, so
        the CPU fallback happens here rather than on the user's first
        sentence. A failure is logged, not raised: the same error will be
        reported, and degraded to "didn't catch that", on the first request.
        """
        await self._get_model()
        started = time.perf_counter()
        silence = np.zeros(round(WARM_UP_SECONDS * WHISPER_SAMPLE_RATE), dtype=np.float32)
        try:
            await self.transcribe(silence, WHISPER_SAMPLE_RATE)
        except TranscriptionError as error:
            log.warning("stt.warm_up_failed", error=str(error))
            return
        log.info("stt.warmed_up", elapsed_ms=round((time.perf_counter() - started) * 1000))

    async def transcribe(
        self, samples: Samples, sample_rate: int = ENGINE_SAMPLE_RATE
    ) -> Transcript:
        """Transcribe ``samples``; silence yields an empty transcript.

        Raises:
            TranscriptionError: If ``samples`` is not one-dimensional or the
                model fails.
        """
        audio = prepare_whisper_audio(samples, sample_rate)
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
        """A CPU fallback is left and ``error`` says the GPU cannot be used."""
        gpu_unusable = isinstance(error, GpuUnavailableError) or is_gpu_library_error(error)
        return self._cpu_fallback is not None and gpu_unusable

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
            beam_size=self._decode.beam_size,
            vad_filter=self._decode.vad_filter,
            hotwords=self._hotwords,
            without_timestamps=self._decode.without_timestamps,
            condition_on_previous_text=self._decode.condition_on_previous_text,
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
    if isinstance(error, GpuUnavailableError):
        return "a CUDA device"
    match = _LIBRARY_NAME.search(str(error))
    return match.group(0) if match else "a CUDA runtime library"


def _compute_type(configured: str, *, gpu: bool) -> str:
    """``auto`` becomes int8_float16 on the GPU; anything explicit is kept."""
    if gpu and configured == AUTO_COMPUTE_TYPE:
        return GPU_DEFAULT_COMPUTE_TYPE
    return configured


def prepare_whisper_audio(samples: Samples, sample_rate: int) -> Samples:
    """Mono float32 at Whisper's 16 kHz; both Whisper providers feed this front end."""
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


def _import_faster_whisper() -> ModuleType:
    try:
        # Imported lazily by name: an optional, slow-to-import dependency
        # that text-only mode never needs.
        return importlib.import_module("faster_whisper")
    except ImportError as error:
        raise ConfigError(
            "faster-whisper is not installed; run 'uv sync --extra cloud' "
            "or switch to JARVIS_MODE=local once step 12 lands."
        ) from error


def _require_cuda_device() -> None:
    """Fail fast (and fall back) when CTranslate2 sees no NVIDIA GPU at all."""
    count = getattr(importlib.import_module("ctranslate2"), "get_cuda_device_count", None)
    if count is not None and count() == 0:
        raise GpuUnavailableError("CTranslate2 found no CUDA device")


def _faster_whisper_factory(
    model_size: str, *, device: str, compute_type: str, download_root: str
) -> ModelFactory:
    def factory() -> WhisperModel:
        if device == CUDA_DEVICE:
            register_cuda_libraries()  # before CTranslate2's CUDA backend loads
        faster_whisper = _import_faster_whisper()
        if device == CUDA_DEVICE:
            _require_cuda_device()
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
# - Two model settings instead of one plus a silent downgrade: the right
#   Whisper size differs by device. large-v3-turbo transcribes 14 s of
#   Swedish in about 0.6 s on an RTX 4070 Laptop GPU, but is far too slow
#   on a laptop CPU, where small (about 2.6 s for the same clip) is the
#   sweet spot. A fallback that kept the GPU model on the CPU would turn a
#   missing DLL into multi-second answers. JARVIS_STT_MODEL still pins
#   one model for every device.
# - Hotwords rather than an initial prompt: faster-whisper adds them to
#   the decoder prompt without making the model continue a fake previous
#   sentence, which an initial prompt can do on very short utterances.
# - The load lock prevents two concurrent first calls from loading the
#   weights twice; after that it is uncontended.
# - Decoding options (ADR-0030), measured on 20 Swedish edge-tts commands on
#   the RTX 4070 Laptop GPU, interleaved so clock changes hit every variant:
#   greedy + no timestamps + no conditioning gave the same transcripts as
#   beam 5 and was faster at the median and much tighter at p90; the
#   built-in vad_filter added time and changed nothing; a forced language
#   (JARVIS_STT_LANGUAGE=sv) roughly halved the time against auto-detection,
#   which runs the encoder a second time. The first decode after a load
#   took about 2 s against about 0.4 s warm, hence the silent warm-up decode.
# - Confidence is exp(avg_logprob): a well-known approximation, not a
#   calibrated probability. It is exposed so the agent can ask "did you say
#   ...?" below a threshold rather than act on a bad transcript.
