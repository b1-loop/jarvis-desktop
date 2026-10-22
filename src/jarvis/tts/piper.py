"""Piper backed ``TTS`` (the LOCAL-mode, fully offline voice).

Piper is a VITS voice exported to ONNX plus espeak-ng phonemisation. The
``piper-tts`` package (the ``local`` extra) ships both as wheels for
Windows and Linux, runs the voice on onnxruntime and exposes
``PiperVoice.synthesize(text)``, a lazy generator that yields one float32
chunk per sentence. This provider pulls that generator one chunk at a time
in ``asyncio.to_thread``, so the first sentence plays while Piper is still
synthesising the next, and the event loop never runs the ONNX model.
"""

from __future__ import annotations

import asyncio
import importlib
import json
import time
from collections.abc import AsyncIterator, Callable, Iterable
from typing import TYPE_CHECKING, Final, Protocol, cast

import numpy as np

from jarvis.inference.models import piper_config_path
from jarvis.inference.native import DOWNLOAD_REMEDY
from jarvis.tts.base import TTS, AudioChunk, Samples, SynthesisError
from jarvis.utils.errors import ConfigError, JarvisError
from jarvis.utils.logging import get_logger

if TYPE_CHECKING:
    from pathlib import Path

    from jarvis.config import Settings

MONO: Final = 1
INSTALL_REMEDY: Final = "run 'uv sync --extra local'"

log = get_logger(__name__)


class PiperChunk(Protocol):
    """Shape of ``piper.AudioChunk`` (the fields this module reads)."""

    @property
    def sample_rate(self) -> int:
        """Rate of the chunk; the voice's rate."""

    @property
    def sample_channels(self) -> int:
        """Always 1 for Piper voices."""

    @property
    def audio_float_array(self) -> Samples:
        """PCM in ``[-1, 1]``."""


class PiperSynthesizer(Protocol):
    """Shape of ``piper.PiperVoice`` (the method this module calls)."""

    def synthesize(self, text: str) -> Iterable[PiperChunk]:
        """Lazily synthesise ``text``, one chunk per sentence."""


VoiceFactory = Callable[[], PiperSynthesizer]
"""Loads the voice on demand; the real one takes a few seconds (ONNX + espeak data)."""


class PiperTTS(TTS):
    """``TTS`` over a local Piper voice, streamed sentence by sentence."""

    def __init__(self, factory: VoiceFactory, *, voice: str, sample_rate: int) -> None:
        self._factory = factory
        self._voice = voice
        self._sample_rate = sample_rate
        self._synthesizer: PiperSynthesizer | None = None
        self._load_lock = asyncio.Lock()

    @classmethod
    def from_settings(cls, settings: Settings) -> PiperTTS:
        """Build the real provider for ``settings.local_tts_model``.

        Reads the voice's small ``.onnx.json`` now, because the playback
        device must be opened at the voice's rate before any speech exists.

        Raises:
            ConfigError: If ``piper-tts`` is not installed or the voice or its
                config is missing or unreadable.
        """
        voice_path = settings.local_tts_model
        sample_rate = read_voice_sample_rate(voice_path)
        return cls(piper_voice_factory(voice_path), voice=voice_path.stem, sample_rate=sample_rate)

    @property
    def voice(self) -> str:
        """The voice file's stem, e.g. ``en_GB-northern_english_male-medium``."""
        return self._voice

    @property
    def sample_rate(self) -> int:
        """The voice's native rate (22.05 kHz for medium voices)."""
        return self._sample_rate

    @property
    def loaded(self) -> bool:
        """True once the voice is in memory."""
        return self._synthesizer is not None

    async def warm_up(self) -> None:
        """Load the voice now; the first load takes seconds."""
        await self._get_synthesizer()

    async def synthesize(self, text: str) -> AsyncIterator[AudioChunk]:
        """Stream ``text`` as one chunk per sentence.

        Raises:
            SynthesisError: If the voice cannot be loaded, Piper fails, or a
                chunk is not mono at the voice's rate.
        """
        if not text.strip():
            return
        synthesizer = await self._get_synthesizer()
        started = time.perf_counter()
        chunks = await _call(lambda: iter(synthesizer.synthesize(text)))
        count = 0
        while (chunk := await _call(lambda: next(chunks, None))) is not None:
            count += 1
            yield self._convert(chunk)
            if count == 1:
                first_ms = round((time.perf_counter() - started) * 1000)
                log.debug("tts.first_chunk", provider="piper", first_chunk_ms=first_ms)
        if count == 0:
            raise SynthesisError(f"Piper produced no audio for {len(text)} characters")
        total_ms = round((time.perf_counter() - started) * 1000)
        log.info(
            "tts.synthesized", provider="piper", chars=len(text), chunks=count, total_ms=total_ms
        )

    async def _get_synthesizer(self) -> PiperSynthesizer:
        async with self._load_lock:
            if self._synthesizer is None:
                started = time.perf_counter()
                self._synthesizer = await asyncio.to_thread(self._load)
                elapsed_ms = round((time.perf_counter() - started) * 1000)
                log.info("tts.voice_loaded", provider="piper", elapsed_ms=elapsed_ms)
            return self._synthesizer

    def _load(self) -> PiperSynthesizer:
        try:
            return self._factory()
        except JarvisError:
            raise
        except Exception as error:
            raise SynthesisError(f"could not load Piper voice {self._voice}: {error}") from error

    def _convert(self, chunk: PiperChunk) -> AudioChunk:
        if chunk.sample_rate != self._sample_rate or chunk.sample_channels != MONO:
            raise SynthesisError(
                f"Piper returned {chunk.sample_channels} channel(s) at {chunk.sample_rate} Hz; "
                f"expected mono at {self._sample_rate} Hz"
            )
        samples = np.ascontiguousarray(chunk.audio_float_array, dtype=np.float32)
        return AudioChunk(samples, self._sample_rate)


async def _call[T](step: Callable[[], T]) -> T:
    try:
        return await asyncio.to_thread(step)
    except Exception as error:
        raise SynthesisError(f"Piper synthesis failed: {error}") from error


def read_voice_sample_rate(voice_path: Path) -> int:
    """The ``audio.sample_rate`` from the voice's ``.onnx.json``.

    Raises:
        ConfigError: If the voice or its config is missing or malformed.
    """
    config_path = piper_config_path(voice_path)
    for required in (voice_path, config_path):
        if not required.is_file():
            raise ConfigError(
                f"Piper voice file not found: {required}. {DOWNLOAD_REMEDY.capitalize()} "
                "or point JARVIS_LOCAL_TTS_MODEL at a voice whose .onnx.json sits next to it."
            )
    try:
        rate = json.loads(config_path.read_text(encoding="utf-8"))["audio"]["sample_rate"]
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise ConfigError(f"{config_path} is not a valid Piper voice config: {error}") from error
    if not isinstance(rate, int) or rate <= 0:
        raise ConfigError(f"{config_path} has an invalid audio.sample_rate: {rate!r}")
    return rate


def piper_voice_factory(voice_path: Path) -> VoiceFactory:
    """The real ``VoiceFactory`` over ``piper.PiperVoice.load``.

    The package is imported now, so a missing extra is a startup
    ``ConfigError``; the voice itself loads later, on demand.

    Raises:
        ConfigError: If ``piper-tts`` is not installed.
    """
    try:
        # Imported by name: an optional extra that CLOUD mode never needs.
        piper = importlib.import_module("piper")
    except ImportError as error:
        raise ConfigError(f"LOCAL-mode TTS needs piper-tts; {INSTALL_REMEDY}") from error

    def factory() -> PiperSynthesizer:
        return cast(PiperSynthesizer, piper.PiperVoice.load(voice_path))

    return factory


# Design notes:
# - Why the piper-tts package and not onnxruntime + espeak-ng by hand or the
#   Piper CLI: since 1.3 the package publishes abi3 wheels for Windows and
#   manylinux that bundle espeak-ng and its data, so `uv sync --extra local`
#   works on both the dev machine and CI with no compiler or system
#   package. Its only runtime dependency, onnxruntime, is already installed
#   by the cloud extra (faster-whisper's VAD). Re-implementing phonemisation
#   would mean shipping espeak-ng ourselves, and a CLI subprocess per
#   sentence would add process start-up to every sentence's latency.
# - Licence: piper-tts and espeak-ng are GPL-3.0-or-later. They are an
#   optional extra, imported only at run time in LOCAL mode, and not
#   vendored; Jarvis' own source stays MIT. Anyone redistributing a bundle
#   that includes the extra must honour the GPL (see ADR-0013).
# - onnxruntime coexistence: the audio engine loads its own onnxruntime.dll
#   from src/jarvis/_native; whichever copy loads first serves both, which
#   was verified in both import orders on the dev machine (1.31 on both
#   sides). A version skew between the two would show up in the native
#   Piper test.
# - Piper splits text into sentences itself and yields one chunk each, so a
#   multi-sentence input still streams; the orchestrator normally sends one
#   sentence, which arrives as one chunk.
