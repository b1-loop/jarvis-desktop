"""Builds the STT, TTS and LLM providers for the configured runtime mode.

This is the provider half of the composition root: the one module outside
the provider packages that knows the concrete classes, so the orchestrator
and the agent keep depending on the ``STT`` / ``TTS`` / ``LLMClient``
interfaces only. The mode decides the speech stack:

=========  ===============  ===========  =====================================
Mode       STT              TTS          LLM
=========  ===============  ===========  =====================================
CLOUD      faster-whisper   edge-tts     ``JARVIS_LLM_PROVIDER`` (any of three)
LOCAL      whisper.cpp      Piper        llama.cpp, or Ollama if selected
=========  ===============  ===========  =====================================

Every prerequisite (optional extra, native build, model file, API key) is
checked here, at startup, and reported as a
:class:`~jarvis.utils.errors.ConfigError` that names the fix. Nothing falls
back to another provider silently.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Callable
from dataclasses import dataclass
from typing import Final

from jarvis import _native
from jarvis.config import RuntimeMode, Settings
from jarvis.llm.base import LLMClient
from jarvis.llm.factory import create_llm_client
from jarvis.stt.base import STT
from jarvis.stt.faster_whisper import FasterWhisperSTT
from jarvis.stt.whisper_cpp import WhisperCppSTT
from jarvis.tts.base import TTS
from jarvis.tts.edge_tts import EdgeTTS
from jarvis.tts.piper import PiperTTS
from jarvis.utils.errors import ConfigError, NativeUnavailableError
from jarvis.utils.logging import get_logger

CLOUD_EXTRA_REMEDY: Final = "run 'uv sync --extra cloud'"
BUILD_REMEDY: Final = "run 'uv run python scripts/build_native.py' or start with --text-only"

log = get_logger(__name__)


def _module_installed(name: str) -> bool:
    # find_spec answers without importing, so probing faster-whisper does
    # not pay its multi-second import at startup.
    return importlib.util.find_spec(name) is not None


def _native_problem(name: str) -> str | None:
    try:
        _native.load(name)
    except NativeUnavailableError as error:
        return error.reason
    return None


@dataclass(frozen=True, slots=True)
class DependencyProbe:
    """How the builders check CLOUD-mode prerequisites; replaced in tests.

    Attributes:
        module_installed: True if a Python package can be imported.
        native_problem: ``None`` if a native module loads, else the reason.
    """

    module_installed: Callable[[str], bool] = _module_installed
    native_problem: Callable[[str], str | None] = _native_problem


DEFAULT_PROBE: Final = DependencyProbe()


def build_stt(settings: Settings, *, probe: DependencyProbe = DEFAULT_PROBE) -> STT:
    """Speech-to-text for ``settings.mode``; weights load lazily on first use.

    Raises:
        ConfigError: CLOUD without the ``cloud`` extra; LOCAL without the
            native ``jarvis_inference`` build or the whisper model file.
    """
    if settings.mode is RuntimeMode.LOCAL:
        return WhisperCppSTT.from_settings(settings)
    _require_package(probe, "faster_whisper", "faster-whisper")
    return FasterWhisperSTT.from_settings(settings, language=settings.stt_language)


def build_tts(settings: Settings, *, probe: DependencyProbe = DEFAULT_PROBE) -> TTS:
    """Text-to-speech for ``settings.mode``; the voice loads lazily on first use.

    Raises:
        ConfigError: CLOUD without the ``cloud`` extra or the native
            ``jarvis_audio`` build (it decodes edge-tts' MP3); LOCAL without
            the ``local`` extra or the Piper voice and its ``.onnx.json``.
    """
    if settings.mode is RuntimeMode.LOCAL:
        return PiperTTS.from_settings(settings)
    _require_package(probe, "edge_tts", "edge-tts")
    problem = probe.native_problem("jarvis_audio")
    if problem is not None:
        raise ConfigError(
            f"CLOUD-mode TTS decodes edge-tts audio natively, but jarvis_audio is "
            f"unavailable ({problem}); {BUILD_REMEDY}"
        )
    return EdgeTTS.from_settings(settings)


def build_llm(settings: Settings) -> LLMClient:
    """The LLM client for ``settings.mode`` and ``settings.llm_provider``.

    CLOUD honours ``llm_provider`` as is. LOCAL never sends a request off
    the machine: it uses llama.cpp, unless ``llm_provider`` was explicitly
    set to ``ollama``, which is also local (localhost).

    Raises:
        ConfigError: If the selected client's credentials, native build or
            model file are missing.
    """
    if settings.mode is RuntimeMode.CLOUD or _local_mode_keeps_provider(settings):
        return create_llm_client(settings)
    log.info("providers.local_llm_selected", ignored_provider=settings.llm_provider)
    return create_llm_client(settings.model_copy(update={"llm_provider": "local"}))


def _local_mode_keeps_provider(settings: Settings) -> bool:
    """llama.cpp, or Ollama when the user chose it rather than inheriting the default."""
    if settings.llm_provider == "local":
        return True
    return settings.llm_provider == "ollama" and "llm_provider" in settings.model_fields_set


def _require_package(probe: DependencyProbe, module: str, package: str) -> None:
    if not probe.module_installed(module):
        raise ConfigError(
            f"CLOUD mode needs {package}, which is not installed; {CLOUD_EXTRA_REMEDY} "
            "or start with --mode local."
        )


# Design notes:
# - LOCAL-mode prerequisites are checked by each provider's from_settings
#   (native module, model files, piper-tts), because the same checks guard
#   JARVIS_LLM_PROVIDER=local in CLOUD mode; CLOUD-mode package checks live
#   here because FasterWhisperSTT and EdgeTTS import their packages lazily
#   and would otherwise fail on the first utterance.
# - In LOCAL mode an `anthropic` provider (the default, and what
#   .env.example sets) is replaced by llama.cpp rather than rejected:
#   `--mode local` must work on a checkout configured for the cloud, and the
#   mode's promise ("nothing leaves the machine") outranks a provider
#   default. The switch is logged so it is never invisible.
# - The probe is a dataclass of two callables rather than a Protocol class
#   hierarchy: tests replace one function, production needs no wiring.
# - Ollama became the default ``llm_provider`` (ADR-0013 update). LOCAL mode
#   still means llama.cpp unless Ollama was set explicitly (env, .env or
#   code: pydantic's ``model_fields_set``), so ``--mode local`` on a fresh
#   checkout stays the self-contained C++ stack the mode documents.
