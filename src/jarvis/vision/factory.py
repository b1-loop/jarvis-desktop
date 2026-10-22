"""Builds the vision backend and ``ScreenReader`` from ``Settings``.

The composition root calls :func:`create_screen_reader` and puts the result
in ``ToolContext.vision``. Concrete LLM clients are reached only through
:func:`jarvis.llm.factory.create_llm_client`, with the model swapped for the
configured vision model, so this module knows provider names but no
provider classes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, assert_never

from jarvis.llm.factory import create_llm_client
from jarvis.vision.screen import Redactor, ScreenReader, VisionBackend, VisionOptions

if TYPE_CHECKING:
    from jarvis.config import Settings
    from jarvis.os_integration.base import Platform
    from jarvis.utils.timing import Clock


def create_vision_backend(settings: Settings) -> VisionBackend:
    """The vision-capable client for ``settings.llm_provider``, if it has one.

    Anthropic answers with ``settings.vision_model``. Ollama answers only when
    ``ollama_vision_model`` names a vision model, because the default chat
    model is text-only. The offline llama.cpp wrapper has no image input.

    Raises:
        ConfigError: If the Anthropic provider is selected without a key.
    """
    match settings.llm_provider:
        case "anthropic":
            llm = create_llm_client(
                settings.model_copy(update={"llm_model": settings.vision_model})
            )
            return VisionBackend(llm, provider="Anthropic", sends_to_cloud=True)
        case "ollama":
            return _ollama_backend(settings)
        case "local":
            return VisionBackend(
                None,
                provider="the offline model",
                sends_to_cloud=False,
                unavailable_reason="the offline llama.cpp model cannot see images",
            )
        case _ as unreachable:
            assert_never(unreachable)


def create_screen_reader(
    settings: Settings,
    platform: Platform | None,
    *,
    redactor: Redactor | None = None,
    clock: Clock | None = None,
) -> ScreenReader:
    """A ``ScreenReader`` wired from settings; ``platform`` may be ``None`` in text-only runs."""
    return ScreenReader(
        platform,
        create_vision_backend(settings),
        options=VisionOptions.from_settings(settings),
        redactor=redactor,
        clock=clock,
    )


def _ollama_backend(settings: Settings) -> VisionBackend:
    if settings.ollama_vision_model is None:
        return VisionBackend(
            None,
            provider="Ollama",
            sends_to_cloud=False,
            unavailable_reason=(
                "the local model cannot see images; set JARVIS_OLLAMA_VISION_MODEL "
                "to a vision model such as qwen2.5vl:7b"
            ),
        )
    update = {"ollama_model": settings.ollama_vision_model}
    llm = create_llm_client(settings.model_copy(update=update))
    return VisionBackend(llm, provider="Ollama", sends_to_cloud=False)


# Design notes:
# - Ollama counts as local (sends_to_cloud=False) on the assumption that
#   JARVIS_OLLAMA_BASE_URL points at this machine or the user's own network,
#   which is what the setting exists for.
# - A separate vision client, not the chat client: the vision model may
#   differ from the chat model (and for Ollama it almost always does), and a
#   screen question is a one-shot request outside the conversation history.
