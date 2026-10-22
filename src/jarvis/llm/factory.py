"""Builds the configured ``LLMClient``.

This is the one place inside the ``llm`` package that knows every concrete
client. The composition root calls it (directly or through
``jarvis.core.providers.build_llm``) so that nothing else needs to import a
provider class.
"""

from __future__ import annotations

from typing import assert_never

from jarvis.config import Settings
from jarvis.llm.anthropic_client import AnthropicClient
from jarvis.llm.base import LLMClient
from jarvis.llm.local_llama_client import LocalLlamaClient
from jarvis.llm.ollama_client import OllamaClient


def create_llm_client(settings: Settings) -> LLMClient:
    """Return the client selected by ``settings.llm_provider``.

    Raises:
        ConfigError: If the provider is missing credentials (``anthropic``),
            or the native module or GGUF model it needs is absent (``local``).
    """
    match settings.llm_provider:
        case "anthropic":
            return AnthropicClient.from_settings(settings)
        case "ollama":
            return OllamaClient.from_settings(settings)
        case "local":
            return LocalLlamaClient.from_settings(settings)
        case _ as unreachable:
            assert_never(unreachable)


# Design notes:
# - Every branch raises ConfigError with a remedy rather than falling back,
#   keeping the promise that Jarvis never silently uses a different
#   provider than the one asked for.
