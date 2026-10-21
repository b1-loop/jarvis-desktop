"""Builds the configured ``LLMClient``.

This is the one place inside the ``llm`` package that knows every concrete
client. The composition root (``__main__``, step 6) calls it so that nothing
else needs to import a provider class.
"""

from __future__ import annotations

from typing import assert_never

from jarvis.config import Settings
from jarvis.llm.anthropic_client import AnthropicClient
from jarvis.llm.base import LLMClient
from jarvis.llm.ollama_client import OllamaClient
from jarvis.utils.errors import ConfigError


def create_llm_client(settings: Settings) -> LLMClient:
    """Return the client selected by ``settings.llm_provider``.

    Raises:
        ConfigError: If the provider is not implemented yet (``local`` lands
            in step 12) or is missing credentials.
    """
    match settings.llm_provider:
        case "anthropic":
            return AnthropicClient.from_settings(settings)
        case "ollama":
            return OllamaClient.from_settings(settings)
        case "local":
            raise ConfigError(
                "JARVIS_LLM_PROVIDER=local (llama.cpp) is not implemented yet; it arrives "
                "in step 12. Use 'anthropic' or 'ollama' for now."
            )
        case _ as unreachable:
            assert_never(unreachable)


# Design notes:
# - Raising for "local" instead of falling back keeps the promise that
#   Jarvis never silently uses a different provider than the one asked for.
