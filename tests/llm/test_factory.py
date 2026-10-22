from collections.abc import Callable

import pytest

from jarvis.config import Settings
from jarvis.llm.anthropic_client import AnthropicClient
from jarvis.llm.factory import create_llm_client
from jarvis.llm.ollama_client import OllamaClient
from jarvis.utils.errors import ConfigError


def test_anthropic_is_the_default_provider(make_settings: Callable[..., Settings]) -> None:
    client = create_llm_client(make_settings(anthropic_api_key="sk-ant-test"))

    assert isinstance(client, AnthropicClient)
    assert client.model == "claude-sonnet-5-5"


def test_anthropic_without_a_key_explains_the_remedy(
    make_settings: Callable[..., Settings],
) -> None:
    with pytest.raises(ConfigError, match="ANTHROPIC_API_KEY"):
        create_llm_client(make_settings())


async def test_ollama_provider_uses_the_configured_model_and_url(
    make_settings: Callable[..., Settings],
) -> None:
    settings = make_settings(
        llm_provider="ollama", ollama_model="llama3.2:3b", ollama_base_url="http://box:11434"
    )

    client = create_llm_client(settings)

    assert isinstance(client, OllamaClient)
    assert client.model == "llama3.2:3b"
    await client.aclose()


def test_local_provider_is_not_implemented_yet(make_settings: Callable[..., Settings]) -> None:
    with pytest.raises(ConfigError, match="step 12"):
        create_llm_client(make_settings(llm_provider="local"))
