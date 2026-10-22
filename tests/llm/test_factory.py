from collections.abc import Callable
from pathlib import Path

import pytest

from jarvis.config import Settings
from jarvis.llm import local_llama_client
from jarvis.llm.anthropic_client import AnthropicClient
from jarvis.llm.factory import create_llm_client
from jarvis.llm.local_llama_client import LocalLlamaClient
from jarvis.llm.ollama_client import OllamaClient
from jarvis.utils.errors import ConfigError


def test_ollama_is_the_default_provider(make_settings: Callable[..., Settings]) -> None:
    client = create_llm_client(make_settings())

    assert isinstance(client, OllamaClient)
    assert client.model == "qwen2.5:7b"


def test_anthropic_is_selected_when_configured(make_settings: Callable[..., Settings]) -> None:
    client = create_llm_client(
        make_settings(llm_provider="anthropic", anthropic_api_key="sk-ant-test")
    )

    assert isinstance(client, AnthropicClient)
    assert client.model == "claude-sonnet-5-5"


def test_anthropic_without_a_key_explains_the_remedy(
    make_settings: Callable[..., Settings],
) -> None:
    with pytest.raises(ConfigError, match="ANTHROPIC_API_KEY"):
        create_llm_client(make_settings(llm_provider="anthropic"))


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


def test_local_provider_builds_the_llama_cpp_client(
    make_settings: Callable[..., Settings], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = tmp_path / "model.gguf"
    model.write_bytes(b"GGUF")
    monkeypatch.setattr(local_llama_client, "load_inference_module", object)

    client = create_llm_client(make_settings(llm_provider="local", local_llm_model=model))

    assert isinstance(client, LocalLlamaClient)
    assert client.model == "model.gguf"


def test_local_provider_without_a_model_explains_the_remedy(
    make_settings: Callable[..., Settings], tmp_path: Path
) -> None:
    settings = make_settings(llm_provider="local", local_llm_model=tmp_path / "absent.gguf")

    with pytest.raises(ConfigError, match=r"download_models\.py --local"):
        create_llm_client(settings)
