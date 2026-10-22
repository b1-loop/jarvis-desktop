from collections.abc import Callable
from pathlib import Path

import pytest

from jarvis.config import Settings
from jarvis.llm.ollama_client import OllamaClient
from jarvis.os_integration.stub import StubPlatform
from jarvis.utils.errors import ConfigError
from jarvis.vision.errors import VisionUnavailableError
from jarvis.vision.factory import create_screen_reader, create_vision_backend
from jarvis.vision.screen import VisionOptions


def test_anthropic_answers_with_the_vision_model(make_settings: Callable[..., Settings]) -> None:
    settings = make_settings(
        anthropic_api_key="sk-ant-test", llm_model="chat-model", vision_model="claude-opus-5-5"
    )

    backend = create_vision_backend(settings)

    assert backend.llm is not None
    assert backend.llm.model == "claude-opus-5-5"
    assert backend.provider == "Anthropic"
    assert backend.sends_to_cloud
    assert settings.llm_model == "chat-model"  # the copy, not the original, was changed


def test_anthropic_without_a_key_fails_at_startup(make_settings: Callable[..., Settings]) -> None:
    with pytest.raises(ConfigError, match="ANTHROPIC_API_KEY"):
        create_vision_backend(make_settings())


def test_ollama_without_a_vision_model_is_unavailable(
    make_settings: Callable[..., Settings],
) -> None:
    backend = create_vision_backend(make_settings(llm_provider="ollama"))

    assert backend.llm is None
    assert not backend.sends_to_cloud
    assert "JARVIS_OLLAMA_VISION_MODEL" in backend.unavailable_reason


async def test_ollama_with_a_vision_model_uses_it(make_settings: Callable[..., Settings]) -> None:
    settings = make_settings(llm_provider="ollama", ollama_vision_model="qwen2.5vl:7b")

    backend = create_vision_backend(settings)

    assert isinstance(backend.llm, OllamaClient)
    assert backend.llm.model == "qwen2.5vl:7b"
    assert backend.provider == "Ollama"
    assert not backend.sends_to_cloud
    await backend.llm.aclose()


def test_local_llama_cpp_has_no_vision(make_settings: Callable[..., Settings]) -> None:
    backend = create_vision_backend(make_settings(llm_provider="local"))

    assert backend.llm is None
    assert "cannot see images" in backend.unavailable_reason


def test_options_come_from_settings(make_settings: Callable[..., Settings]) -> None:
    settings = make_settings(
        data_dir=Path("store"),
        vision_enabled=False,
        vision_max_long_edge=1568,
        vision_max_image_bytes=2_000_000,
    )

    assert VisionOptions.from_settings(settings) == VisionOptions(
        enabled=False,
        max_long_edge=1568,
        max_image_bytes=2_000_000,
        screenshot_dir=Path("store") / "screenshots",
    )


def test_screenshots_are_not_kept_when_disabled(make_settings: Callable[..., Settings]) -> None:
    options = VisionOptions.from_settings(make_settings(vision_save_screenshots=False))

    assert options.screenshot_dir is None


async def test_screen_reader_is_wired_from_settings(
    make_settings: Callable[..., Settings],
) -> None:
    reader = create_screen_reader(make_settings(llm_provider="local"), StubPlatform())

    assert reader.provider == "the offline model"
    with pytest.raises(VisionUnavailableError, match="cannot see images"):
        await reader.ask("What is this?")


def test_vision_settings_read_the_environment(
    make_settings: Callable[..., Settings], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("JARVIS_VISION_ENABLED", "false")
    monkeypatch.setenv("JARVIS_VISION_MAX_LONG_EDGE", "2576")
    monkeypatch.setenv("JARVIS_OLLAMA_VISION_MODEL", "")

    settings = make_settings()

    assert not settings.vision_enabled
    assert settings.vision_max_long_edge == 2576
    assert settings.ollama_vision_model is None
    assert settings.vision_max_image_bytes == 5_000_000


def test_absurd_vision_limits_are_rejected(make_settings: Callable[..., Settings]) -> None:
    with pytest.raises(ValueError, match="vision_max_long_edge"):
        make_settings(vision_max_long_edge=10)
