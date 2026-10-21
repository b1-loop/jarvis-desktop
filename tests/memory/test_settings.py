from collections.abc import Callable

import pytest

from jarvis.config import Settings


def test_memory_defaults(make_settings: Callable[..., Settings]) -> None:
    settings = make_settings()

    assert settings.memory_recent_turns == 10
    assert settings.memory_semantic_hits == 5


def test_memory_settings_are_read_from_the_environment(
    monkeypatch: pytest.MonkeyPatch, make_settings: Callable[..., Settings]
) -> None:
    monkeypatch.setenv("JARVIS_MEMORY_RECENT_TURNS", "3")
    monkeypatch.setenv("JARVIS_MEMORY_SEMANTIC_HITS", "0")

    settings = make_settings()

    assert settings.memory_recent_turns == 3
    assert settings.memory_semantic_hits == 0
