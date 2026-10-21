"""Shared fixtures.

Settings are constructed with ``_env_file=None`` everywhere in the suite so a
developer's real ``.env`` can never leak into test results, and every
``JARVIS_*`` / credential variable is cleared from the environment first.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator
from typing import Any

import pytest

from jarvis.config import Settings

CREDENTIAL_VARIABLES = (
    "ANTHROPIC_API_KEY",
    "GOOGLE_CLIENT_SECRETS_FILE",
    "MS_CLIENT_ID",
    "MS_TENANT_ID",
    "SPOTIFY_CLIENT_ID",
    "SPOTIFY_CLIENT_SECRET",
    "SLACK_BOT_TOKEN",
    "DISCORD_BOT_TOKEN",
    "HOME_ASSISTANT_URL",
    "HOME_ASSISTANT_TOKEN",
    "BRAVE_SEARCH_API_KEY",
)


@pytest.fixture(autouse=True)
def _isolated_environment(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Remove every Jarvis-related variable so tests start from defaults."""
    for name in list(os.environ):
        if name.startswith("JARVIS_") or name in CREDENTIAL_VARIABLES:
            monkeypatch.delenv(name, raising=False)
    yield


@pytest.fixture
def make_settings() -> Callable[..., Settings]:
    """Factory for ``Settings`` that ignores any ``.env`` file on disk."""

    def factory(**overrides: Any) -> Settings:
        return Settings(_env_file=None, **overrides)

    return factory
