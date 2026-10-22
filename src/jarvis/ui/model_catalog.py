"""Choices the settings page offers: edge-tts voices and installed Ollama models.

The voices are a short curated list (Swedish and English, the languages
Jarvis is used in) rather than edge-tts's full catalogue of 400+ voices,
which would need a network round trip and does not fit a HUD combo box.
Ollama models are listed live from ``GET /api/tags`` each time the settings
page opens, because the user installs and removes models outside Jarvis.
"""

from __future__ import annotations

from typing import Final

import httpx

from jarvis.utils.errors import JarvisError
from jarvis.utils.logging import get_logger

log = get_logger(__name__)

EDGE_VOICES: Final[tuple[tuple[str, str], ...]] = (
    ("sv-SE-MattiasNeural", "Svenska - Mattias"),
    ("sv-SE-SofieNeural", "Svenska - Sofie"),
    ("en-GB-RyanNeural", "English (UK) - Ryan"),
    ("en-GB-SoniaNeural", "English (UK) - Sonia"),
    ("en-GB-ThomasNeural", "English (UK) - Thomas"),
    ("en-US-GuyNeural", "English (US) - Guy"),
    ("en-US-JennyNeural", "English (US) - Jenny"),
    ("en-US-AriaNeural", "English (US) - Aria"),
)
"""``(voice id, label)`` pairs offered for ``tts_voice``."""

STT_LANGUAGES: Final[tuple[tuple[str, str], ...]] = (
    ("auto", "Detect automatically"),
    ("sv", "Svenska"),
    ("en", "English"),
)
"""``stt_language`` choices; ``auto`` is stored as no language (None)."""

TAGS_PATH: Final = "/api/tags"
TAGS_TIMEOUT_SECONDS: Final = 3.0
"""Ollama answers from memory; three seconds means it is not running."""


class ModelCatalogError(JarvisError):
    """The Ollama model list could not be fetched or understood."""


def with_current(choices: tuple[tuple[str, str], ...], current: str) -> list[tuple[str, str]]:
    """``choices`` plus ``current`` at the end when it is not one of them.

    A value set by hand in ``.env`` (another voice, a model that has since
    been removed) must still show as selected instead of silently changing.
    """
    values = [value for value, _ in choices]
    listed = list(choices)
    if current and current not in values:
        listed.append((current, f"{current} (current)"))
    return listed


class OllamaModels:
    """Lists the models installed in an Ollama server.

    Args:
        client: The HTTP client (an ``httpx.MockTransport`` one in tests).
        base_url: The server, e.g. ``http://localhost:11434``.
    """

    def __init__(self, client: httpx.AsyncClient, base_url: str) -> None:
        self._client = client
        self._url = base_url.rstrip("/") + TAGS_PATH

    async def names(self) -> list[str]:
        """Installed model names, sorted, e.g. ``["llama3.2:3b", "qwen2.5:7b"]``.

        Raises:
            ModelCatalogError: If Ollama is not reachable or answers nonsense.
        """
        try:
            response = await self._client.get(self._url, timeout=TAGS_TIMEOUT_SECONDS)
            response.raise_for_status()
            payload = response.json()
        except (httpx.HTTPError, ValueError) as error:
            raise ModelCatalogError(f"Ollama model list unavailable: {error}") from error
        return _names_from(payload)


def _names_from(payload: object) -> list[str]:
    if not isinstance(payload, dict) or not isinstance(payload.get("models"), list):
        raise ModelCatalogError("Ollama /api/tags did not return a model list")
    names = [
        str(model["name"])
        for model in payload["models"]
        if isinstance(model, dict) and isinstance(model.get("name"), str)
    ]
    return sorted(names)


# Design notes:
# - The client is injected (Protocol-free: httpx.AsyncClient is already a
#   seam through its transport), so tests use httpx.MockTransport and never
#   reach a real server.
# - A failure is an exception, not an empty list: "no models installed" and
#   "Ollama is not running" need different notes on the settings page.
