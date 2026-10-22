"""Ollama backed ``TextEmbedder`` over ``POST /api/embed``.

The same local server that runs the chat model also serves small
embedding models (``ollama pull bge-m3``). One request embeds a whole batch,
which is how the tool descriptions are embedded at startup; a query is a
batch of one. Tests drive the client through ``httpx.MockTransport``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Final

import httpx

from jarvis.llm.embeddings import (
    EmbeddingError,
    EmbeddingModelUnavailableError,
    TextEmbedder,
    Vector,
    normalised,
)

if TYPE_CHECKING:
    from jarvis.config import Settings

EMBED_PATH: Final = "/api/embed"
EMBED_TIMEOUT: Final = httpx.Timeout(30.0, connect=2.0)
"""Loading the model on first use takes seconds; a refused connection should fail fast.

Per-turn latency is bounded by the caller (``Settings.tool_embedding_timeout_ms``),
not by this timeout, so a slow first request can finish and warm the model up.
"""

KEEP_ALIVE: Final = "30m"
"""How long Ollama keeps the embedding model loaded after a request.

Longer than Ollama's 5-minute default so a quiet half hour does not cost a
model load (and therefore a keyword-only ranking) on the next request; the
model is small (under 1 GB of VRAM for ``bge-m3``).
"""

_PROMPTS: Final[dict[str, tuple[str, str]]] = {
    "embeddinggemma": ("task: search result | query: ", "title: none | text: "),
    "nomic-embed-text": ("search_query: ", "search_document: "),
}
"""(query, document) instruction prefixes for model families that are trained with them."""

_PERMANENT_STATUSES: Final = frozenset({400, 404, 501})
"""Unknown model (404), invalid request (400), or a model that cannot embed (501)."""


class OllamaEmbedder(TextEmbedder):
    """``TextEmbedder`` over Ollama's batch embedding endpoint.

    Args:
        http: Client whose ``base_url`` is the Ollama server.
        model: Embedding model tag, e.g. ``bge-m3``.
    """

    def __init__(self, http: httpx.AsyncClient, *, model: str) -> None:
        self._http = http
        self._model = model
        family = model.split(":", 1)[0]
        self._query_prefix, self._document_prefix = _PROMPTS.get(family, ("", ""))

    @classmethod
    def from_settings(cls, settings: Settings, model: str) -> OllamaEmbedder:
        """The real client, pointed at ``settings.ollama_base_url``."""
        http = httpx.AsyncClient(base_url=settings.ollama_base_url, timeout=EMBED_TIMEOUT)
        return cls(http, model=model)

    @property
    def model(self) -> str:
        """The Ollama embedding model tag."""
        return self._model

    async def aclose(self) -> None:
        """Release the HTTP connection pool."""
        await self._http.aclose()

    async def embed_documents(self, texts: Sequence[str]) -> list[Vector]:
        """Embed ``texts`` in one request; see :meth:`TextEmbedder.embed_documents`."""
        if not texts:
            return []
        return await self._embed([self._document_prefix + text for text in texts])

    async def embed_query(self, text: str) -> Vector:
        """Embed one query; see :meth:`TextEmbedder.embed_query`."""
        (vector,) = await self._embed([self._query_prefix + text])
        return vector

    async def _embed(self, inputs: list[str]) -> list[Vector]:
        payload = {"model": self._model, "input": inputs, "keep_alive": KEEP_ALIVE}
        try:
            response = await self._http.post(EMBED_PATH, json=payload)
            response.raise_for_status()
        except httpx.HTTPStatusError as error:
            raise self._status_error(error.response) from error
        except httpx.HTTPError as error:
            raise EmbeddingError(
                f"could not reach Ollama at {self._http.base_url} for embeddings: {error!r}"
            ) from error
        return _vectors(response, expected=len(inputs))

    def _status_error(self, response: httpx.Response) -> EmbeddingError:
        detail = _error_text(response)
        if response.status_code in _PERMANENT_STATUSES:
            return EmbeddingModelUnavailableError(
                f"Ollama cannot embed with '{self._model}' ({detail}); run "
                f"'ollama pull {self._model}' or change JARVIS_TOOL_EMBEDDING_MODEL"
            )
        return EmbeddingError(f"Ollama embeddings returned HTTP {response.status_code}: {detail}")


def _vectors(response: httpx.Response, *, expected: int) -> list[Vector]:
    try:
        body: Any = response.json()
    except ValueError as error:
        raise EmbeddingError("Ollama sent an embedding reply that is not JSON") from error
    embeddings = body.get("embeddings") if isinstance(body, dict) else None
    if not isinstance(embeddings, list) or len(embeddings) != expected:
        raise EmbeddingError(f"Ollama returned no usable embeddings for {expected} input(s)")
    return [normalised(values) for values in embeddings]


def _error_text(response: httpx.Response) -> str:
    try:
        body: Any = response.json()
    except ValueError:
        return response.text[:200]
    if isinstance(body, dict) and isinstance(body.get("error"), str):
        return str(body["error"])
    return response.text[:200]


# Design notes:
# - No retries here: the caller already budgets each request to a fraction
#   of a second and falls back to word overlap, so a retry loop would only
#   hold a connection open for an answer nobody is waiting for.
# - 404/400/501 are permanent (wrong or missing model) and say how to fix
#   it; everything else is transient. The caller reports permanent errors
#   once per session and retries transient ones after a pause.
# - Instruction prefixes are applied here, per model family, so the ranker
#   and its cache stay model-agnostic; the model tag is part of the cache key.
