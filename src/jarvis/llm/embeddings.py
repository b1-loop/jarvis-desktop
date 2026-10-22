"""``TextEmbedder`` interface: turn text into vectors for similarity search.

The agent's tool selection (ADR-0020, update) compares a request with every
tool's description in embedding space, which works across languages when
the embedding model is multilingual. Only this Protocol and its errors are
shared; :mod:`jarvis.llm.ollama_embeddings` is the one implementation.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from jarvis.utils.errors import JarvisError

Vector = NDArray[np.float32]
"""One embedding, L2-normalised so a dot product is the cosine similarity."""


class EmbeddingError(JarvisError):
    """The embedding service failed this time (unreachable, overloaded, bad reply).

    Transient: the caller may try again later.
    """


class EmbeddingModelUnavailableError(EmbeddingError):
    """The configured model is not there or cannot embed; retrying will not help.

    The message names the fix (``ollama pull <model>`` or another model).
    """


class TextEmbedder(Protocol):
    """Embeds search documents and queries with one model.

    Documents and queries are separate calls because several retrieval
    models expect a different instruction prefix for each.
    """

    @property
    def model(self) -> str:
        """Identifier of the embedding model; part of every cache key."""

    async def embed_documents(self, texts: Sequence[str]) -> list[Vector]:
        """One normalised vector per text, in order.

        Raises:
            EmbeddingError: When the service fails.
        """

    async def embed_query(self, text: str) -> Vector:
        """The normalised vector of one search query.

        Raises:
            EmbeddingError: When the service fails.
        """


def normalised(values: Sequence[float]) -> Vector:
    """``values`` as a unit-length float32 vector (a zero vector stays zero)."""
    vector = np.asarray(values, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 0 else vector


# Design notes:
# - Vectors are normalised once, where they are made, so every consumer can
#   use a plain dot product and cached vectors need no extra bookkeeping.
# - Two error classes because the caller reacts differently: a missing
#   model is reported once and never retried in the session, while a
#   timeout or connection failure is retried after a pause.
