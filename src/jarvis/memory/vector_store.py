"""Semantic memory: find past turns and notes by meaning rather than by words.

The agent asks "what do I know that relates to this?" and gets back the
nearest documents in embedding space. ``VectorStore`` is the Protocol the
rest of the package depends on; ``ChromaVectorStore`` is the persistent
implementation and ``InMemoryVectorStore`` is a dependency-free one used by
the test-suite and by ``--text-only`` runs that do not want ChromaDB.
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Protocol

import numpy as np
from numpy.typing import NDArray

from jarvis.memory.errors import MemoryStoreError, StoreClosedError
from jarvis.utils.errors import ConfigError
from jarvis.utils.logging import get_logger

log = get_logger(__name__)

MetadataValue = str | int | float | bool
Metadata = dict[str, MetadataValue]
"""Flat metadata; the scalar types every vector database accepts as filters."""

Vector = NDArray[np.float32]
Embedder = Callable[[str], Vector]

_WORD: Final = re.compile(r"\w+")
_HASH_BYTES: Final = 4
DEFAULT_EMBEDDING_DIM: Final = 256


@dataclass(frozen=True, slots=True)
class VectorHit:
    """One result of a similarity query.

    Attributes:
        id: Document id as passed to ``add``.
        text: The stored document.
        metadata: The stored metadata (empty dict when none was given).
        distance: Cosine distance in [0, 2]; lower is closer.
    """

    id: str
    text: str
    metadata: Metadata
    distance: float


class VectorStore(Protocol):
    """What the memory manager needs from a vector database."""

    async def add(self, doc_id: str, text: str, metadata: Metadata | None = None) -> None:
        """Insert or replace a document."""

    async def query(
        self, text: str, *, k: int = 5, where: Metadata | None = None
    ) -> list[VectorHit]:
        """Up to ``k`` nearest documents, closest first, matching all of ``where``."""

    async def delete(self, *doc_ids: str) -> None:
        """Remove documents; unknown ids are ignored."""

    async def count(self) -> int:
        """Number of stored documents."""


def hashed_bag_of_words(text: str, dim: int = DEFAULT_EMBEDDING_DIM) -> Vector:
    """A trivial, deterministic embedding: hashed word counts, L2-normalised.

    Good enough to make "my python project" closer to "the jarvis python
    code" than to "dinner on friday", which is all the tests need. It is
    not a substitute for a real model in production.
    """
    vector = np.zeros(dim, dtype=np.float32)
    for word in _WORD.findall(text.lower()):
        digest = hashlib.blake2b(word.encode("utf-8"), digest_size=_HASH_BYTES).digest()
        vector[int.from_bytes(digest, "little") % dim] += 1.0
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 0.0 else vector


def _matches(metadata: Metadata, where: Metadata | None) -> bool:
    return where is None or all(metadata.get(key) == value for key, value in where.items())


@dataclass(frozen=True, slots=True)
class _Entry:
    text: str
    metadata: Metadata
    vector: Vector


class InMemoryVectorStore:
    """Exact cosine search over a dict. For tests and ChromaDB-free runs."""

    def __init__(self, embed: Embedder = hashed_bag_of_words) -> None:
        self._embed = embed
        self._entries: dict[str, _Entry] = {}

    async def add(self, doc_id: str, text: str, metadata: Metadata | None = None) -> None:
        """Insert or replace a document."""
        self._entries[doc_id] = _Entry(text, dict(metadata or {}), self._embed(text))

    async def query(
        self, text: str, *, k: int = 5, where: Metadata | None = None
    ) -> list[VectorHit]:
        """Up to ``k`` nearest documents, closest first, matching all of ``where``."""
        needle = self._embed(text)
        hits = [
            VectorHit(doc_id, entry.text, dict(entry.metadata), 1.0 - float(entry.vector @ needle))
            for doc_id, entry in self._entries.items()
            if _matches(entry.metadata, where)
        ]
        hits.sort(key=lambda hit: hit.distance)
        return hits[:k]

    async def delete(self, *doc_ids: str) -> None:
        """Remove documents; unknown ids are ignored."""
        for doc_id in doc_ids:
            self._entries.pop(doc_id, None)

    async def count(self) -> int:
        """Number of stored documents."""
        return len(self._entries)


def _chroma_where(where: Metadata | None) -> dict[str, Any] | None:
    """Chroma wants a single ``{key: value}`` or an explicit ``$and`` list."""
    if not where:
        return None
    if len(where) == 1:
        return dict(where)
    return {"$and": [{key: value} for key, value in where.items()]}


class ChromaVectorStore:
    """Persistent semantic memory under ``data_dir`` using ChromaDB.

    Every ChromaDB call runs in a worker thread because the client is
    synchronous and the default embedding model takes tens of milliseconds
    per document. ChromaDB itself is imported lazily so the package imports
    without the ``memory`` extra installed.

    Args:
        directory: Where ChromaDB persists its files.
        collection_name: One collection holds turns and notes, told apart
            by their ``kind`` metadata.
        embedding_function: A ChromaDB ``EmbeddingFunction``; ``None`` uses
            Chroma's default (all-MiniLM-L6-v2 via ONNX, downloaded on first
            use).
    """

    def __init__(
        self,
        directory: Path,
        *,
        collection_name: str = "jarvis_memory",
        embedding_function: object | None = None,
    ) -> None:
        self._directory = directory
        self._collection_name = collection_name
        self._embedding_function = embedding_function
        self._collection: Any = None

    async def __aenter__(self) -> ChromaVectorStore:
        await self.open()
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.close()

    async def open(self) -> None:
        """Create the persistent client and collection. Idempotent.

        Raises:
            ConfigError: If ``chromadb`` is not installed.
            MemoryStoreError: If ChromaDB fails to open its files.
        """
        if self._collection is not None:
            return
        try:
            chromadb = importlib.import_module("chromadb")
        except ImportError as error:
            raise ConfigError(
                "chromadb is not installed; run 'uv sync --extra memory' or use the "
                "in-memory vector store."
            ) from error
        try:
            self._collection = await asyncio.to_thread(self._connect, chromadb)
        except Exception as error:
            raise MemoryStoreError(f"cannot open ChromaDB at {self._directory}: {error}") from error
        log.info("memory.chroma.opened", path=str(self._directory), count=await self.count())

    def _connect(self, chromadb: Any) -> Any:
        self._directory.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(
            path=str(self._directory),
            settings=chromadb.Settings(anonymized_telemetry=False),
        )
        options: dict[str, Any] = {"metadata": {"hnsw:space": "cosine"}}
        if self._embedding_function is not None:
            options["embedding_function"] = self._embedding_function
        return client.get_or_create_collection(self._collection_name, **options)

    async def close(self) -> None:
        """Drop the collection handle. ChromaDB flushes on every write."""
        self._collection = None

    def _require_open(self) -> Any:
        if self._collection is None:
            raise StoreClosedError("ChromaVectorStore is not open")
        return self._collection

    async def add(self, doc_id: str, text: str, metadata: Metadata | None = None) -> None:
        """Insert or replace a document."""
        collection = self._require_open()
        options: dict[str, Any] = {"ids": [doc_id], "documents": [text]}
        if metadata:
            options["metadatas"] = [dict(metadata)]
        await asyncio.to_thread(lambda: collection.upsert(**options))

    async def query(
        self, text: str, *, k: int = 5, where: Metadata | None = None
    ) -> list[VectorHit]:
        """Up to ``k`` nearest documents, closest first, matching all of ``where``."""
        collection = self._require_open()
        limit = min(k, await self.count())
        if limit <= 0:
            return []
        result = await asyncio.to_thread(
            lambda: collection.query(
                query_texts=[text],
                n_results=limit,
                where=_chroma_where(where),
                include=["documents", "metadatas", "distances"],
            )
        )
        return _hits_from_result(result)

    async def delete(self, *doc_ids: str) -> None:
        """Remove documents; unknown ids are ignored."""
        collection = self._require_open()
        if doc_ids:
            await asyncio.to_thread(lambda: collection.delete(ids=list(doc_ids)))

    async def count(self) -> int:
        """Number of stored documents."""
        collection = self._require_open()
        return int(await asyncio.to_thread(collection.count))


def _hits_from_result(result: Any) -> list[VectorHit]:
    ids = result["ids"][0]
    documents = result["documents"][0]
    metadatas = result["metadatas"][0]
    distances = result["distances"][0]
    return [
        VectorHit(
            id=str(doc_id),
            text=str(document),
            metadata=dict(metadata or {}),
            distance=float(distance),
        )
        for doc_id, document, metadata, distance in zip(
            ids, documents, metadatas, distances, strict=True
        )
    ]


# Design notes:
# - The Protocol is tiny on purpose: add, query, delete, count. Anything the
#   agent wants beyond nearest-neighbour search (filtering by date, by kind)
#   is expressed through metadata, which every candidate backend supports.
# - ChromaDB is typed as Any here: its own type hints are partial and the
#   package is optional, so the mypy override in pyproject ignores it. The
#   boundary is kept to this one class, with tests against the real library
#   marked ``integration``.
# - Cosine space rather than Chroma's default L2 so that distances from
#   both implementations mean the same thing and the budget logic in the
#   manager can treat them alike.
# - Assumption: one collection for turns and notes. Chroma's per-collection
#   overhead is small, and a single query across both kinds is what recall
#   wants; the ``kind`` metadata keeps them separable.
