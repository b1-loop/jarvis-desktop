"""Contract tests for ``VectorStore`` implementations.

The in-memory store always runs. ``ChromaVectorStore`` runs the same
contract when ``chromadb`` is importable, marked ``integration``, with a
registered hashing embedding function so no model is downloaded.
"""

from __future__ import annotations

import importlib
import importlib.util
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from jarvis.memory.errors import StoreClosedError
from jarvis.memory.vector_store import (
    ChromaVectorStore,
    InMemoryVectorStore,
    VectorStore,
    hashed_bag_of_words,
)
from jarvis.utils.errors import ConfigError

HAS_CHROMA = importlib.util.find_spec("chromadb") is not None

if HAS_CHROMA:  # guarded so collecting this module never requires chromadb
    import chromadb
    from chromadb.api.types import Documents, Embeddings
    from chromadb.utils.embedding_functions import register_embedding_function


def hashing_embedding_function() -> Any:
    """A ChromaDB ``EmbeddingFunction`` over ``hashed_bag_of_words``."""

    @register_embedding_function
    class HashingEmbeddingFunction(chromadb.EmbeddingFunction[Documents]):  # type: ignore[misc]
        def __init__(self) -> None:
            pass

        def __call__(self, input: Documents) -> Embeddings:  # noqa: A002  (Chroma's name)
            return [hashed_bag_of_words(text) for text in input]

        @staticmethod
        def name() -> str:
            return "jarvis_test_hashing"

        @staticmethod
        def build_from_config(_config: dict[str, Any]) -> HashingEmbeddingFunction:
            return HashingEmbeddingFunction()

        def get_config(self) -> dict[str, Any]:
            return {}

    return HashingEmbeddingFunction()


@pytest.fixture(
    params=[
        "memory",
        pytest.param(
            "chroma",
            marks=[
                pytest.mark.integration,
                pytest.mark.skipif(not HAS_CHROMA, reason="chromadb not installed"),
            ],
        ),
    ]
)
async def store(request: pytest.FixtureRequest, tmp_path: Path) -> AsyncIterator[VectorStore]:
    if request.param == "memory":
        yield InMemoryVectorStore()
        return
    async with ChromaVectorStore(
        tmp_path / "chroma", embedding_function=hashing_embedding_function()
    ) as chroma:
        yield chroma


async def test_empty_store(store: VectorStore) -> None:
    assert await store.count() == 0
    assert await store.query("anything") == []


async def test_nearest_document_comes_first(store: VectorStore) -> None:
    await store.add("code", "the jarvis python project", {"kind": "turn"})
    await store.add("food", "dinner at the italian place on friday", {"kind": "note"})

    hits = await store.query("python project", k=2)

    assert [hit.id for hit in hits] == ["code", "food"]
    assert hits[0].text == "the jarvis python project"
    assert hits[0].metadata == {"kind": "turn"}
    assert hits[0].distance < hits[1].distance


async def test_k_limits_results(store: VectorStore) -> None:
    for index in range(4):
        await store.add(str(index), f"document number {index}")

    assert len(await store.query("document", k=2)) == 2
    assert len(await store.query("document", k=10)) == 4


async def test_where_filters_on_every_key(store: VectorStore) -> None:
    await store.add("a", "same words", {"kind": "turn", "conversation_id": 1})
    await store.add("b", "same words", {"kind": "turn", "conversation_id": 2})
    await store.add("c", "same words", {"kind": "note", "conversation_id": 1})

    assert [h.id for h in await store.query("same words", where={"kind": "note"})] == ["c"]
    both = await store.query("same words", where={"kind": "turn", "conversation_id": 1})
    assert [h.id for h in both] == ["a"]


async def test_add_replaces_an_existing_id(store: VectorStore) -> None:
    await store.add("x", "first version")
    await store.add("x", "second version", {"v": 2})

    assert await store.count() == 1
    hits = await store.query("second version")
    assert hits[0].text == "second version"
    assert hits[0].metadata == {"v": 2}


async def test_delete_ignores_unknown_ids(store: VectorStore) -> None:
    await store.add("keep", "stays")
    await store.add("drop", "goes")

    await store.delete("drop", "never-existed")
    await store.delete()

    assert await store.count() == 1
    assert [h.id for h in await store.query("goes")] == ["keep"]


async def test_documents_without_metadata_come_back_with_an_empty_dict(
    store: VectorStore,
) -> None:
    await store.add("bare", "no metadata here")

    assert (await store.query("metadata"))[0].metadata == {}


# -- embedding and Chroma specifics ----------------------------------------


def test_hashed_bag_of_words_is_normalised_and_deterministic() -> None:
    first = hashed_bag_of_words("Hello, hello world")
    second = hashed_bag_of_words("hello world HELLO")

    assert np.allclose(first, second)
    assert np.linalg.norm(first) == pytest.approx(1.0)
    assert not hashed_bag_of_words("").any()


async def test_chroma_store_refuses_use_before_open(tmp_path: Path) -> None:
    store = ChromaVectorStore(tmp_path / "chroma")

    with pytest.raises(StoreClosedError):
        await store.count()


async def test_missing_chromadb_is_a_config_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def no_chroma(name: str) -> Any:
        raise ImportError(f"No module named {name!r}")

    monkeypatch.setattr(importlib, "import_module", no_chroma)

    with pytest.raises(ConfigError, match="uv sync --extra memory"):
        await ChromaVectorStore(tmp_path / "chroma").open()


@pytest.mark.integration
@pytest.mark.skipif(not HAS_CHROMA, reason="chromadb not installed")
async def test_chroma_store_persists_across_reopen(tmp_path: Path) -> None:
    directory = tmp_path / "chroma"
    async with ChromaVectorStore(
        directory, embedding_function=hashing_embedding_function()
    ) as store:
        await store.add("persisted", "survives a restart", {"kind": "note"})

    async with ChromaVectorStore(
        directory, embedding_function=hashing_embedding_function()
    ) as store:
        assert await store.count() == 1
        assert (await store.query("restart"))[0].id == "persisted"
