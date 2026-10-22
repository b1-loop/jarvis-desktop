import asyncio
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest
from structlog.testing import capture_logs

from jarvis.agent.registry import AnyTool
from jarvis.agent.tool_embeddings import (
    EmbeddingToolRanker,
    ToolVectorCache,
    document_key,
    tool_document,
)
from jarvis.llm.embeddings import (
    EmbeddingError,
    EmbeddingModelUnavailableError,
    Vector,
    normalised,
)
from tests.agent.fakes import DestructiveTool, EchoTool, SendTool

AXES = ("repeat", "send", "delete")
"""Each fake vector has one dimension per idea, so similarities are easy to read."""


def vector_for(text: str) -> Vector:
    lowered = text.lower()
    hits = [1.0 if axis in lowered else 0.0 for axis in AXES]
    return normalised([*hits, 0.1])


class FakeEmbedder:
    """A ``TextEmbedder`` whose speed and failures a test controls."""

    def __init__(self, model: str = "fake-embed") -> None:
        self._model = model
        self.documents: list[str] = []
        self.queries: list[str] = []
        self.gate: asyncio.Event | None = None
        self.failure: EmbeddingError | None = None

    @property
    def model(self) -> str:
        return self._model

    async def embed_documents(self, texts: Sequence[str]) -> list[Vector]:
        self._raise_if_failing()
        self.documents.extend(texts)
        return [vector_for(text) for text in texts]

    async def embed_query(self, text: str) -> Vector:
        self.queries.append(text)
        if self.gate is not None:
            await self.gate.wait()
        self._raise_if_failing()
        return vector_for(text)

    def _raise_if_failing(self) -> None:
        if self.failure is not None:
            raise self.failure


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def tools() -> list[AnyTool]:
    return [EchoTool(), SendTool(), DestructiveTool()]


def make_ranker(
    embedder: FakeEmbedder, path: Path | None = None, **kwargs: object
) -> EmbeddingToolRanker:
    cache = ToolVectorCache(path) if path is not None else None
    return EmbeddingToolRanker(embedder, cache=cache, **kwargs)  # type: ignore[arg-type]


def test_a_tool_is_embedded_as_name_description_and_keywords() -> None:
    class Keyworded(EchoTool):
        name = "echo_twice"
        keywords = ("upprepa", "eko")

    assert tool_document(EchoTool()) == "echo: Repeat a piece of text."
    assert tool_document(Keyworded()) == (
        "echo twice: Repeat a piece of text. Keywords: upprepa, eko."
    )


async def test_similarities_are_cosines_per_tool(tools: list[AnyTool]) -> None:
    ranker = make_ranker(FakeEmbedder())
    assert await ranker.prepare(tools)

    result = await ranker.similarities("please repeat that", tools)

    assert result is not None
    assert max(result, key=lambda name: result[name]) == "echo"
    assert result["echo"] == pytest.approx(float(vector_for("repeat") @ vector_for("x repeat")))


async def test_tool_vectors_persist_and_are_not_re_embedded(
    tools: list[AnyTool], tmp_path: Path
) -> None:
    path = tmp_path / "cache" / "tool_embeddings.json"
    first = FakeEmbedder()
    await make_ranker(first, path).prepare(tools)
    second = FakeEmbedder()

    assert await make_ranker(second, path).prepare(tools)

    assert len(first.documents) == len(tools)
    assert second.documents == []
    assert path.exists()


async def test_a_changed_model_or_description_is_re_embedded(
    tools: list[AnyTool], tmp_path: Path
) -> None:
    class Renamed(EchoTool):
        description = "Say the text again."

    path = tmp_path / "tool_embeddings.json"
    await make_ranker(FakeEmbedder(), path).prepare(tools)
    other_model = FakeEmbedder(model="other-embed")
    await make_ranker(other_model, path).prepare(tools)
    edited = FakeEmbedder(model="other-embed")

    await make_ranker(edited, path).prepare([Renamed(), *tools[1:]])

    assert len(other_model.documents) == len(tools)
    assert edited.documents == ["echo: Say the text again."]


async def test_the_saved_cache_holds_only_the_current_tools(
    tools: list[AnyTool], tmp_path: Path
) -> None:
    path = tmp_path / "tool_embeddings.json"
    embedder = FakeEmbedder()
    await make_ranker(embedder, path).prepare(tools)

    await make_ranker(embedder, path).prepare(tools[:1])

    stored = ToolVectorCache(path).load()
    assert set(stored) == {document_key("fake-embed", tool_document(tools[0]))}


@pytest.mark.parametrize("content", ["{not json", '{"format": 99, "vectors": {}}', "[1, 2]"])
def test_an_unreadable_or_foreign_cache_is_empty(tmp_path: Path, content: str) -> None:
    path = tmp_path / "tool_embeddings.json"
    path.write_text(content, "utf-8")

    assert ToolVectorCache(path).load() == {}


async def test_before_warm_up_finishes_the_turn_ranks_by_words(tools: list[AnyTool]) -> None:
    ranker = make_ranker(FakeEmbedder())

    assert await ranker.similarities("repeat", tools) is None
    await asyncio.sleep(0)  # the warm-up task it started runs

    assert await ranker.similarities("repeat", tools) is not None
    await ranker.aclose()


async def test_a_late_request_embedding_is_skipped_then_reused(tools: list[AnyTool]) -> None:
    embedder = FakeEmbedder()
    ranker = make_ranker(embedder, budget_seconds=0.01)
    await ranker.prepare(tools)
    embedder.gate = asyncio.Event()

    late = await ranker.similarities("repeat", tools)
    embedder.gate.set()
    await asyncio.sleep(0)
    later = await ranker.similarities("repeat", tools)

    assert late is None
    assert later is not None
    assert embedder.queries == ["repeat"]  # the late one finished and was memoised


async def test_a_missing_model_is_reported_once_and_never_asked_again(
    tools: list[AnyTool],
) -> None:
    embedder = FakeEmbedder()
    embedder.failure = EmbeddingModelUnavailableError("run 'ollama pull fake-embed'")
    ranker = make_ranker(embedder)

    with capture_logs() as logs:
        assert not await ranker.prepare(tools)
        assert await ranker.similarities("repeat", tools) is None
        assert await ranker.similarities("send it", tools) is None

    warnings = [entry for entry in logs if entry["event"] == "tool_selection.semantic_unavailable"]
    assert len(warnings) == 1
    assert warnings[0]["permanent"] is True
    assert embedder.queries == []


async def test_a_transient_failure_backs_off_then_recovers(tools: list[AnyTool]) -> None:
    embedder = FakeEmbedder()
    clock = Clock()
    ranker = make_ranker(embedder, retry_after_seconds=60.0, clock=clock)
    await ranker.prepare(tools)
    embedder.failure = EmbeddingError("connection refused")

    with capture_logs() as logs:
        assert await ranker.similarities("repeat", tools) is None
        clock.now = 30.0
        assert await ranker.similarities("repeat", tools) is None
        embedder.failure = None
        clock.now = 61.0
        recovered = await ranker.similarities("repeat", tools)

    assert recovered is not None
    assert embedder.queries == ["repeat", "repeat"]  # nothing asked during the pause
    events = [entry["event"] for entry in logs]
    assert events.count("tool_selection.semantic_unavailable") == 1
    assert "tool_selection.semantic_recovered" in events


async def test_close_cancels_pending_embeddings(tools: list[AnyTool]) -> None:
    embedder = FakeEmbedder()
    ranker = make_ranker(embedder, budget_seconds=0.01)
    await ranker.prepare(tools)
    embedder.gate = asyncio.Event()
    await ranker.similarities("repeat", tools)

    await ranker.aclose()

    assert all(task.done() for task in ranker._requests.values())


def test_cached_vectors_round_trip(tmp_path: Path) -> None:
    cache = ToolVectorCache(tmp_path / "deep" / "cache.json")
    vector = normalised([1.0, 2.0, 2.0])

    cache.save({"k": vector})

    np.testing.assert_allclose(cache.load()["k"], vector, rtol=1e-6)
