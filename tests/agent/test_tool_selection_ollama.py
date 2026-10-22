"""Hybrid tool selection against a real Ollama embedding model (ADR-0020, update).

Needs Ollama on ``localhost:11434`` with the default embedding model pulled
(``ollama pull bge-m3``); skipped otherwise, so the pure suite never
depends on a server. The thresholds sit a little below the measured rates
because GPU embeddings are not bit-for-bit reproducible between runs.
"""

from collections.abc import AsyncIterator, Mapping
from pathlib import Path
from typing import Any, Final

import httpx
import pytest
import yaml

from jarvis.agent.registry import AnyTool, ToolRegistry
from jarvis.agent.tool_embeddings import EmbeddingToolRanker, ToolVectorCache
from jarvis.agent.tool_selection import RelevanceToolSelector
from jarvis.config import DEFAULT_TOOL_EMBEDDING_MODEL, Settings
from jarvis.llm.ollama_embeddings import OllamaEmbedder

pytestmark = pytest.mark.integration

OLLAMA_URL: Final = "http://localhost:11434"
CASES_FILE: Final = Path(__file__).parent / "data" / "tool_selection_cases.yaml"
GENEROUS_BUDGET_SECONDS: Final = 30.0
"""The test measures ranking quality, not latency; a cold model may take seconds to load."""

GROUPS: Final = {"en": ("en", False), "sv": ("sv", False), "sv-held-out": ("sv", True)}
MIN_SELECTED: Final = {"en": 0.97, "sv": 0.97, "sv-held-out": 0.93}
"""Measured 1.00 in every group with the defaults."""
MIN_TOP_3: Final = {"en": 0.9, "sv": 0.9, "sv-held-out": 0.85}
"""Measured 1.00, 1.00 and 0.93."""


def _model_is_pulled(model: str) -> bool:
    try:
        response = httpx.get(f"{OLLAMA_URL}/api/tags", timeout=2.0)
        response.raise_for_status()
    except httpx.HTTPError:
        return False
    tags = {entry.get("name", "") for entry in response.json().get("models", [])}
    return any(tag.split(":", 1)[0] == model for tag in tags)


@pytest.fixture(scope="module")
def cases() -> list[dict[str, Any]]:
    loaded: dict[str, list[dict[str, Any]]] = yaml.safe_load(CASES_FILE.read_text("utf-8"))
    return loaded["cases"]


@pytest.fixture
async def ranker(tmp_path: Path) -> AsyncIterator[EmbeddingToolRanker]:
    if not _model_is_pulled(DEFAULT_TOOL_EMBEDDING_MODEL):
        pytest.skip(f"Ollama with '{DEFAULT_TOOL_EMBEDDING_MODEL}' is not available")
    http = httpx.AsyncClient(base_url=OLLAMA_URL, timeout=60.0)
    embedder = OllamaEmbedder(http, model=DEFAULT_TOOL_EMBEDDING_MODEL)
    ranker = EmbeddingToolRanker(
        embedder,
        cache=ToolVectorCache(tmp_path / "tool_embeddings.json"),
        budget_seconds=GENEROUS_BUDGET_SECONDS,
    )
    yield ranker
    await ranker.aclose()
    await embedder.aclose()


async def similarities_per_case(
    ranker: EmbeddingToolRanker, tools: list[AnyTool], cases: list[dict[str, Any]]
) -> list[Mapping[str, float]]:
    assert await ranker.prepare(tools)
    found = [await ranker.similarities(case["request"], tools) for case in cases]
    assert all(values is not None for values in found)
    return [values for values in found if values is not None]


def rate(hits: list[bool]) -> float:
    return sum(hits) / len(hits)


async def test_hybrid_selection_finds_swedish_and_english_tools(
    ranker: EmbeddingToolRanker,
    cases: list[dict[str, Any]],
    make_settings: Any,
) -> None:
    tools = ToolRegistry.discover().tools()
    settings: Settings = make_settings()
    selector = RelevanceToolSelector.from_settings(settings, ranker)
    similarities = await similarities_per_case(ranker, tools, cases)

    for group, (lang, holdout) in GROUPS.items():
        members = [
            index
            for index, case in enumerate(cases)
            if case.get("lang", "en") == lang and bool(case.get("holdout")) == holdout
        ]
        selected, top_3 = [], []
        for index in members:
            request, expected = cases[index]["request"], set(cases[index]["expected"])
            chosen = selector.choose(request, tools, similarities=similarities[index])
            ranked = selector.rank(request, tools, similarities[index])[:3]
            selected.append(bool(expected & {tool.name for tool in chosen}))
            top_3.append(bool(expected & {tool.name for tool in ranked}))

        assert rate(selected) >= MIN_SELECTED[group], group
        assert rate(top_3) >= MIN_TOP_3[group], group


async def test_the_reported_swedish_failures_now_offer_the_right_tool(
    ranker: EmbeddingToolRanker, make_settings: Any
) -> None:
    tools = ToolRegistry.discover().tools()
    selector = RelevanceToolSelector.from_settings(make_settings(), ranker)
    await ranker.prepare(tools)

    for request, expected in (
        ("hur mycket batteri har jag?", "battery_status"),
        ("lägg till mjölk på att-göra-listan", "add_todo"),
        ("vad står på min att-göra-lista?", "list_todos"),
    ):
        chosen = await selector.select(request, tools)

        assert expected in {tool.name for tool in chosen}, request
