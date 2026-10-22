from collections.abc import Callable

import pytest

from jarvis.agent.registry import ToolRegistry
from jarvis.config import Settings
from jarvis.integrations.services import IntegrationServices
from jarvis.integrations.web import SearchResult, StubWebSearchProvider, WebSearchError
from jarvis.tools.web_search import WebSearchTool
from tests.tools.integration_helpers import integration_context

RESULTS = [
    SearchResult("Miniaudio docs", "https://www.miniaud.io/docs/", "A single-file audio library."),
    SearchResult("Ring buffers", "https://example.org/ring", "Lock-free SPSC queues."),
]


class FailingSearch:
    async def search(self, query: str, limit: int = 5) -> list[SearchResult]:
        raise WebSearchError("DuckDuckGo request failed: 503")


@pytest.fixture
def settings(make_settings: Callable[..., Settings]) -> Settings:
    return make_settings()


def test_is_discovered_and_read_only() -> None:
    tool = ToolRegistry.discover().get("web_search")

    assert isinstance(tool, WebSearchTool)
    assert not tool.requires_confirmation


async def test_returns_ranked_results(settings: Settings) -> None:
    provider = StubWebSearchProvider(RESULTS)
    context = integration_context(settings, IntegrationServices(web_search=provider))

    result = await WebSearchTool().call({"query": "miniaudio", "limit": 2}, context)

    assert result.ok
    assert result.spoken == "I found 2 results; the top one is Miniaudio docs on miniaud.io."
    assert "1. Miniaudio docs\n   https://www.miniaud.io/docs/" in result.detail
    assert result.data == {
        "query": "miniaudio",
        "results": [{"title": r.title, "url": r.url, "snippet": r.snippet} for r in RESULTS],
    }
    assert provider.queries == ["miniaudio"]


async def test_no_results(settings: Settings) -> None:
    context = integration_context(settings, IntegrationServices(web_search=StubWebSearchProvider()))

    result = await WebSearchTool().call({"query": "zzzz"}, context)

    assert result.ok
    assert result.spoken == "I found nothing on the web for zzzz."


async def test_backend_failure_is_a_failed_result(settings: Settings) -> None:
    context = integration_context(settings, IntegrationServices(web_search=FailingSearch()))

    result = await WebSearchTool().call({"query": "x"}, context)

    assert not result.ok
    assert "503" in result.detail


async def test_limit_is_bounded(settings: Settings) -> None:
    context = integration_context(settings, IntegrationServices(web_search=StubWebSearchProvider()))

    result = await WebSearchTool().call({"query": "x", "limit": 50}, context)

    assert not result.ok
    assert "limit" in result.detail
