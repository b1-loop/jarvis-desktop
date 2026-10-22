from collections.abc import Callable
from pathlib import Path

import httpx
import pytest

from jarvis.agent.registry import ToolRegistry
from jarvis.config import Settings
from jarvis.integrations.services import IntegrationServices
from jarvis.tools.read_article import ReadArticleTool, local_address_reason
from tests.tools.integration_helpers import integration_context

ARTICLE_HTML = (Path(__file__).parents[1] / "integrations" / "fixtures" / "article.html").read_text(
    encoding="utf-8"
)


@pytest.fixture
def settings(make_settings: Callable[..., Settings]) -> Settings:
    return make_settings()


def serving(status: int, body: str, requests: list[str]) -> httpx.AsyncClient:
    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(str(request.url))
        return httpx.Response(status, text=body, headers={"content-type": "text/html"})

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def test_is_discovered_and_read_only() -> None:
    tool = ToolRegistry.discover().get("read_article")

    assert isinstance(tool, ReadArticleTool)
    assert not tool.requires_confirmation


async def test_returns_the_article_text(settings: Settings) -> None:
    requests: list[str] = []
    async with serving(200, ARTICLE_HTML, requests) as client:
        context = integration_context(settings, IntegrationServices(web_client=client))
        result = await ReadArticleTool().call({"url": "https://blog.test/audio"}, context)

    assert result.ok
    assert result.spoken.startswith(
        "I've got Why the audio callback must never block by Bozhidar Ivanov, about "
    )
    assert result.detail.startswith("Title: Why the audio callback must never block")
    assert "Every audio API hands you a callback" in result.detail
    assert result.data is not None
    assert result.data["truncated"] is False
    assert result.data["byline"] == "Bozhidar Ivanov"
    assert requests == ["https://blog.test/audio"]


async def test_respects_the_character_budget(settings: Settings) -> None:
    async with serving(200, ARTICLE_HTML, []) as client:
        context = integration_context(settings, IntegrationServices(web_client=client))
        result = await ReadArticleTool().call(
            {"url": "https://blog.test/audio", "max_chars": 1000}, context
        )

    assert result.data is not None
    assert result.data["truncated"] is True
    assert "cut to the character budget" in result.detail


async def test_http_error_is_spoken_plainly(settings: Settings) -> None:
    async with serving(404, "gone", []) as client:
        context = integration_context(settings, IntegrationServices(web_client=client))
        result = await ReadArticleTool().call({"url": "https://blog.test/missing"}, context)

    assert not result.ok
    assert result.spoken == "I couldn't read that page."
    assert "404" in result.detail


@pytest.mark.parametrize(
    "url",
    [
        "http://localhost:8123/api",
        "http://127.0.0.1/",
        "http://192.168.1.1/admin",
        "http://10.0.0.5/",
        "http://[::1]/",
        "http://169.254.169.254/latest/meta-data",
        "http://homeassistant.local/",
    ],
)
async def test_local_addresses_are_refused_without_a_request(settings: Settings, url: str) -> None:
    requests: list[str] = []
    async with serving(200, ARTICLE_HTML, requests) as client:
        context = integration_context(settings, IntegrationServices(web_client=client))
        result = await ReadArticleTool().call({"url": url}, context)

    assert not result.ok
    assert result.spoken == "I won't open addresses on your own network."
    assert requests == []


@pytest.mark.parametrize("url", ["https://example.org/a", "http://93.184.216.34/"])
def test_public_addresses_are_allowed(url: str) -> None:
    assert local_address_reason(url) is None


async def test_non_http_urls_are_rejected_by_the_schema(settings: Settings) -> None:
    context = integration_context(settings, IntegrationServices())

    result = await ReadArticleTool().call({"url": "file:///etc/passwd"}, context)

    assert not result.ok
    assert "url" in result.detail
