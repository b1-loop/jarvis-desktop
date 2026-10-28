import json
from collections.abc import Callable
from pathlib import Path

import httpx
import pytest

from jarvis.integrations.web import (
    DEFAULT_USER_AGENT,
    MAX_RESULTS_LIMIT,
    ArticleFetchError,
    BraveSearchProvider,
    DuckDuckGoSearchProvider,
    SearchResult,
    StubWebSearchProvider,
    WebSearchError,
    create_web_client,
    fetch_article,
    parse_duckduckgo_html,
)

FIXTURES = Path(__file__).parent / "fixtures"
DDG_HTML = (FIXTURES / "duckduckgo_results.html").read_text(encoding="utf-8")
ARTICLE_HTML = (FIXTURES / "article.html").read_text(encoding="utf-8")

Handler = Callable[[httpx.Request], httpx.Response]


def client_with(handler: Handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def respond(status: int, content: bytes | str, content_type: str = "text/html") -> Handler:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, content=content, headers={"content-type": content_type})

    return handler


# --- DuckDuckGo --------------------------------------------------------------


def test_parse_duckduckgo_fixture_unwraps_redirects_and_drops_ads() -> None:
    results = parse_duckduckgo_html(DDG_HTML)

    assert [result.url for result in results] == [
        "https://pybind11.readthedocs.io/en/stable/basics.html",
        "https://github.com/pybind/pybind11",
        "https://example.org/plain-link",
    ]
    assert results[0].title == "First steps — pybind11 documentation"
    assert results[0].snippet.startswith(
        "This sections demonstrates the basic features of pybind11."
    )
    assert results[2].snippet == ""


def test_parse_duckduckgo_tolerates_unexpected_markup() -> None:
    assert parse_duckduckgo_html("<html><body><p>Nothing here</p></body></html>") == []
    assert parse_duckduckgo_html("<div class='result__a'>no href<a class=result__a>") == []


async def test_duckduckgo_provider_sends_query_and_limits_results() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, text=DDG_HTML)

    async with client_with(handler) as client:
        results = await DuckDuckGoSearchProvider(client).search("pybind11 tutorial", limit=2)

    assert len(results) == 2
    assert seen[0].url.params["q"] == "pybind11 tutorial"
    assert seen[0].url.host == "html.duckduckgo.com"


async def test_duckduckgo_provider_clamps_limit() -> None:
    async with client_with(respond(200, DDG_HTML)) as client:
        provider = DuckDuckGoSearchProvider(client)
        assert len(await provider.search("x", limit=0)) == 1
        assert len(await provider.search("x", limit=MAX_RESULTS_LIMIT + 50)) == 3


async def test_duckduckgo_http_error_becomes_search_error() -> None:
    async with client_with(respond(429, "slow down")) as client:
        with pytest.raises(WebSearchError, match="DuckDuckGo"):
            await DuckDuckGoSearchProvider(client).search("anything")


async def test_duckduckgo_empty_page_returns_no_results() -> None:
    async with client_with(respond(200, "<html><body>No results.</body></html>")) as client:
        assert await DuckDuckGoSearchProvider(client).search("zzz") == []


# --- Brave -------------------------------------------------------------------

BRAVE_PAYLOAD = {
    "web": {
        "results": [
            {"title": "A", "url": "https://a.test/", "description": "first"},
            {"title": "B", "url": "https://b.test/"},
            {"title": "no url"},
        ]
    }
}


async def test_brave_provider_sends_token_header_and_maps_json() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=BRAVE_PAYLOAD)

    async with client_with(handler) as client:
        results = await BraveSearchProvider(client, "brave-token").search("query", limit=5)

    assert results == [
        SearchResult(title="A", url="https://a.test/", snippet="first"),
        SearchResult(title="B", url="https://b.test/", snippet=""),
    ]
    assert seen[0].headers["X-Subscription-Token"] == "brave-token"
    assert seen[0].url.params["count"] == "5"


async def test_brave_http_error_becomes_search_error() -> None:
    async with client_with(respond(401, "{}", "application/json")) as client:
        with pytest.raises(WebSearchError, match="Brave"):
            await BraveSearchProvider(client, "bad").search("query")


async def test_brave_non_json_and_wrong_shape_become_search_errors() -> None:
    async with client_with(respond(200, "<html>", "text/html")) as client:
        with pytest.raises(WebSearchError, match="non-JSON"):
            await BraveSearchProvider(client, "k").search("query")
    async with client_with(respond(200, json.dumps([1, 2]), "application/json")) as client:
        with pytest.raises(WebSearchError, match="unexpected payload"):
            await BraveSearchProvider(client, "k").search("query")


async def test_brave_without_web_section_returns_nothing() -> None:
    async with client_with(respond(200, json.dumps({"query": {}}), "application/json")) as client:
        assert await BraveSearchProvider(client, "k").search("query") == []


# --- Stub --------------------------------------------------------------------


async def test_stub_provider_returns_fixtures_and_records_queries() -> None:
    canned = [SearchResult("t", "https://t.test/", "s")] * 3
    stub = StubWebSearchProvider(canned)

    assert await stub.search("hello", limit=2) == canned[:2]
    assert stub.queries == ["hello"]
    assert await StubWebSearchProvider().search("empty") == []


# --- fetch_article -----------------------------------------------------------


async def test_fetch_article_extracts_readable_text() -> None:
    async with client_with(respond(200, ARTICLE_HTML)) as client:
        article = await fetch_article(client, "https://blog.test/audio-callback")

    assert article.url == "https://blog.test/audio-callback"
    assert article.title == "Why the audio callback must never block"
    assert article.byline == "Bozhidar Ivanov"
    assert article.published == "2026-10-20T09:30:00+02:00"
    assert not article.truncated
    assert article.text.startswith("Every audio API hands you a callback")
    assert "Push, do not process" in article.text
    assert "Dropped samples are counted" in article.text
    assert "The worker owns every model" in article.text
    assert "cookies" not in article.text
    assert "sidebar" not in article.text
    assert "All rights reserved" not in article.text
    assert "Tweet" not in article.text


async def test_fetch_article_follows_redirects_and_reports_final_url() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/old":
            return httpx.Response(301, headers={"location": "https://blog.test/new"})
        return httpx.Response(200, text=ARTICLE_HTML)

    async with client_with(handler) as client:
        article = await fetch_article(client, "https://blog.test/old")

    assert article.url == "https://blog.test/new"


async def test_fetch_article_caps_characters_on_paragraph_boundary() -> None:
    async with client_with(respond(200, ARTICLE_HTML)) as client:
        article = await fetch_article(client, "https://blog.test/", max_chars=400)

    assert article.truncated
    assert len(article.text) <= 400
    assert article.text.endswith("every single time.")


async def test_fetch_article_cuts_a_single_oversized_paragraph() -> None:
    html = "<html><body><article><p>" + "word " * 200 + "</p></article></body></html>"
    async with client_with(respond(200, html)) as client:
        article = await fetch_article(client, "https://blog.test/", max_chars=50)

    assert article.truncated
    assert len(article.text) == 50
    assert article.text.endswith("…")


async def test_fetch_article_caps_downloaded_bytes() -> None:
    body = ARTICLE_HTML.encode() + b"<p>" + b"x" * (1024 * 1024) + b"</p>"
    async with client_with(respond(200, body)) as client:
        article = await fetch_article(client, "https://blog.test/", max_bytes=len(ARTICLE_HTML))

    assert "xxxx" not in article.text


async def test_fetch_article_rejects_non_text_content() -> None:
    async with client_with(respond(200, b"%PDF-1.7", "application/pdf")) as client:
        with pytest.raises(ArticleFetchError, match="application/pdf"):
            await fetch_article(client, "https://blog.test/file.pdf")


async def test_fetch_article_reports_http_and_transport_errors() -> None:
    async with client_with(respond(404, "gone")) as client:
        with pytest.raises(ArticleFetchError, match="404"):
            await fetch_article(client, "https://blog.test/missing")

    def boom(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    async with client_with(boom) as client:
        with pytest.raises(ArticleFetchError, match="refused") as info:
            await fetch_article(client, "https://blog.test/")
    assert info.value.url == "https://blog.test/"


async def test_fetch_article_with_no_readable_body_is_an_error() -> None:
    html = "<html><body><nav><a href='/'>Home</a></nav></body></html>"
    async with client_with(respond(200, html)) as client:
        with pytest.raises(ArticleFetchError, match="no readable text"):
            await fetch_article(client, "https://blog.test/")


async def test_fetch_article_decodes_meta_charset_when_header_has_none() -> None:
    html = (
        '<html><head><meta charset="iso-8859-1"></head><body><article><p>'
        "Caf\xe9 au lait is a drink that people enjoy in the morning with a croissant."
        "</p></article></body></html>"
    ).encode("latin-1")

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=html, headers={"content-type": "text/html"})

    async with client_with(handler) as client:
        article = await fetch_article(client, "https://blog.test/")

    assert article.text.startswith("Café au lait")


async def test_fetch_article_falls_back_to_utf8_for_unknown_charset() -> None:
    html = '<html><head><meta charset="no-such-charset"></head><body><article><p>'
    html += "A paragraph long enough to be kept by the extractor." + "</p></article></body></html>"
    async with client_with(respond(200, html)) as client:
        article = await fetch_article(client, "https://blog.test/")

    assert article.text.startswith("A paragraph")


async def test_create_web_client_sets_polite_defaults() -> None:
    async with create_web_client() as client:
        assert client.headers["User-Agent"] == DEFAULT_USER_AGENT
        assert client.follow_redirects
        assert client.timeout.read is not None


async def test_web_client_sends_the_configured_headers_on_every_web_request() -> None:
    seen: list[httpx.Headers] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers)
        if request.url.host == "html.duckduckgo.com":
            return httpx.Response(200, text=DDG_HTML)
        return httpx.Response(200, text=ARTICLE_HTML, headers={"content-type": "text/html"})

    transport = httpx.MockTransport(handler)
    async with create_web_client("TestAgent/9.9", transport=transport) as client:
        await DuckDuckGoSearchProvider(client).search("pybind11")
        await fetch_article(client, "https://blog.test/audio-callback")

    assert [headers["User-Agent"] for headers in seen] == ["TestAgent/9.9"] * 2
    assert all(headers["Accept-Language"] == "en" for headers in seen)


async def test_create_web_client_accepts_a_full_timeout() -> None:
    async with create_web_client(timeout=httpx.Timeout(30.0, connect=10.0)) as client:
        assert client.timeout.connect == 10.0
        assert client.timeout.read == 30.0
