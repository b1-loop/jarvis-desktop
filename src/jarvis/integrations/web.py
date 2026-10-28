"""Web search and article reading.

Two capabilities the agent needs for "look that up" and "read me that
article": a :class:`WebSearchProvider` that returns ranked links with
snippets, and :func:`fetch_article`, which downloads a page and reduces it
to readable text through :mod:`jarvis.integrations.html_text`.

All HTTP goes through an injected ``httpx.AsyncClient`` so tests use
``httpx.MockTransport`` and production uses :func:`create_web_client`, which
sets the polite defaults (identifying User-Agent, timeouts, redirects). The
default search provider needs no API key; a paid one is available behind an
optional setting. The trade-offs are recorded in ADR-0012.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Final, Protocol
from urllib.parse import parse_qs, urlsplit

import httpx

from jarvis.integrations.html_text import extract_readable
from jarvis.utils.errors import JarvisError
from jarvis.utils.logging import get_logger

log = get_logger(__name__)

DEFAULT_USER_AGENT: Final = "JarvisDesktop/0.1 (+https://github.com/b1-loop/jarvis-desktop)"
DEFAULT_TIMEOUT_SECONDS: Final = 15.0
DEFAULT_MAX_RESULTS: Final = 5
MAX_RESULTS_LIMIT: Final = 20
DEFAULT_MAX_PAGE_BYTES: Final = 2 * 1024 * 1024
DEFAULT_MAX_ARTICLE_CHARS: Final = 20_000

_DUCKDUCKGO_URL: Final = "https://html.duckduckgo.com/html/"
_BRAVE_URL: Final = "https://api.search.brave.com/res/v1/web/search"
_READABLE_CONTENT_TYPES: Final = ("text/html", "application/xhtml+xml", "text/plain")
_META_CHARSET: Final = re.compile(rb'charset=["\']?([A-Za-z0-9_-]+)', re.IGNORECASE)
_CHUNK_BYTES: Final = 64 * 1024
_ELLIPSIS: Final = "…"


class WebError(JarvisError):
    """Base class for web search and fetch failures."""


class WebSearchError(WebError):
    """A search backend was unreachable, refused the request or returned garbage."""


class ArticleFetchError(WebError):
    """A page could not be downloaded or holds nothing readable.

    Attributes:
        url: The requested URL, for the spoken error message.
    """

    def __init__(self, url: str, reason: str) -> None:
        self.url = url
        super().__init__(f"could not read {url}: {reason}")


@dataclass(frozen=True, slots=True)
class SearchResult:
    """One hit from a web search.

    Attributes:
        title: Page title as the engine shows it.
        url: Destination URL, already unwrapped from any redirect.
        snippet: The engine's summary line; may be empty.
    """

    title: str
    url: str
    snippet: str


@dataclass(frozen=True, slots=True)
class Article:
    """Readable text extracted from a web page.

    Attributes:
        url: Final URL after redirects.
        title: Headline, or empty if none could be found.
        text: Body paragraphs separated by blank lines.
        byline: Author line if the page states one.
        published: Publication timestamp as the page states it.
        truncated: True if ``text`` was cut to the character cap.
    """

    url: str
    title: str
    text: str
    byline: str | None
    published: str | None
    truncated: bool


class WebSearchProvider(Protocol):
    """Anything that can turn a query into ranked links."""

    async def search(self, query: str, limit: int = DEFAULT_MAX_RESULTS) -> list[SearchResult]:
        """Return up to ``limit`` results, best first.

        Raises:
            WebSearchError: If the backend fails; an empty list means a
                successful search with no hits.
        """


class StubWebSearchProvider:
    """Returns canned results. Used in tests and when no network is wanted.

    This is a stub: it never performs a search. Results come from the fixture
    list given to the constructor, filtered to the requested limit.
    """

    def __init__(self, results: list[SearchResult] | None = None) -> None:
        self._results = list(results or [])
        self.queries: list[str] = []

    async def search(self, query: str, limit: int = DEFAULT_MAX_RESULTS) -> list[SearchResult]:
        """Record the query and return the canned results."""
        self.queries.append(query)
        return self._results[:limit]


class DuckDuckGoSearchProvider:
    """Keyless search by scraping DuckDuckGo's HTML endpoint.

    The endpoint is meant for browsers without JavaScript, not for programs,
    so the markup can change without notice and heavy use is rate-limited.
    Parsing is defensive: an unexpected layout yields an empty list and a
    warning, never an exception, while HTTP failures are reported as
    :class:`WebSearchError`.
    """

    def __init__(self, client: httpx.AsyncClient) -> None:
        self._client = client

    async def search(self, query: str, limit: int = DEFAULT_MAX_RESULTS) -> list[SearchResult]:
        """Query the HTML endpoint and parse the result list."""
        limit = _clamp_limit(limit)
        try:
            response = await self._client.get(_DUCKDUCKGO_URL, params={"q": query})
            response.raise_for_status()
        except httpx.HTTPError as error:
            raise WebSearchError(f"DuckDuckGo request failed: {error}") from error
        results = parse_duckduckgo_html(response.text)[:limit]
        if not results:
            log.warning("web.search.no_results", provider="duckduckgo", bytes=len(response.content))
        log.info("web.search.done", provider="duckduckgo", results=len(results))
        return results


class BraveSearchProvider:
    """Search through the Brave Search API, which needs a subscription token.

    The token is sent as a header and never logged. Brave returns JSON, so
    this provider is robust where the DuckDuckGo scraper is brittle.
    """

    def __init__(self, client: httpx.AsyncClient, api_key: str) -> None:
        self._client = client
        self._api_key = api_key

    async def search(self, query: str, limit: int = DEFAULT_MAX_RESULTS) -> list[SearchResult]:
        """Query the web search endpoint and map its JSON to results."""
        limit = _clamp_limit(limit)
        headers = {"Accept": "application/json", "X-Subscription-Token": self._api_key}
        try:
            response = await self._client.get(
                _BRAVE_URL, params={"q": query, "count": limit}, headers=headers
            )
            response.raise_for_status()
            payload = response.json()
        except httpx.HTTPError as error:
            raise WebSearchError(f"Brave Search request failed: {error}") from error
        except ValueError as error:
            raise WebSearchError("Brave Search returned a non-JSON body") from error
        results = _parse_brave_payload(payload)[:limit]
        log.info("web.search.done", provider="brave", results=len(results))
        return results


def _parse_brave_payload(payload: object) -> list[SearchResult]:
    if not isinstance(payload, dict):
        raise WebSearchError("Brave Search returned an unexpected payload shape")
    web = payload.get("web")
    entries = web.get("results", []) if isinstance(web, dict) else []
    results = []
    for entry in entries:
        if not isinstance(entry, dict) or not entry.get("url"):
            continue
        results.append(
            SearchResult(
                title=str(entry.get("title", "")),
                url=str(entry["url"]),
                snippet=str(entry.get("description", "")),
            )
        )
    return results


class _DuckDuckGoParser(HTMLParser):
    """Collects ``a.result__a`` (title + link) and ``.result__snippet`` text."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.results: list[SearchResult] = []
        self._title: list[str] = []
        self._snippet: list[str] = []
        self._url: str | None = None
        self._capturing: str | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        """Start capturing when a result link or snippet opens."""
        classes = dict(attrs).get("class") or ""
        if tag == "a" and "result__a" in classes:
            self._flush()
            self._url = _unwrap_redirect(dict(attrs).get("href") or "")
            self._capturing = "title"
        elif "result__snippet" in classes:
            self._capturing = "snippet"

    def handle_endtag(self, tag: str) -> None:
        """Stop capturing at the end of the element being captured."""
        if self._capturing and tag in ("a", "div", "td"):
            self._capturing = None

    def handle_data(self, data: str) -> None:
        """Accumulate text for whichever field is open."""
        if self._capturing == "title":
            self._title.append(data)
        elif self._capturing == "snippet":
            self._snippet.append(data)

    def close(self) -> None:
        """Finish parsing and emit the last pending result."""
        super().close()
        self._flush()

    def _flush(self) -> None:
        if self._url:
            self.results.append(
                SearchResult(
                    title=_squash(self._title), url=self._url, snippet=_squash(self._snippet)
                )
            )
        self._title, self._snippet, self._url = [], [], None


def parse_duckduckgo_html(html: str) -> list[SearchResult]:
    """Parse a DuckDuckGo HTML results page into :class:`SearchResult`s.

    Ads and results without a destination URL are dropped. Exposed so the
    parser can be tested against a saved fixture without any HTTP.
    """
    parser = _DuckDuckGoParser()
    parser.feed(html)
    parser.close()
    return [result for result in parser.results if _is_external(result.url)]


def _is_external(url: str) -> bool:
    """False for ad and internal links, which stay on a duckduckgo.com host."""
    host = urlsplit(url).hostname or ""
    return url.startswith("http") and not host.endswith("duckduckgo.com")


def _unwrap_redirect(href: str) -> str:
    """Turn DuckDuckGo's ``/l/?uddg=<url>`` tracking link into the real URL."""
    if href.startswith("//"):
        href = "https:" + href
    parts = urlsplit(href)
    if parts.path.startswith("/l/") and "uddg" in parse_qs(parts.query):
        return parse_qs(parts.query)["uddg"][0]
    return href


def _squash(pieces: list[str]) -> str:
    return " ".join("".join(pieces).split())


def _clamp_limit(limit: int) -> int:
    return max(1, min(limit, MAX_RESULTS_LIMIT))


async def fetch_article(
    client: httpx.AsyncClient,
    url: str,
    *,
    max_bytes: int = DEFAULT_MAX_PAGE_BYTES,
    max_chars: int = DEFAULT_MAX_ARTICLE_CHARS,
) -> Article:
    """Download ``url`` and return its readable text.

    The body is streamed and abandoned once ``max_bytes`` arrive, so a
    runaway download can neither exhaust memory nor stall the pipeline.

    Raises:
        ArticleFetchError: On HTTP failure, a non-text content type, or a page
            with no readable body.
    """
    try:
        async with client.stream("GET", url, follow_redirects=True) as response:
            response.raise_for_status()
            _require_readable_content_type(url, response)
            html = _decode(await _read_capped(response, max_bytes), response)
            final_url = str(response.url)
    except httpx.HTTPError as error:
        raise ArticleFetchError(url, str(error)) from error
    readable = extract_readable(html)
    if not readable.paragraphs:
        raise ArticleFetchError(url, "no readable text found on the page")
    text, truncated = _cap_text(readable.paragraphs, max_chars)
    log.info("web.article.fetched", url=final_url, chars=len(text), truncated=truncated)
    return Article(
        url=final_url,
        title=readable.title,
        text=text,
        byline=readable.byline,
        published=readable.published,
        truncated=truncated,
    )


def _require_readable_content_type(url: str, response: httpx.Response) -> None:
    content_type = response.headers.get("content-type", "text/html").split(";")[0].strip()
    if content_type.lower() not in _READABLE_CONTENT_TYPES:
        raise ArticleFetchError(url, f"content type {content_type} is not readable text")


async def _read_capped(response: httpx.Response, max_bytes: int) -> bytes:
    chunks = bytearray()
    async for chunk in response.aiter_bytes(_CHUNK_BYTES):
        chunks.extend(chunk)
        if len(chunks) >= max_bytes:
            log.warning("web.article.size_capped", url=str(response.url), max_bytes=max_bytes)
            break
    return bytes(chunks[:max_bytes])


def _decode(body: bytes, response: httpx.Response) -> str:
    """Decode with the header charset, else the ``<meta charset>``, else UTF-8."""
    encoding = response.charset_encoding
    if encoding is None and (match := _META_CHARSET.search(body[:4096])):
        encoding = match.group(1).decode("ascii")
    try:
        return body.decode(encoding or "utf-8", errors="replace")
    except LookupError:
        return body.decode("utf-8", errors="replace")


def _cap_text(paragraphs: tuple[str, ...], max_chars: int) -> tuple[str, bool]:
    """Join paragraphs up to the budget, cutting on a paragraph boundary."""
    kept: list[str] = []
    used = 0
    for paragraph in paragraphs:
        if used + len(paragraph) > max_chars:
            if not kept:
                kept.append(paragraph[: max(0, max_chars - 1)] + _ELLIPSIS)
            return "\n\n".join(kept), True
        kept.append(paragraph)
        used += len(paragraph) + 2
    return "\n\n".join(kept), False


def create_web_client(
    user_agent: str = DEFAULT_USER_AGENT,
    timeout: httpx.Timeout | float = DEFAULT_TIMEOUT_SECONDS,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
) -> httpx.AsyncClient:
    """Build the HTTP client the web helpers expect in production.

    Identifies Jarvis honestly in the User-Agent (sites can block it if they
    wish), bounds every phase of a request with a timeout and follows
    redirects, since search engines hand out redirect links.

    Args:
        user_agent: Sent on every request; ``JARVIS_WEB_USER_AGENT`` in config.
        timeout: Seconds for every phase, or a full ``httpx.Timeout``.
        transport: Tests pass ``httpx.MockTransport``; production leaves it
            ``None`` for the real network.
    """
    return httpx.AsyncClient(
        headers={"User-Agent": user_agent, "Accept-Language": "en"},
        timeout=httpx.Timeout(timeout),
        follow_redirects=True,
        transport=transport,
    )


# Design notes:
# - The client is injected rather than created per call so that one
#   connection pool, one User-Agent and one timeout policy serve every web
#   tool, and tests can swap in httpx.MockTransport.
# - DuckDuckGo scraping is the default because the project must work out of
#   the box with no sign-up; Brave is the sturdier option for anyone willing
#   to create a key. The parser is exported so a layout change is a one-file,
#   fixture-driven fix.
# - fetch_article caps bytes while streaming and characters after extraction:
#   the first protects memory, the second protects the LLM context budget.
# - No robots.txt fetch: Jarvis reads one page the user explicitly asked for,
#   which is the browser use case robots.txt does not govern. The honest
#   User-Agent lets sites that disagree block it.
