"""Tool: fetch a web page and return its readable text.

Backs "read me that article" and "what does this page say?". The page is
reduced to its article body by ``fetch_article`` (ADR-0012), capped to a
character budget, and returned for the LLM to read aloud or summarise as
the user asked. The tool itself never decides between the two.

The URL may come from search results or from text the model read, so it is
treated as untrusted: only http(s), and never a loopback, private or
link-local address, which keeps a prompt-injected page from making Jarvis
read the user's router or local services.
"""

from __future__ import annotations

import ipaddress
from typing import Final
from urllib.parse import urlsplit

from pydantic import Field

from jarvis.agent.tool import Tool, ToolCategory, ToolContext, ToolInput, ToolResult
from jarvis.integrations.web import (
    DEFAULT_MAX_ARTICLE_CHARS,
    Article,
    ArticleFetchError,
    fetch_article,
)
from jarvis.tools._integration_support import Service, not_configured, trim, web_client

MAX_URL_CHARS: Final = 2048
MIN_CHARS: Final = 1000
_LOCAL_SUFFIXES: Final = (".local", ".localhost", ".internal", ".lan", ".home.arpa")
_SPOKEN_TITLE_CHARS: Final = 100


class ReadArticleInput(ToolInput):
    """Arguments for :class:`ReadArticleTool`."""

    url: str = Field(
        min_length=8,
        max_length=MAX_URL_CHARS,
        pattern=r"^https?://\S+$",
        description="Full http(s) URL of the page, e.g. from web_search results.",
    )
    max_chars: int = Field(
        default=8000,
        ge=MIN_CHARS,
        le=DEFAULT_MAX_ARTICLE_CHARS,
        description="Most characters of article text to return; lower for a quick summary.",
    )


class ReadArticleTool(Tool[ReadArticleInput]):
    """Downloads a page and extracts the article text."""

    name = "read_article"
    description = (
        "Fetch a web page and return its main article text (title, byline, body), without "
        "menus or ads. Use to read an article aloud or to summarise or answer questions about "
        "a specific URL. Returns the text; you decide whether to read it or summarise it."
    )
    Input = ReadArticleInput
    category = ToolCategory.WEB

    async def run(self, params: ReadArticleInput, context: ToolContext) -> ToolResult:
        """Refuse local addresses, then fetch and package the article."""
        client = web_client(context)
        if client is None:
            return not_configured(Service.WEB_PAGES)
        if reason := local_address_reason(params.url):
            return ToolResult.failure(
                "I won't open addresses on your own network.",
                detail=f"refused {params.url}: {reason}",
            )
        try:
            article = await fetch_article(client, params.url, max_chars=params.max_chars)
        except ArticleFetchError as error:
            return ToolResult.failure("I couldn't read that page.", detail=str(error))
        return _result(article)


def local_address_reason(url: str) -> str | None:
    """Why ``url`` points at this machine or the local network, or ``None``.

    Only literal hosts are checked; a public name that resolves to a private
    address is not caught (that needs resolver-level pinning, see notes).
    """
    host = (urlsplit(url).hostname or "").lower().rstrip(".")
    if not host or host == "localhost" or host.endswith(_LOCAL_SUFFIXES):
        return f"host {host!r} is local"
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return None
    if address.is_private or address.is_loopback or address.is_link_local or address.is_reserved:
        return f"address {address} is not public"
    return None


def _result(article: Article) -> ToolResult:
    words = len(article.text.split())
    byline = f" by {article.byline}" if article.byline else ""
    title = trim(article.title, _SPOKEN_TITLE_CHARS) or "the page"
    header = f"Title: {article.title}\nURL: {article.url}\nBy: {article.byline or 'unknown'}"
    note = "\n(Text was cut to the character budget.)" if article.truncated else ""
    return ToolResult.success(
        f"I've got {title}{byline}, about {words} words.",
        detail=f"{header}\nPublished: {article.published or 'unknown'}{note}\n\n{article.text}",
        data={
            "url": article.url,
            "title": article.title,
            "byline": article.byline,
            "published": article.published,
            "words": words,
            "truncated": article.truncated,
            "text": article.text,
        },
    )


# Design notes:
# - The local-address guard checks the literal host only. Resolving the name
#   first and pinning the connection to the checked address would close the
#   DNS-rebinding gap (and a public URL redirecting to a private one), but
#   needs a custom httpx transport; the literal check already stops the
#   realistic injected "read http://192.168.1.1/" case.
# - max_chars defaults well below fetch_article's own cap: a spoken answer
#   rarely needs more than a few thousand characters, and the context is
#   shared with the rest of the turn.
