"""Tool: search the web and return ranked links with snippets.

"Look that up" for anything the model does not know or that changes over
time. The backend is whatever the composition root wired (keyless
DuckDuckGo by default, Brave with a key; ADR-0012). The snippets are often
enough to answer; ``read_article`` fetches a result when they are not.
"""

from __future__ import annotations

from typing import Final
from urllib.parse import urlsplit

from pydantic import Field, JsonValue

from jarvis.agent.tool import Tool, ToolCategory, ToolContext, ToolInput, ToolResult
from jarvis.integrations.web import MAX_RESULTS_LIMIT, SearchResult
from jarvis.tools._integration_support import (
    Service,
    count_phrase,
    not_configured,
    trim,
    web_search_provider,
)

MAX_QUERY_CHARS: Final = 300
MAX_LIMIT: Final = min(10, MAX_RESULTS_LIMIT)
_SPOKEN_TITLE_CHARS: Final = 80


class WebSearchInput(ToolInput):
    """Arguments for :class:`WebSearchTool`."""

    query: str = Field(
        min_length=1,
        max_length=MAX_QUERY_CHARS,
        description="What to search for, phrased as a search-engine query.",
    )
    limit: int = Field(default=5, ge=1, le=MAX_LIMIT, description="Number of results.")


class WebSearchTool(Tool[WebSearchInput]):
    """Runs a web search through the configured backend."""

    name = "web_search"
    description = (
        "Search the web. Use for current events, facts you are unsure of, prices, opening "
        "hours, documentation, or when the user says 'look it up'. Returns titles, URLs and "
        "snippets, best first; call read_article on a URL when the snippets are not enough."
    )
    Input = WebSearchInput
    category = ToolCategory.WEB
    keywords = (
        "weather",
        "news",
        "google",
        "internet",
        "online",
        "sök",
        "söka",
        "googla",
        "väder",
        "vädret",
        "nyheter",
    )

    async def run(self, params: WebSearchInput, context: ToolContext) -> ToolResult:
        """Query the backend and format the hits."""
        provider = web_search_provider(context)
        if provider is None:
            return not_configured(Service.WEB_SEARCH)
        results = await provider.search(params.query, params.limit)
        items: list[JsonValue] = [_item(result) for result in results]
        lines = [f"Search results for {params.query!r}:"]
        lines.extend(
            f"{rank}. {r.title}\n   {r.url}\n   {r.snippet}" for rank, r in enumerate(results, 1)
        )
        return ToolResult.success(
            _spoken(params.query, results),
            detail="\n".join(lines),
            data={"query": params.query, "results": items},
        )


def _item(result: SearchResult) -> dict[str, JsonValue]:
    return {"title": result.title, "url": result.url, "snippet": result.snippet}


def _spoken(query: str, results: list[SearchResult]) -> str:
    if not results:
        return f"I found nothing on the web for {query}."
    top = results[0]
    site = (urlsplit(top.url).hostname or "").removeprefix("www.")
    title = trim(top.title, _SPOKEN_TITLE_CHARS)
    return f"I found {count_phrase(len(results), 'result')}; the top one is {title} on {site}."


# Design notes:
# - The spoken line names only the top hit: the LLM normally answers from
#   the snippets in ``detail``, and reading five titles aloud is noise.
# - WebSearchError propagates to Tool.execute, which turns it into a failed
#   result with the backend's reason; there is nothing better to do here.
