# 0012. Web search, article reading and local document access

Date: 2026-10-21
Status: Accepted

## Context

Phase 2 asks Jarvis to "look that up", "read me that article", "summarise
that YouTube video" and "find / summarise / open the last thing I worked on".
Three decisions sit underneath those features:

1. **Which search engine.** Every mainstream search API needs a key and most
   charge per query; the project must work on a fresh clone with only an
   Anthropic key.
2. **How to turn a web page into text.** The LLM context is the scarce
   resource; a raw page is 90 % navigation, scripts and footers.
3. **Which files Jarvis may read.** A voice assistant that silently walks the
   whole disk is both slow and unsettling.

Constraints from `CLAUDE.md`: `mypy --strict`, no secrets in logs, no
network in tests, everything async, stubs named `Stub*`.

## Decision

**Search: keyless DuckDuckGo HTML scraping by default, Brave Search API
behind an optional key.** `DuckDuckGoSearchProvider` parses the
`html.duckduckgo.com/html/` page with `html.parser`; `BraveSearchProvider`
calls the JSON API when `BRAVE_SEARCH_API_KEY` is set. Both implement the
`WebSearchProvider` Protocol next to `StubWebSearchProvider`, and the parser
is a pure function tested against a saved fixture so a layout change is a
one-file fix. The scraper's brittleness is accepted and documented in its
docstring rather than hidden.

**Extraction: an in-house Readability-style heuristic over the standard
library** (`integrations/html_text.py`). Paragraph-like elements vote for
their parent and grandparent with their text length, discounted by link
density; the winner's block-level own-text becomes the article. Title,
byline and date come from Open Graph / meta tags with DOM fallbacks.
`fetch_article` streams the body with a byte cap, decodes with the header or
`<meta>` charset, and caps characters on a paragraph boundary.

**Documents: explicit roots, name-first search, bounded content search.**
`Settings.document_roots` defaults to `~/Documents`, `~/Desktop` and
`~/Downloads` and is overridable with an `os.pathsep`-separated list.
`find_documents` walks only those roots (skipping hidden, tooling and
`node_modules` folders), ranks by modification time and opens at most the
40 most recent candidates when content matching is requested.
`extract_text` handles PDF (`pypdf`), Word (`python-docx`), Markdown, text
and CSV behind size caps, and `summarise_for_llm` water-fills a character
budget across sections so headings and short sections survive trimming.

YouTube transcripts use `youtube-transcript-api` behind a `TranscriptFetcher`
Protocol; the library is synchronous and runs in a worker thread.

## Alternatives considered

- **Google Custom Search / Bing / SerpAPI as the default** – all need a key
  and a billing account; a reviewer cloning the repository could not try the
  feature. Kept as the shape of the paid fallback (Brave was chosen for its
  simple header-token API and generous free tier).
- **DuckDuckGo's `lite` endpoint** – even simpler table markup, but it is
  rate-limited more aggressively in practice and carries less snippet text.
- **`trafilatura`** – the best extraction quality available in Python, but
  it pulls `lxml`, `htmldate`, `courlan` and `justext` (compiled and partly
  untyped packages, so `lxml-stubs` and overrides follow). For a page the user
  explicitly asked to hear, a heuristic that gets the body right on the
  common article layout is enough; the extractor is a single module that can
  be swapped for `trafilatura` behind the same `ReadableHtml` type later.
- **`readability-lxml`** – smaller than trafilatura but also untyped,
  unmaintained for stretches, and still needs `lxml`.
- **Indexing the whole disk (Windows Search, Everything, a Chroma index)** –
  far more capable, but a background indexer is a separate feature with its
  own privacy story. Explicit roots and a bounded scan answer "that file I
  edited yesterday", which is what people ask a voice assistant for.
- **Reading file content in the event loop** – rejected outright; `pypdf` on
  a 300-page PDF blocks for seconds, which would freeze the audio poller.

## Consequences

- Search works with zero configuration, at the cost of depending on an
  unofficial endpoint. A breakage surfaces as an empty result list with a
  warning log, and the fix is a fixture update plus a parser tweak.
- Extraction quality is good on article-shaped pages and weaker on
  single-page apps that render client-side; those return "no readable text"
  rather than garbage.
- No new third-party dependencies for this step; `pypdf`, `python-docx` and
  `youtube-transcript-api` were already in the `documents` extra and are
  imported lazily so the package imports without it.
- Document search never leaves the configured roots; adding a root is a
  config change, not a code change.
- The tools exposing these helpers to the LLM (step 8) must treat `fetch_article`
  and `extract_text` as read-only and keep `summarise_for_llm` budgets well
  below the model's context window.
