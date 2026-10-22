"""Heuristic "readable text" extraction from HTML, on the standard library only.

Given a web page, find the element that holds the article body, drop
navigation, scripts and other boilerplate, and return the remaining text as
paragraphs together with the title, byline and publication date. The
approach is a slimmed-down Readability: paragraph-like elements vote for
their ancestors with their text length, discounted by link density, and the
ancestor with the most votes wins.

The parser is ``html.parser`` from the standard library, so this module adds
no dependency and no compiled wheel. See ADR-0012 for why that trade-off was
chosen over ``trafilatura`` or ``readability-lxml``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from html.parser import HTMLParser
from typing import Final

_VOID_TAGS: Final = frozenset(
    {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "source", "wbr"}
)
_DROPPED_TAGS: Final = frozenset(
    {
        "script",
        "style",
        "noscript",
        "template",
        "svg",
        "iframe",
        "form",
        "button",
        "nav",
        "footer",
        "aside",
        "menu",
    }
)
_BLOCK_TAGS: Final = frozenset(
    {
        "address",
        "article",
        "blockquote",
        "body",
        "dd",
        "div",
        "dl",
        "dt",
        "figure",
        "figcaption",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "header",
        "hr",
        "li",
        "main",
        "ol",
        "p",
        "pre",
        "section",
        "table",
        "td",
        "th",
        "tr",
        "ul",
    }
)
_SELF_CLOSING_SIBLINGS: Final = frozenset({"li", "dt", "dd", "tr", "td", "th"})
_PARAGRAPH_TAGS: Final = frozenset({"p", "pre", "blockquote", "li", "dd", "td", "figcaption"})
_HEADING_TAGS: Final = frozenset({"h1", "h2", "h3", "h4", "h5", "h6"})
_CANDIDATE_TAGS: Final = frozenset({"article", "main", "section", "div", "body", "td"})
_BOILERPLATE_MARKER: Final = re.compile(
    r"(^|[\s_-])(nav|menu|footer|sidebar|comment|share|social|related|promo|cookie|"
    r"advert|banner|popup|subscribe|newsletter|breadcrumb)",
    re.IGNORECASE,
)
_BYLINE_MARKER: Final = re.compile(r"byline|author", re.IGNORECASE)
_WHITESPACE: Final = re.compile(r"\s+")
_MIN_VOTING_PARAGRAPH_CHARS: Final = 25
_MIN_OUTPUT_PARAGRAPH_CHARS: Final = 20
_MAX_LINK_DENSITY: Final = 0.5
_MAX_BYLINE_CHARS: Final = 120
_GRANDPARENT_VOTE_SHARE: Final = 0.5


@dataclass(slots=True)
class _Node:
    """One element of the lightweight DOM built by :class:`_TreeBuilder`."""

    tag: str
    attrs: dict[str, str]
    parent: _Node | None
    children: list[_Node | str] = field(default_factory=list)

    @property
    def marker(self) -> str:
        """``class`` and ``id`` joined, for boilerplate heuristics."""
        return f"{self.attrs.get('class', '')} {self.attrs.get('id', '')}"

    def text(self, *, stop: frozenset[str] = frozenset()) -> str:
        """Descendant text with whitespace collapsed, not entering ``stop`` tags.

        With ``stop=_BLOCK_TAGS`` this is the element's *own* text: what a
        ``<li>`` says before its nested list, what a ``<div>`` says outside
        its paragraphs.
        """
        return _collapse(" ".join(self._text_pieces(stop)))

    def _text_pieces(self, stop: frozenset[str]) -> list[str]:
        pieces: list[str] = []
        for child in self.children:
            if isinstance(child, str):
                pieces.append(child)
            elif child.tag not in stop:
                pieces.extend(child._text_pieces(stop))
        return pieces

    def elements(self, *, stop: frozenset[str] = frozenset()) -> list[_Node]:
        """This node and every descendant element, in document order.

        Descendants with a tag in ``stop`` are listed but not entered.
        """
        found = [self]
        for child in self.children:
            if isinstance(child, _Node):
                found.extend(child.elements(stop=stop) if child.tag not in stop else [child])
        return found

    def link_density(self, *, stop: frozenset[str] = frozenset()) -> float:
        """Share of the text that sits inside anchors; menus score near 1."""
        total = len(self.text(stop=stop))
        if total == 0:
            return 0.0
        anchors = (node for node in self.elements(stop=stop)[1:] if node.tag == "a")
        linked = sum(len(node.text(stop=stop)) for node in anchors)
        return min(1.0, linked / total)


class _TreeBuilder(HTMLParser):
    """Builds a ``_Node`` tree, tolerating the unclosed tags real pages have."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.root = _Node("document", {}, None)
        self._open = [self.root]
        self.meta: dict[str, str] = {}

    @property
    def _current(self) -> _Node:
        return self._open[-1]

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        """Open an element, closing an unterminated ``<p>`` or ``<li>`` first."""
        attributes = {key: value or "" for key, value in attrs}
        if tag == "meta":
            self._remember_meta(attributes)
            return
        if (tag in _BLOCK_TAGS and self._current.tag == "p") or (
            tag == self._current.tag and tag in _SELF_CLOSING_SIBLINGS
        ):
            self._open.pop()
        node = _Node(tag, attributes, self._current)
        self._current.children.append(node)
        if tag not in _VOID_TAGS:
            self._open.append(node)

    def handle_endtag(self, tag: str) -> None:
        """Close the nearest open element with this tag; ignore strays."""
        for index in range(len(self._open) - 1, 0, -1):
            if self._open[index].tag == tag:
                del self._open[index:]
                return

    def handle_data(self, data: str) -> None:
        """Attach text to the current element; scripts and styles are kept out."""
        if self._current.tag not in _DROPPED_TAGS and data.strip():
            self._current.children.append(data)

    def _remember_meta(self, attributes: dict[str, str]) -> None:
        key = attributes.get("property") or attributes.get("name")
        content = attributes.get("content")
        if key and content and key.lower() not in self.meta:
            self.meta[key.lower()] = _collapse(content)


@dataclass(frozen=True, slots=True)
class ReadableHtml:
    """The parts of a page a reader (human or LLM) actually wants.

    Attributes:
        title: Best available title; empty if the page has none.
        byline: Author line if one was found, else ``None``.
        published: Publication timestamp as the page states it, else ``None``.
        paragraphs: Body text, one entry per paragraph or heading, in order.
    """

    title: str
    byline: str | None
    published: str | None
    paragraphs: tuple[str, ...]

    @property
    def text(self) -> str:
        """Paragraphs joined with blank lines, ready to read or summarise."""
        return "\n\n".join(self.paragraphs)


def extract_readable(html: str) -> ReadableHtml:
    """Extract title, byline, date and body paragraphs from ``html``.

    Never raises on malformed markup: the worst case is an empty body, which
    the caller reports as "nothing readable on that page".
    """
    builder = _TreeBuilder()
    builder.feed(html)
    builder.close()
    root = builder.root
    title = _find_title(root, builder.meta)
    byline = _find_byline(root, builder.meta)
    published = _find_published(root, builder.meta)
    _prune_boilerplate(root)
    body = _best_candidate(root)
    paragraphs = _collect_paragraphs(body) if body is not None else ()
    return ReadableHtml(title=title, byline=byline, published=published, paragraphs=paragraphs)


def _find_title(root: _Node, meta: dict[str, str]) -> str:
    # og:title is the headline the site chose for sharing; <title> usually
    # carries a " - Site Name" suffix, so it is the last resort.
    if og_title := meta.get("og:title"):
        return og_title
    for node in root.elements():
        if node.tag == "h1" and node.text():
            return node.text()
    for node in root.elements():
        if node.tag == "title":
            return node.text()
    return ""


def _find_byline(root: _Node, meta: dict[str, str]) -> str | None:
    if author := meta.get("author") or meta.get("article:author"):
        return author
    for node in root.elements():
        if node.attrs.get("rel") == "author" or _BYLINE_MARKER.search(node.marker):
            text = node.text()
            if 0 < len(text) <= _MAX_BYLINE_CHARS:
                return text
    return None


def _find_published(root: _Node, meta: dict[str, str]) -> str | None:
    for key in ("article:published_time", "datepublished", "date", "pubdate", "og:updated_time"):
        if value := meta.get(key):
            return value
    for node in root.elements():
        if node.tag == "time" and node.attrs.get("datetime"):
            return node.attrs["datetime"]
    return None


def _prune_boilerplate(root: _Node) -> None:
    """Drop whole subtrees that never hold article text."""
    for node in root.elements():
        node.children = [
            child for child in node.children if isinstance(child, str) or not _is_boilerplate(child)
        ]


def _is_boilerplate(node: _Node) -> bool:
    if node.tag in _DROPPED_TAGS:
        return True
    # <header> at the top of an <article> holds the headline, which is
    # captured separately; everywhere else it is site chrome.
    if node.tag == "header":
        return True
    return node.tag not in _HEADING_TAGS and bool(_BOILERPLATE_MARKER.search(node.marker))


def _best_candidate(root: _Node) -> _Node | None:
    """The ancestor that paragraph votes point at, or the body as fallback."""
    scores: dict[int, float] = {}
    nodes_by_id: dict[int, _Node] = {}
    for node in root.elements():
        if node.tag not in _PARAGRAPH_TAGS:
            continue
        text = node.text()
        if len(text) < _MIN_VOTING_PARAGRAPH_CHARS or node.link_density() > _MAX_LINK_DENSITY:
            continue
        _vote(node, len(text), scores, nodes_by_id)
    if not scores:
        return next((node for node in root.elements() if node.tag == "body"), root)
    winner = max(scores, key=lambda key: scores[key])
    return nodes_by_id[winner]


def _vote(
    node: _Node, weight: int, scores: dict[int, float], nodes_by_id: dict[int, _Node]
) -> None:
    ancestors = (node.parent, node.parent.parent if node.parent else None)
    shares = (1.0, _GRANDPARENT_VOTE_SHARE)
    for ancestor, share in zip(ancestors, shares, strict=True):
        if ancestor is None or ancestor.tag not in _CANDIDATE_TAGS:
            continue
        scores[id(ancestor)] = scores.get(id(ancestor), 0.0) + weight * share
        nodes_by_id[id(ancestor)] = ancestor


def _collect_paragraphs(body: _Node) -> tuple[str, ...]:
    """One entry per block element's own text, so nested blocks never repeat."""
    paragraphs: list[str] = []
    for node in body.elements():
        if node.tag not in _BLOCK_TAGS and node is not body:
            continue
        text = node.text(stop=_BLOCK_TAGS)
        if node.tag in _HEADING_TAGS:
            if text:
                paragraphs.append(text)
        elif (
            len(text) >= _MIN_OUTPUT_PARAGRAPH_CHARS
            and node.link_density(stop=_BLOCK_TAGS) <= _MAX_LINK_DENSITY
        ):
            paragraphs.append(text)
    return tuple(paragraphs)


def _collapse(text: str) -> str:
    return _WHITESPACE.sub(" ", text).strip()


# Design notes:
# - html.parser over lxml/selectolax: the stdlib parser is slower, but an
#   article is at most a few hundred kilobytes and extraction runs once per
#   spoken request, so speed does not matter; zero dependencies does.
# - Votes go to the parent (full weight) and grandparent (half) only, like the
#   original Readability algorithm. Deeper propagation makes <body> win on
#   pages with several short content blocks.
# - Link density is the single most effective boilerplate signal: menus,
#   "related articles" and footers are almost entirely anchors.
# - The extractor never raises on bad markup. A page with no recognisable
#   body yields zero paragraphs and the caller decides how to report that.
