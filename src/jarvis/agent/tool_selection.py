"""Per-turn tool selection: offer the model the tools this request needs.

Large hosted models pick the right tool out of fifty; small local ones do
not. Measured against Ollama ``qwen2.5:7b`` with all 57 tool schemas in the
request, "what time is it?" became a ``shell_command`` call running
``date``, while the same request offering three tools produced a correct
``current_time`` call. :class:`RelevanceToolSelector` therefore ranks the
registry against the user's words each turn and hands the model a short
list: a small configurable core, every tool already used in this turn,
and the best-matching rest up to ``Settings.max_tools_per_turn``.

The ranking is deterministic word overlap, weighted towards a tool's name
and its declared ``keywords``; see ADR-0020 for the measurements that chose
it over the memory module's hashing embedding.
"""

from __future__ import annotations

import re
from collections.abc import Collection, Iterable, Sequence
from dataclasses import dataclass
from functools import cache
from typing import TYPE_CHECKING, Final, Protocol

if TYPE_CHECKING:
    from jarvis.agent.registry import AnyTool
    from jarvis.config import Settings

ALL_TOOLS: Final = 0
"""A ``max_tools`` of zero means "no selection": every tool is offered."""

STRONG_WEIGHT: Final = 3.0
"""Score per request word found in a tool's name or keywords."""

WEAK_WEIGHT: Final = 1.0
"""Score per request word found only in its description or category."""

_MIN_STEM_LENGTH: Final = 4
_WORD: Final = re.compile(r"[a-z0-9]+")
_STOPWORD_TEXT: Final = """
a an and any are am as at be by can could do does for from get give have how i in
is it its me my of on or please s so some than that the their them then there this
to us want was we what when which who will with would you your
"""
_STOPWORDS: Final = frozenset(_STOPWORD_TEXT.split())
"""Words that carry no intent; matching on them would rank by description length."""


class ToolSelector(Protocol):
    """Chooses which tools the model is offered for one request."""

    def select(
        self, request: str, tools: Sequence[AnyTool], *, keep: Collection[str] = ()
    ) -> list[AnyTool]:
        """The tools to offer, in registry order.

        Args:
            request: What the user said this turn.
            tools: Every registered tool, in registry order.
            keep: Names that must stay available (tools already used this turn).
        """


@dataclass(frozen=True, slots=True)
class _Profile:
    """The words a tool is known by, split by how much a match on them counts."""

    strong: frozenset[str]
    weak: frozenset[str]


class RelevanceToolSelector:
    """Ranks tools by word overlap with the request and keeps the best few.

    Args:
        max_tools: Upper bound on the tools offered; ``ALL_TOOLS`` (0)
            disables selection. Core and kept tools are always offered,
            even if they alone exceed the bound.
        core: Tool names offered on every turn regardless of the request.
    """

    def __init__(self, *, max_tools: int = ALL_TOOLS, core: Iterable[str] = ()) -> None:
        self.max_tools = max_tools
        self.core = frozenset(core)

    @classmethod
    def from_settings(cls, settings: Settings) -> RelevanceToolSelector:
        """The selector for the configured provider's cap and core tools."""
        return cls(max_tools=settings.tools_per_turn(), core=settings.core_tools)

    def select(
        self, request: str, tools: Sequence[AnyTool], *, keep: Collection[str] = ()
    ) -> list[AnyTool]:
        """Core, kept, then best-ranked tools up to ``max_tools``, in registry order."""
        if self.max_tools == ALL_TOOLS or len(tools) <= self.max_tools:
            return list(tools)
        pinned = {tool.name for tool in tools if tool.name in self.core or tool.name in keep}
        room = max(self.max_tools - len(pinned), 0)
        ranked = [tool.name for tool in self.rank(request, tools) if tool.name not in pinned]
        chosen = pinned | set(ranked[:room])
        return [tool for tool in tools if tool.name in chosen]

    def rank(self, request: str, tools: Sequence[AnyTool]) -> list[AnyTool]:
        """Tools that match the request at all, best first; ties keep registry order."""
        words = terms(request)
        scored = [(score(words, _profile(type(tool))), index) for index, tool in enumerate(tools)]
        ordered = sorted((pair for pair in scored if pair[0] > 0), key=lambda p: (-p[0], p[1]))
        return [tools[index] for _, index in ordered]


def score(words: frozenset[str], profile: _Profile) -> float:
    """Weighted overlap: name and keyword hits count more than description hits."""
    strong = words & profile.strong
    weak = (words & profile.weak) - strong
    return STRONG_WEIGHT * len(strong) + WEAK_WEIGHT * len(weak)


def terms(text: str) -> frozenset[str]:
    """Lower-cased content words of ``text`` with a plural ``s`` removed."""
    return frozenset(_stem(word) for word in _WORD.findall(text.lower()) if word not in _STOPWORDS)


def _stem(word: str) -> str:
    """``emails`` -> ``email``; deliberately crude, applied to both sides alike."""
    if len(word) >= _MIN_STEM_LENGTH and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


@cache
def _profile(tool_class: type[AnyTool]) -> _Profile:
    """A tool's matching words; cached per class because definitions are static."""
    strong = terms(" ".join((tool_class.name.replace("_", " "), *tool_class.keywords)))
    weak = terms(f"{tool_class.description} {tool_class.category.value.replace('_', ' ')}")
    return _Profile(strong=strong, weak=weak)


# Design notes:
# - Deterministic word overlap rather than an LLM pre-pass: asking a 7B
#   model to pick tools first doubles latency and reintroduces the very
#   failure it is meant to fix. Overlap is explainable (a test lists what
#   was offered and why) and costs microseconds.
# - Tools that match no word are not offered at all, even when the cap
#   leaves room: padding the list with unrelated schemas is exactly what
#   confuses small models. The core set covers requests with no matching
#   words, and the measured hit rate (ADR-0020) bounds what this misses.
# - Kept tools (used earlier in the turn) are pinned so a multi-round
#   chain never loses a tool between rounds; providers also expect every
#   tool named in the history to stay defined.
# - The selection is offered in registry order, not rank order, so the
#   request is stable across rounds and prompt caching keeps working.
# - Assumption: requests are English, like the tool descriptions. A
#   request in another language matches nothing and gets the core set;
#   raising the cap or setting it to 0 restores the full list.
