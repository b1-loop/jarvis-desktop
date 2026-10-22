"""Per-turn tool selection: offer the model the tools this request needs.

Large hosted models pick the right tool out of fifty; small local ones do
not. Measured against Ollama ``qwen2.5:7b`` with all 57 tool schemas in the
request, "what time is it?" became a ``shell_command`` call running
``date``, while the same request offering three tools produced a correct
``current_time`` call. :class:`RelevanceToolSelector` therefore ranks the
registry against the user's words each turn and hands the model a short
list: a small configurable core, every tool already used in this turn,
and the best-matching rest up to ``Settings.max_tools_per_turn``.

The ranking is hybrid (ADR-0020 and its update). A deterministic word
overlap, weighted towards a tool's name and its declared ``keywords``
(English and Swedish), always runs. When a :class:`SemanticScorer` is
configured and answers in time, cosine similarities from a multilingual
embedding model are blended in, which is what lets a Swedish request find
an English tool description it shares no word with.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import cache
from typing import TYPE_CHECKING, Final, Protocol

if TYPE_CHECKING:
    from jarvis.agent.registry import AnyTool
    from jarvis.config import Settings

ALL_TOOLS: Final = 0
"""A ``max_tools`` of zero means "no selection": every tool is offered."""

STRONG_WEIGHT: Final = 3.0
"""Score per request word equal to a word of a tool's name or keywords."""

PREFIX_WEIGHT: Final = 2.0
"""Score per request word that merely starts with a name or keyword word.

Swedish inflects by suffix and compounds freely ("batteriet",
"skärmdumpen", "ljusstyrkan"), so a keyword is also found at the start of
a longer word. It counts less than an exact match so English "listening"
still prefers ``now_playing`` (exact) over the ``list_*`` tools (prefix).
"""

WEAK_WEIGHT: Final = 1.0
"""Score per request word found only in its description or category."""

DEFAULT_KEYWORD_WEIGHT: Final = 0.4
"""Share of the hybrid score from word overlap; see :class:`HybridWeights`."""

DEFAULT_SEMANTIC_WEIGHT: Final = 0.6
"""Share of the hybrid score from embedding similarity."""

SEMANTIC_MARGIN: Final = 0.08
"""Cosine distance below the best tool at which semantic credit reaches zero.

Embedding similarities are never zero, so without a cut-off every tool
would "match". Only tools within this margin of the request's best match
get semantic credit, scaled linearly from 1 (the best) to 0.
"""

_MIN_STEM_LENGTH: Final = 4
_MIN_PREFIX_LENGTH: Final = 4
"""Shorter keywords must match exactly: "lås" must not claim "låst", "tid" not "tidning"."""

_WORD: Final = re.compile(r"[^\W_]+(?:-[^\W_]+)*")
"""Letters and digits in any script, keeping hyphenated compounds together."""

_STOPWORD_TEXT: Final = """
a an and any are am as at be by can could do does for from get give have how i in
is it its me my of on or please s so some than that the their them then there this
to us want was we what when which who will with would you your
att av de den det din ditt du efter eller en ett från för gör göra har helt här hur
jag just kan lite man med mig min mina mitt mycket nu när någon något några och om
på sig ska som så till under var varför vad vi vilken vilket vilka vill är över
"""
_STOPWORDS: Final = frozenset(_STOPWORD_TEXT.split())
"""Words that carry no intent, English then Swedish; matching on them ranks by length."""


class SemanticScorer(Protocol):
    """Similarity of a request to each tool, from an embedding model."""

    async def similarities(
        self, request: str, tools: Sequence[AnyTool]
    ) -> Mapping[str, float] | None:
        """Cosine similarity per tool name, or ``None`` when unavailable right now.

        Must return within a small latency budget; ``None`` makes the
        selector fall back to word overlap for this request.
        """


class ToolSelector(Protocol):
    """Chooses which tools the model is offered for one request."""

    async def select(
        self, request: str, tools: Sequence[AnyTool], *, keep: Collection[str] = ()
    ) -> list[AnyTool]:
        """The tools to offer, in registry order.

        Args:
            request: What the user said this turn.
            tools: Every registered tool, in registry order.
            keep: Names that must stay available (tools already used this turn).
        """


@dataclass(frozen=True, slots=True)
class HybridWeights:
    """How word overlap and embedding similarity are blended.

    Each signal is first scaled to [0, 1] for the request: overlap relative
    to the best-overlapping tool, similarity by :data:`SEMANTIC_MARGIN`
    below the most similar tool. The weights then say how much each counts.

    Attributes:
        keyword: Weight of the normalised word-overlap score.
        semantic: Weight of the normalised embedding similarity.
    """

    keyword: float = DEFAULT_KEYWORD_WEIGHT
    semantic: float = DEFAULT_SEMANTIC_WEIGHT


@dataclass(frozen=True, slots=True)
class _Profile:
    """The words a tool is known by, split by how much a match on them counts."""

    strong: frozenset[str]
    prefixes: tuple[str, ...]
    weak: frozenset[str]


class RelevanceToolSelector:
    """Ranks tools by word overlap (and similarity, if available) and keeps the best few.

    Args:
        max_tools: Upper bound on the tools offered; ``ALL_TOOLS`` (0)
            disables selection. Core and kept tools are always offered,
            even if they alone exceed the bound.
        core: Tool names offered on every turn regardless of the request.
        semantic: Embedding similarity source; ``None`` ranks by words only.
        weights: How the two signals are blended when both are present.
    """

    def __init__(
        self,
        *,
        max_tools: int = ALL_TOOLS,
        core: Iterable[str] = (),
        semantic: SemanticScorer | None = None,
        weights: HybridWeights | None = None,
    ) -> None:
        self.max_tools = max_tools
        self.core = frozenset(core)
        self.semantic = semantic
        self.weights = weights or HybridWeights()

    @classmethod
    def from_settings(
        cls, settings: Settings, semantic: SemanticScorer | None = None
    ) -> RelevanceToolSelector:
        """The selector for the configured provider's cap, core tools and weights."""
        weights = HybridWeights(
            keyword=settings.tool_selection_keyword_weight,
            semantic=settings.tool_selection_semantic_weight,
        )
        return cls(
            max_tools=settings.tools_per_turn(),
            core=settings.core_tools,
            semantic=semantic,
            weights=weights,
        )

    async def select(
        self, request: str, tools: Sequence[AnyTool], *, keep: Collection[str] = ()
    ) -> list[AnyTool]:
        """Core, kept, then best-ranked tools up to ``max_tools``, in registry order."""
        if self._offers_everything(tools):
            return list(tools)
        similarities = None
        if self.semantic is not None:
            similarities = await self.semantic.similarities(request, tools)
        return self.choose(request, tools, keep=keep, similarities=similarities)

    def choose(
        self,
        request: str,
        tools: Sequence[AnyTool],
        *,
        keep: Collection[str] = (),
        similarities: Mapping[str, float] | None = None,
    ) -> list[AnyTool]:
        """The synchronous core of :meth:`select`, given the similarities (if any)."""
        if self._offers_everything(tools):
            return list(tools)
        pinned = {tool.name for tool in tools if tool.name in self.core or tool.name in keep}
        room = max(self.max_tools - len(pinned), 0)
        ranked = self.rank(request, tools, similarities)
        chosen = pinned | set(_first(ranked, pinned, room))
        return [tool for tool in tools if tool.name in chosen]

    def rank(
        self,
        request: str,
        tools: Sequence[AnyTool],
        similarities: Mapping[str, float] | None = None,
    ) -> list[AnyTool]:
        """Tools that match the request at all, best first; ties keep registry order."""
        words = terms(request)
        overlap = [score(words, _profile(type(tool))) for tool in tools]
        blended = (
            hybrid_scores(overlap, [similarities.get(t.name) for t in tools], self.weights)
            if similarities
            else overlap
        )
        ordered = sorted(
            (index for index, value in enumerate(blended) if value > 0),
            key=lambda index: (-blended[index], index),
        )
        return [tools[index] for index in ordered]

    def _offers_everything(self, tools: Sequence[AnyTool]) -> bool:
        return self.max_tools == ALL_TOOLS or len(tools) <= self.max_tools


def hybrid_scores(
    overlap: Sequence[float], similarity: Sequence[float | None], weights: HybridWeights
) -> list[float]:
    """Blend word overlap and cosine similarity per tool; zero means "not a match".

    Args:
        overlap: Raw :func:`score` per tool.
        similarity: Cosine similarity per tool, ``None`` where unknown.
        weights: The blend.
    """
    best_overlap = max(overlap, default=0.0)
    known = [value for value in similarity if value is not None]
    best_similarity = max(known, default=0.0)
    blended = []
    for words, cosine in zip(overlap, similarity, strict=True):
        keyword_part = words / best_overlap if best_overlap > 0 else 0.0
        semantic_part = 0.0
        if cosine is not None:
            semantic_part = max(0.0, 1.0 - (best_similarity - cosine) / SEMANTIC_MARGIN)
        blended.append(weights.keyword * keyword_part + weights.semantic * semantic_part)
    return blended


def score(words: frozenset[str], profile: _Profile) -> float:
    """Weighted overlap: exact name/keyword hits count most, then prefixes, then descriptions."""
    strong = words & profile.strong
    prefixed = {
        word
        for word in words - strong
        if any(word.startswith(prefix) for prefix in profile.prefixes)
    }
    weak = (words & profile.weak) - strong - prefixed
    return STRONG_WEIGHT * len(strong) + PREFIX_WEIGHT * len(prefixed) + WEAK_WEIGHT * len(weak)


def terms(text: str) -> frozenset[str]:
    """Lower-cased content words of ``text``, plural ``s`` removed, compounds split.

    A hyphenated compound yields its parts and the joined word, so
    "att-göra-listan" matches the keyword "attgöra" as well as "list".
    """
    normalised = unicodedata.normalize("NFC", text.lower())
    words: set[str] = set()
    for token in _WORD.findall(normalised):
        parts = token.split("-")
        words.update(parts)
        if len(parts) > 1:
            words.add("".join(parts))
    return frozenset(_stem(word) for word in words if word not in _STOPWORDS)


def _first(ranked: Sequence[AnyTool], pinned: Collection[str], room: int) -> list[str]:
    """Names of the best ``room`` ranked tools that are not pinned already."""
    return [tool.name for tool in ranked if tool.name not in pinned][:room]


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
    prefixes = tuple(sorted(word for word in strong if len(word) >= _MIN_PREFIX_LENGTH))
    return _Profile(strong=strong, prefixes=prefixes, weak=weak)


# Design notes:
# - Word overlap always runs and is never skipped: it is deterministic,
#   explainable (a test lists what was offered and why) and costs
#   microseconds. The embedding similarity is an addition that may be
#   missing on any turn (model not pulled, Ollama down, over the latency
#   budget); then the ranking is exactly the keyword ranking.
# - Hybrid normalisation: overlap is divided by the request's best overlap
#   and similarity is measured down from the request's best similarity, so
#   both live in [0, 1] whatever the request length or embedding model.
#   Absolute cosine values differ between models; distances from the best
#   match are far more comparable. Weights and margin are in ADR-0020.
# - Tools with a blended score of zero are not offered, even when the cap
#   leaves room: padding the list with unrelated schemas is exactly what
#   confuses small models. The semantic margin is what makes "unrelated"
#   meaningful for embeddings, whose similarities are never zero.
# - Kept tools (used earlier in the turn) are pinned so a multi-round
#   chain never loses a tool between rounds; providers also expect every
#   tool named in the history to stay defined.
# - The selection is offered in registry order, not rank order, so the
#   request is stable across rounds and prompt caching keeps working.
# - Swedish support in the word overlap is deliberately modest: Swedish
#   stopwords, Unicode words, prefix matching for suffixes and compounds,
#   and Swedish ``keywords`` on the commonly used tools. It is the cheap
#   baseline that still works when the embedding model is unavailable.
