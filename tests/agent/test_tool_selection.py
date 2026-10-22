from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import pytest
import yaml

from jarvis.agent.registry import AnyTool, ToolRegistry
from jarvis.agent.tool import Tool, ToolContext, ToolInput, ToolResult
from jarvis.agent.tool_selection import (
    ALL_TOOLS,
    HybridWeights,
    RelevanceToolSelector,
    hybrid_scores,
    terms,
)
from jarvis.config import RuntimeMode, Settings
from tests.agent.fakes import DestructiveTool, EchoTool, SendTool

CASES_FILE: Final = Path(__file__).parent / "data" / "tool_selection_cases.yaml"

MIN_TOP_1_HIT_RATE: Final = 0.85
"""English keyword ranking: measured 0.89 (0.80 without keywords); ADR-0020."""

MIN_TOP_3_HIT_RATE: Final = 0.95
"""English keyword ranking: measured 1.00 (0.91 without keywords)."""

MIN_SELECTED_HIT_RATE: Final = 1.0
"""What the model is actually offered under the default Ollama settings."""

MAX_HELD_OUT_KEYWORD_RATE: Final = 0.8
"""Keywords alone found 0.57 of the held-out Swedish cases; far above that means
keywords were tuned on them, which defeats the point of holding them out."""


class _ClockInput(ToolInput):
    pass


class ClockTool(Tool[_ClockInput]):
    name = "clock_reading"
    description = "Read the hands."
    Input = _ClockInput
    keywords = ("time", "hour", "klockan", "batteri")

    async def run(self, params: _ClockInput, context: ToolContext) -> ToolResult:
        return ToolResult.success("Noon.")


class FixedSimilarities:
    """A ``SemanticScorer`` with canned answers; records what it was asked."""

    def __init__(self, values: Mapping[str, float] | None) -> None:
        self.values = values
        self.requests: list[str] = []

    async def similarities(
        self, request: str, tools: Sequence[AnyTool]
    ) -> Mapping[str, float] | None:
        self.requests.append(request)
        return self.values


def names(tools: list[AnyTool]) -> list[str]:
    return [tool.name for tool in tools]


@pytest.fixture
def fakes() -> list[AnyTool]:
    return [ClockTool(), DestructiveTool(), EchoTool(), SendTool()]


# -- word overlap ---------------------------------------------------------------


def test_terms_drop_stopwords_and_plural_s() -> None:
    assert terms("What's in my emails, please?") == {"email"}
    assert terms("address the glass") == {"address", "glass"}


def test_terms_keep_swedish_letters_and_split_compounds() -> None:
    assert terms("Lägg till mjölk på att-göra-listan") == {
        "lägg",
        "mjölk",
        "listan",
        "attgöralistan",
    }
    assert terms("hur mycket batteri har jag?") == {"batteri"}


def test_zero_cap_offers_every_tool(fakes: list[AnyTool]) -> None:
    selector = RelevanceToolSelector(max_tools=ALL_TOOLS)

    assert selector.choose("anything", fakes) == fakes


def test_keyword_match_outranks_a_description_match(fakes: list[AnyTool]) -> None:
    selector = RelevanceToolSelector(max_tools=1)

    assert names(selector.choose("what time is it", fakes)) == ["clock_reading"]


def test_a_keyword_at_the_start_of_an_inflected_word_matches(fakes: list[AnyTool]) -> None:
    selector = RelevanceToolSelector(max_tools=1)

    assert names(selector.choose("hur mycket är kvar på batteriet?", fakes)) == ["clock_reading"]


def test_an_exact_keyword_beats_a_prefix_match() -> None:
    class Listening(ClockTool):
        name = "now_listening"
        keywords = ("listening",)

    class Lister(ClockTool):
        name = "list_things"
        keywords = ()

    selector = RelevanceToolSelector()

    ranked = selector.rank("what am I listening to", [Lister(), Listening()])

    assert names(ranked) == ["now_listening", "list_things"]


def test_unrelated_tools_are_not_offered_even_with_room_left(fakes: list[AnyTool]) -> None:
    selector = RelevanceToolSelector(max_tools=3)

    assert names(selector.choose("repeat this text", fakes)) == ["echo"]


def test_core_and_kept_tools_are_always_offered_in_registry_order(
    fakes: list[AnyTool],
) -> None:
    selector = RelevanceToolSelector(max_tools=3, core=["send"])

    chosen = selector.choose("repeat this text", fakes, keep={"destroy"})

    assert names(chosen) == ["destroy", "echo", "send"]


def test_pinned_tools_may_exceed_the_cap(fakes: list[AnyTool]) -> None:
    selector = RelevanceToolSelector(max_tools=2, core=["send", "clock_reading"])

    chosen = selector.choose("repeat this text", fakes, keep={"destroy"})

    assert names(chosen) == ["clock_reading", "destroy", "send"]


def test_unknown_core_names_are_ignored(fakes: list[AnyTool]) -> None:
    selector = RelevanceToolSelector(max_tools=2, core=["no_such_tool"])

    assert names(selector.choose("repeat this text", fakes)) == ["echo"]


@pytest.mark.parametrize(
    ("overrides", "cap"),
    [
        ({}, 12),
        ({"llm_provider": "anthropic"}, ALL_TOOLS),
        ({"llm_provider": "local"}, 12),
        ({"llm_provider": "anthropic", "mode": RuntimeMode.LOCAL}, 12),
        ({"llm_provider": "anthropic", "max_tools_per_turn": 20}, 20),
    ],
)
def test_from_settings_uses_the_provider_cap(
    make_settings: Callable[..., Settings], overrides: dict[str, Any], cap: int
) -> None:
    selector = RelevanceToolSelector.from_settings(make_settings(**overrides))

    assert selector.max_tools == cap
    assert selector.core == {"current_time", "web_search"}


def test_from_settings_takes_the_hybrid_weights(make_settings: Callable[..., Settings]) -> None:
    settings = make_settings(tool_selection_keyword_weight=0.2, tool_selection_semantic_weight=0.8)

    selector = RelevanceToolSelector.from_settings(settings)

    assert selector.weights == HybridWeights(keyword=0.2, semantic=0.8)
    assert selector.semantic is None


# -- hybrid ranking ---------------------------------------------------------------


def test_hybrid_scores_scale_both_signals_and_cut_off_far_similarities() -> None:
    weights = HybridWeights(keyword=0.4, semantic=0.6)

    blended = hybrid_scores([6.0, 3.0, 0.0, 0.0], [0.50, 0.70, 0.66, 0.30], weights)

    assert blended[0] == pytest.approx(0.4)  # best words, similarity far below the best
    assert blended[1] == pytest.approx(0.2 + 0.6)  # half the words, the best similarity
    assert blended[2] == pytest.approx(0.6 * 0.5)  # no words, half the margin below the best
    assert blended[3] == 0.0  # neither: not a match at all


def test_unknown_similarities_count_as_no_semantic_match() -> None:
    blended = hybrid_scores([3.0, 0.0], [None, None], HybridWeights())

    assert blended == [pytest.approx(HybridWeights().keyword), 0.0]


async def test_similarity_finds_a_tool_that_shares_no_word_with_the_request(
    fakes: list[AnyTool],
) -> None:
    scorer = FixedSimilarities({"clock_reading": 0.4, "destroy": 0.3, "echo": 0.7, "send": 0.2})
    selector = RelevanceToolSelector(max_tools=1, semantic=scorer)

    chosen = await selector.select("upprepa den här meningen", fakes)

    assert names(chosen) == ["echo"]
    assert scorer.requests == ["upprepa den här meningen"]


async def test_without_similarities_the_ranking_is_the_keyword_ranking(
    fakes: list[AnyTool],
) -> None:
    selector = RelevanceToolSelector(max_tools=3, semantic=FixedSimilarities(None))

    chosen = await selector.select("repeat this text", fakes)

    assert names(chosen) == names(
        RelevanceToolSelector(max_tools=3).choose("repeat this text", fakes)
    )


async def test_kept_tools_stay_offered_with_the_hybrid_ranking(fakes: list[AnyTool]) -> None:
    scorer = FixedSimilarities({"clock_reading": 0.1, "destroy": 0.1, "echo": 0.9, "send": 0.1})
    selector = RelevanceToolSelector(max_tools=2, semantic=scorer)

    chosen = await selector.select("upprepa", fakes, keep={"send"})

    assert names(chosen) == ["echo", "send"]


async def test_no_similarity_request_is_made_when_every_tool_is_offered(
    fakes: list[AnyTool],
) -> None:
    scorer = FixedSimilarities({})
    selector = RelevanceToolSelector(max_tools=ALL_TOOLS, semantic=scorer)

    assert await selector.select("anything", fakes) == fakes
    assert scorer.requests == []


# -- hit rate on the labelled cases against the real registry -----------------


@pytest.fixture(scope="module")
def registry_tools() -> list[AnyTool]:
    return ToolRegistry.discover().tools()


@pytest.fixture(scope="module")
def all_cases() -> list[dict[str, Any]]:
    loaded: dict[str, list[dict[str, Any]]] = yaml.safe_load(CASES_FILE.read_text("utf-8"))
    return loaded["cases"]


def subset(
    cases: list[dict[str, Any]], lang: str, *, holdout: bool = False
) -> list[dict[str, Any]]:
    return [
        case
        for case in cases
        if case.get("lang", "en") == lang and bool(case.get("holdout")) == holdout
    ]


def hit_rate(cases: list[dict[str, Any]], pick: Callable[[str], list[AnyTool]]) -> float:
    hits = sum(bool(set(case["expected"]) & set(names(pick(case["request"])))) for case in cases)
    return hits / len(cases)


def test_the_labelled_set_is_big_enough_and_names_real_tools(
    all_cases: list[dict[str, Any]], registry_tools: list[AnyTool]
) -> None:
    known = set(names(registry_tools))

    assert len(subset(all_cases, "en")) >= 25
    assert len(subset(all_cases, "sv")) >= 25
    assert len(subset(all_cases, "sv", holdout=True)) >= 25
    assert all(set(case["expected"]) <= known for case in all_cases)


@pytest.mark.parametrize("lang", ["en", "sv"])
@pytest.mark.parametrize(("k", "minimum"), [(1, MIN_TOP_1_HIT_RATE), (3, MIN_TOP_3_HIT_RATE)])
def test_ranking_puts_the_expected_tool_near_the_top(
    all_cases: list[dict[str, Any]],
    registry_tools: list[AnyTool],
    lang: str,
    k: int,
    minimum: float,
) -> None:
    selector = RelevanceToolSelector()

    rate = hit_rate(subset(all_cases, lang), lambda r: selector.rank(r, registry_tools)[:k])

    assert rate >= minimum


@pytest.mark.parametrize("lang", ["en", "sv"])
def test_default_ollama_selection_offers_the_expected_tool(
    all_cases: list[dict[str, Any]],
    registry_tools: list[AnyTool],
    make_settings: Callable[..., Settings],
    lang: str,
) -> None:
    selector = RelevanceToolSelector.from_settings(make_settings())
    cases = subset(all_cases, lang)

    rate = hit_rate(cases, lambda request: selector.choose(request, registry_tools))

    assert rate >= MIN_SELECTED_HIT_RATE
    assert all(len(selector.choose(c["request"], registry_tools)) <= 12 for c in cases)


def test_held_out_swedish_cases_were_not_used_to_write_keywords(
    all_cases: list[dict[str, Any]], registry_tools: list[AnyTool]
) -> None:
    selector = RelevanceToolSelector()
    held_out = subset(all_cases, "sv", holdout=True)

    rate = hit_rate(held_out, lambda request: selector.rank(request, registry_tools)[:12])

    assert rate <= MAX_HELD_OUT_KEYWORD_RATE


def test_the_measured_failure_cases_offer_the_right_tool(
    registry_tools: list[AnyTool], make_settings: Callable[..., Settings]
) -> None:
    selector = RelevanceToolSelector.from_settings(make_settings())

    assert selector.rank("what time is it?", registry_tools)[0].name == "current_time"
    assert selector.rank("hur mycket batteri har jag?", registry_tools)[0].name == "battery_status"
    assert selector.rank("lägg till mjölk på att-göra-listan", registry_tools)[0].name == "add_todo"
    assert selector.rank("vad står på min att-göra-lista?", registry_tools)[0].name == "list_todos"
