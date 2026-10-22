from collections.abc import Callable
from pathlib import Path
from typing import Any, Final

import pytest
import yaml

from jarvis.agent.registry import AnyTool, ToolRegistry
from jarvis.agent.tool import Tool, ToolContext, ToolInput, ToolResult
from jarvis.agent.tool_selection import ALL_TOOLS, RelevanceToolSelector, terms
from jarvis.config import RuntimeMode, Settings
from tests.agent.fakes import DestructiveTool, EchoTool, SendTool

CASES_FILE: Final = Path(__file__).parent / "data" / "tool_selection_cases.yaml"

MIN_TOP_1_HIT_RATE: Final = 0.85
"""Measured 0.89 with keywords (0.80 without); ADR-0020."""

MIN_TOP_3_HIT_RATE: Final = 0.95
"""Measured 1.00 with keywords (0.91 without)."""

MIN_SELECTED_HIT_RATE: Final = 1.0
"""What the model is actually offered under the default Ollama settings."""


class _ClockInput(ToolInput):
    pass


class ClockTool(Tool[_ClockInput]):
    name = "clock_reading"
    description = "Read the hands."
    Input = _ClockInput
    keywords = ("time", "hour")

    async def run(self, params: _ClockInput, context: ToolContext) -> ToolResult:
        return ToolResult.success("Noon.")


def names(tools: list[AnyTool]) -> list[str]:
    return [tool.name for tool in tools]


@pytest.fixture
def fakes() -> list[AnyTool]:
    return [ClockTool(), DestructiveTool(), EchoTool(), SendTool()]


def test_terms_drop_stopwords_and_plural_s() -> None:
    assert terms("What's in my emails, please?") == {"email"}
    assert terms("address the glass") == {"address", "glass"}


def test_zero_cap_offers_every_tool(fakes: list[AnyTool]) -> None:
    selector = RelevanceToolSelector(max_tools=ALL_TOOLS)

    assert selector.select("anything", fakes) == fakes


def test_keyword_match_outranks_a_description_match(fakes: list[AnyTool]) -> None:
    selector = RelevanceToolSelector(max_tools=1)

    assert names(selector.select("what time is it", fakes)) == ["clock_reading"]


def test_unrelated_tools_are_not_offered_even_with_room_left(fakes: list[AnyTool]) -> None:
    selector = RelevanceToolSelector(max_tools=3)

    assert names(selector.select("repeat this text", fakes)) == ["echo"]


def test_core_and_kept_tools_are_always_offered_in_registry_order(
    fakes: list[AnyTool],
) -> None:
    selector = RelevanceToolSelector(max_tools=3, core=["send"])

    chosen = selector.select("repeat this text", fakes, keep={"destroy"})

    assert names(chosen) == ["destroy", "echo", "send"]


def test_pinned_tools_may_exceed_the_cap(fakes: list[AnyTool]) -> None:
    selector = RelevanceToolSelector(max_tools=2, core=["send", "clock_reading"])

    chosen = selector.select("repeat this text", fakes, keep={"destroy"})

    assert names(chosen) == ["clock_reading", "destroy", "send"]


def test_unknown_core_names_are_ignored(fakes: list[AnyTool]) -> None:
    selector = RelevanceToolSelector(max_tools=2, core=["no_such_tool"])

    assert names(selector.select("repeat this text", fakes)) == ["echo"]


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


# -- hit rate on the labelled cases against the real registry -----------------


@pytest.fixture(scope="module")
def registry_tools() -> list[AnyTool]:
    return ToolRegistry.discover().tools()


@pytest.fixture(scope="module")
def cases() -> list[dict[str, Any]]:
    loaded: dict[str, list[dict[str, Any]]] = yaml.safe_load(CASES_FILE.read_text("utf-8"))
    return loaded["cases"]


def hit_rate(cases: list[dict[str, Any]], pick: Callable[[str], list[AnyTool]]) -> float:
    hits = sum(bool(set(case["expected"]) & set(names(pick(case["request"])))) for case in cases)
    return hits / len(cases)


def test_the_labelled_set_is_big_enough_and_names_real_tools(
    cases: list[dict[str, Any]], registry_tools: list[AnyTool]
) -> None:
    known = set(names(registry_tools))

    assert len(cases) >= 25
    assert all(set(case["expected"]) <= known for case in cases)


@pytest.mark.parametrize(("k", "minimum"), [(1, MIN_TOP_1_HIT_RATE), (3, MIN_TOP_3_HIT_RATE)])
def test_ranking_puts_the_expected_tool_near_the_top(
    cases: list[dict[str, Any]], registry_tools: list[AnyTool], k: int, minimum: float
) -> None:
    selector = RelevanceToolSelector()

    rate = hit_rate(cases, lambda request: selector.rank(request, registry_tools)[:k])

    assert rate >= minimum


def test_default_ollama_selection_offers_the_expected_tool(
    cases: list[dict[str, Any]],
    registry_tools: list[AnyTool],
    make_settings: Callable[..., Settings],
) -> None:
    selector = RelevanceToolSelector.from_settings(make_settings())

    rate = hit_rate(cases, lambda request: selector.select(request, registry_tools))

    assert rate >= MIN_SELECTED_HIT_RATE
    assert all(len(selector.select(c["request"], registry_tools)) <= 12 for c in cases)


def test_the_measured_failure_case_offers_current_time_not_only_the_shell(
    registry_tools: list[AnyTool], make_settings: Callable[..., Settings]
) -> None:
    selector = RelevanceToolSelector.from_settings(make_settings())

    ranked = selector.rank("what time is it?", registry_tools)

    assert ranked[0].name == "current_time"
