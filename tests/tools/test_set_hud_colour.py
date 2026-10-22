"""set_hud_colour: spoken colours to HUD themes, and the HUD it hands them to."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

import pytest

from jarvis.agent.registry import ToolRegistry
from jarvis.agent.tool_selection import RelevanceToolSelector
from jarvis.config import Settings
from jarvis.tools.set_hud_colour import (
    NO_HUD_SPOKEN,
    ResolvedColour,
    SetHudColourTool,
    resolve_colour,
)
from jarvis.utils.errors import JarvisError
from tests.tools.local_fakes import local_context


class FakeHud:
    def __init__(self, error: JarvisError | None = None) -> None:
        self.calls: list[tuple[str, str | None]] = []
        self._error = error

    async def set_colour(self, theme: str, accent: str | None = None) -> None:
        if self._error is not None:
            raise self._error
        self.calls.append((theme, accent))


@pytest.mark.parametrize(
    ("said", "expected"),
    [
        ("red", ResolvedColour("red", None, "red")),
        ("röd", ResolvedColour("red", None, "red")),
        ("lila", ResolvedColour("violet", None, "violet")),
        ("Purple", ResolvedColour("violet", None, "violet")),
        ("vit", ResolvedColour("ice", None, "ice")),
        ("dark green please", ResolvedColour("green", None, "green")),
        ("orange", ResolvedColour("custom", "#ff8a1f", "orange")),
        ("blå", ResolvedColour("custom", "#2f7bff", "blå")),
        ("#FF8800", ResolvedColour("custom", "#ff8800", "#ff8800")),
        ("standard", ResolvedColour("cyan", None, "cyan")),
    ],
)
def test_spoken_colours_resolve_to_a_preset_or_a_custom_accent(
    said: str, expected: ResolvedColour
) -> None:
    assert resolve_colour(said) == expected


def test_a_word_that_names_no_colour_resolves_to_nothing() -> None:
    assert resolve_colour("iron man") is None


async def test_the_colour_is_handed_to_the_hud_without_confirmation(
    make_settings: Callable[..., Settings],
) -> None:
    hud = FakeHud()
    context = replace(local_context(make_settings()), hud=hud)
    tool = SetHudColourTool()

    result = await tool.call({"colour": "lila"}, context)
    custom = await tool.call({"colour": "orange"}, context)

    assert result.ok
    assert result.spoken == "The HUD is now violet."
    assert custom.spoken == "The HUD is now orange."
    assert hud.calls == [("violet", None), ("custom", "#ff8a1f")]
    assert not tool.requires_confirmation


async def test_an_unknown_colour_is_explained_and_nothing_changes(
    make_settings: Callable[..., Settings],
) -> None:
    hud = FakeHud()

    result = await SetHudColourTool().call(
        {"colour": "iron man"}, replace(local_context(make_settings()), hud=hud)
    )

    assert not result.ok
    assert "don't know the colour iron man" in result.spoken
    assert hud.calls == []


async def test_without_a_hud_the_tool_says_so(make_settings: Callable[..., Settings]) -> None:
    result = await SetHudColourTool().call({"colour": "red"}, local_context(make_settings()))

    assert not result.ok
    assert result.spoken == NO_HUD_SPOKEN


async def test_a_rejected_value_is_a_failure_not_a_crash(
    make_settings: Callable[..., Settings],
) -> None:
    hud = FakeHud(error=JarvisError("'x' is not a valid value for overlay_theme"))

    result = await SetHudColourTool().call(
        {"colour": "red"}, replace(local_context(make_settings()), hud=hud)
    )

    assert not result.ok
    assert "not a valid value" in result.detail


@pytest.mark.parametrize(
    "request_text",
    ["byt färg till röd", "change the HUD colour to purple", "gör färgen lila", "make it pink"],
)
def test_the_selector_offers_the_tool_for_colour_requests(
    make_settings: Callable[..., Settings], request_text: str
) -> None:
    tools = ToolRegistry.discover().tools()
    selector = RelevanceToolSelector.from_settings(make_settings())

    assert "set_hud_colour" in [tool.name for tool in selector.choose(request_text, tools)]
