"""Tool: change the HUD's colour by voice ("byt färg till röd", "make the HUD purple").

A colour word resolves either to one of the HUD's presets (red, green,
violet ...) or, for colours without a preset (orange, blue, teal ...), to a
custom accent the native side derives a full palette from (ADR-0028). A hex
code (``#ff8800``) is a custom accent as given. The change goes through
the HUD's settings page (``ToolContext.hud``), so it is validated, saved to
``user_settings.json`` and applied live like a choice made by hand.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final

from pydantic import Field

from jarvis.agent.tool import Tool, ToolCategory, ToolContext, ToolInput, ToolResult
from jarvis.utils.colour import normalise_hex
from jarvis.utils.errors import JarvisError

CUSTOM_THEME: Final = "custom"
MAX_COLOUR_CHARS: Final = 40
NO_HUD_SPOKEN: Final = "There is no HUD running right now, so there is nothing to recolour."

PRESET_WORDS: Final[dict[str, str]] = {
    "cyan": "cyan",
    "default": "cyan",
    "standard": "cyan",
    "amber": "amber",
    "bärnsten": "amber",
    "red": "red",
    "röd": "red",
    "röda": "red",
    "rött": "red",
    "green": "green",
    "grön": "green",
    "gröna": "green",
    "grönt": "green",
    "violet": "violet",
    "purple": "violet",
    "lila": "violet",
    "violett": "violet",
    "purpur": "violet",
    "ice": "ice",
    "white": "ice",
    "vit": "ice",
    "vita": "ice",
    "vitt": "ice",
    "pink": "pink",
    "rosa": "pink",
}
"""Colour words (English and Swedish, inflected) that name a preset theme."""

CUSTOM_WORDS: Final[dict[str, str]] = {
    "blue": "#2f7bff",
    "blå": "#2f7bff",
    "blåa": "#2f7bff",
    "blått": "#2f7bff",
    "orange": "#ff8a1f",
    "yellow": "#ffd60a",
    "gul": "#ffd60a",
    "gula": "#ffd60a",
    "gult": "#ffd60a",
    "gold": "#ffc233",
    "guld": "#ffc233",
    "teal": "#19e6c1",
    "turquoise": "#19e6c1",
    "turkos": "#19e6c1",
    "magenta": "#ff2bd6",
    "lime": "#a6ff2e",
}
"""Colour words without a preset, as the accent a custom theme is derived from."""

_WORD: Final = re.compile(r"[^\W_]+")


class SetHudColourInput(ToolInput):
    """Arguments for :class:`SetHudColourTool`."""

    colour: str = Field(
        min_length=1,
        max_length=MAX_COLOUR_CHARS,
        description=(
            "The colour the user asked for, as said: a name in English or Swedish "
            "(red, röd, purple, lila, orange, white ...) or a hex code like #ff8800."
        ),
    )


@dataclass(frozen=True, slots=True)
class ResolvedColour:
    """What a spoken colour means for the HUD.

    Attributes:
        theme: A preset theme name, or ``custom``.
        accent: The custom accent (``#rrggbb``); None for a preset.
        name: How to say it back: the preset, the colour word or the hex code.
    """

    theme: str
    accent: str | None
    name: str


def resolve_colour(colour: str) -> ResolvedColour | None:
    """The HUD colour a spoken colour names, or None if it names none.

    A hex code is a custom accent as given. Otherwise the whole phrase is
    tried first, then each word in order, so "dark red please" finds red.
    """
    text = colour.strip().lower()
    try:
        accent = normalise_hex(text)
    except ValueError:
        pass  # not a hex code; try it as words
    else:
        return ResolvedColour(CUSTOM_THEME, accent, accent)
    for word in (text, *_WORD.findall(text)):
        if word in PRESET_WORDS:
            return ResolvedColour(PRESET_WORDS[word], None, PRESET_WORDS[word])
        if word in CUSTOM_WORDS:
            return ResolvedColour(CUSTOM_THEME, CUSTOM_WORDS[word], word)
    return None


class SetHudColourTool(Tool[SetHudColourInput]):
    """Recolours the HUD (and its orb) live, and remembers the choice."""

    name = "set_hud_colour"
    description = (
        "Change the colour (theme) of the Jarvis HUD overlay and its orb: presets cyan, "
        "amber, red, green, violet, ice white and pink, or any other colour name or hex "
        "code as a custom colour. Takes effect immediately and is remembered."
    )
    Input = SetHudColourInput
    category = ToolCategory.SYSTEM
    keywords = (
        "color",
        "colour",
        "theme",
        "overlay",
        "hud",
        "färg",
        "färgen",
        "färger",
        "tema",
        "temat",
        "red",
        "röd",
        "purple",
        "lila",
        "green",
        "grön",
        "pink",
        "rosa",
    )

    async def run(self, params: SetHudColourInput, context: ToolContext) -> ToolResult:
        """Resolve the colour and hand it to the HUD's settings."""
        resolved = resolve_colour(params.colour)
        if resolved is None:
            return ToolResult.failure(
                f"I don't know the colour {params.colour}. Try red, green, violet, pink, "
                "ice, amber, cyan, another colour name or a hex code.",
                detail=f"unresolved colour {params.colour!r}",
            )
        if context.hud is None:
            return ToolResult.failure(NO_HUD_SPOKEN, detail="no HUD in this front end")
        try:
            await context.hud.set_colour(resolved.theme, resolved.accent)
        except JarvisError as error:
            return ToolResult.failure(
                "I couldn't change the HUD colour.", detail=f"{type(error).__name__}: {error}"
            )
        return ToolResult.success(
            f"The HUD is now {resolved.name}.",
            data={"theme": resolved.theme, "accent": resolved.accent},
        )


# Design notes:
# - No confirmation: the change is cosmetic, instantly visible and undone
#   by asking again, like volume and brightness.
# - Colour words live here, not in the settings layer, because they are
#   about understanding speech; the settings only know theme names and hex
#   accents and validate both again, so a bad mapping can never be saved.
# - Colours without a preset map to fixed, saturated accents rather than to
#   CSS named colours: CSS "blue" (#0000ff) is too dark to glow, and the
#   palette derivation would only clamp it to something like #2f7bff anyway.
