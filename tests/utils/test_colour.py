"""Hex colour spelling shared by settings, the HUD page and the voice tool."""

from __future__ import annotations

import pytest
from pydantic import TypeAdapter, ValidationError

from jarvis.utils.colour import DEFAULT_ACCENT, HexColour, normalise_hex


@pytest.mark.parametrize(
    ("text", "expected"),
    [("#ff3b30", "#ff3b30"), ("FF3B30", "#ff3b30"), ("  #Ab12eF ", "#ab12ef")],
)
def test_any_spelling_of_six_hex_digits_is_normalised(text: str, expected: str) -> None:
    assert normalise_hex(text) == expected


@pytest.mark.parametrize("text", ["", "#", "#fff", "#ff3b300", "#gg0000", "red", "#ff 3b30"])
def test_anything_else_is_rejected(text: str) -> None:
    with pytest.raises(ValueError, match="not a colour"):
        normalise_hex(text)


def test_the_annotated_type_validates_on_its_own() -> None:
    adapter: TypeAdapter[str] = TypeAdapter(HexColour)

    assert adapter.validate_python("4DD9FF") == DEFAULT_ACCENT
    with pytest.raises(ValidationError):
        adapter.validate_python("cyan")
