"""Hex colour values as settings hold them: ``"#rrggbb"``, lower case.

The HUD's custom accent (``overlay_accent``) is stored, validated and sent
to the native overlay in this one spelling, so ``.env``,
``user_settings.json``, the settings page and the voice tool can never
disagree about what "the same colour" looks like. The palette itself is
derived natively (``native/overlay/palette.hpp``); Python only checks the
spelling.
"""

from __future__ import annotations

import re
from typing import Annotated, Final

from pydantic import AfterValidator

DEFAULT_ACCENT: Final = "#4dd9ff"
"""The original cyan HUD's accent; the custom theme starts from it."""

_HEX: Final = re.compile(r"#?([0-9a-fA-F]{6})")


def normalise_hex(value: str) -> str:
    """``value`` as ``"#rrggbb"``; ``"FF3B30"`` and ``"#ff3b30"`` are the same colour.

    Raises:
        ValueError: If ``value`` is not six hex digits, with or without ``#``.
    """
    match = _HEX.fullmatch(value.strip())
    if match is None:
        raise ValueError(f"{value!r} is not a colour; expected '#rrggbb', e.g. '#ff3b30'")
    return f"#{match.group(1).lower()}"


HexColour = Annotated[str, AfterValidator(normalise_hex)]
"""A pydantic field type that accepts any spelling of a hex colour and stores ``#rrggbb``.

An ``Annotated`` validator rather than a ``field_validator`` so the HUD's
per-field check (``user_settings.validate_field_value``, which validates the
annotation alone) applies it too.
"""
