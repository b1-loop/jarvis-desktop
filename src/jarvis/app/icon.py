"""The application icon (shortcuts, Start menu), drawn from the tray's artwork.

The tray icon is rendered at runtime by :func:`jarvis.ui.tray.render_state_icon`;
the ``.ico`` that shortcuts point to is the same drawing in the idle colour,
rendered at every size Windows asks for, so the taskbar, the Start menu and
the notification area show one consistent mark. The "Jarvis HUD" shortcut
gets the same mark in the listening colour (:data:`HUD_ICON_PATH`), so the
two shortcuts are told apart at a glance and still read as one app.
``scripts/make_icon.py`` regenerates both committed files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

from jarvis.core.state_machine import AssistantState
from jarvis.ui.tray import render_state_icon

ICON_PATH: Final = Path(__file__).resolve().parent / "assets" / "jarvis.ico"
"""Committed icon file the Jarvis shortcuts reference."""

HUD_ICON_PATH: Final = ICON_PATH.with_name("jarvis-hud.ico")
"""Committed icon file of the "Jarvis HUD" shortcut."""

APP_ICON_STATE: Final = AssistantState.IDLE
"""The tray colour the app icon is drawn in."""

HUD_ICON_STATE: Final = AssistantState.LISTENING
"""The tray colour of the HUD icon: the HUD lit up, as the shortcut makes it."""

ICON_SIZES: Final[tuple[int, ...]] = (16, 20, 24, 32, 40, 48, 64, 128, 256)
"""Sizes Windows picks from: small icons, high-DPI taskbar, Start tiles, Explorer."""

_LARGEST: Final = max(ICON_SIZES)


def write_app_icon(path: Path = ICON_PATH, state: AssistantState = APP_ICON_STATE) -> Path:
    """Render the tray artwork for ``state`` at every size and save it as one ``.ico``.

    Each size is drawn natively rather than downscaled from 256 px, so the
    dark ring stays crisp at 16 px instead of blurring into the disc.

    Returns:
        The path written.
    """
    images = {size: render_state_icon(state, size) for size in ICON_SIZES}
    path.parent.mkdir(parents=True, exist_ok=True)
    largest = images[_LARGEST]
    largest.save(
        path,
        format="ICO",
        sizes=[(size, size) for size in ICON_SIZES],
        append_images=[images[size] for size in ICON_SIZES if size != _LARGEST],
    )
    return path


# Design notes:
# - The icon is generated, not drawn by hand, so the tray and the shortcut
#   can never drift apart: change the palette in ui/view_model.py, rerun
#   scripts/make_icon.py, and both follow.
# - The .ico is committed (a few tens of KB) because shortcuts are created
#   from a checkout that may not have Pillow's desktop extra installed.
