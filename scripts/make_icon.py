"""Regenerate the shortcut icons in ``src/jarvis/app/assets/`` from the tray artwork.

``jarvis.ico`` (the app, idle colour) and ``jarvis-hud.ico`` (the "Jarvis
HUD" shortcut, listening colour).

Run after changing the state palette in ``jarvis.ui.view_model``:

    uv run python scripts/make_icon.py

The drawing itself lives in :mod:`jarvis.app.icon` so it is type-checked
and tested with the rest of the package; this script only calls it.
"""

from __future__ import annotations

from jarvis.app.icon import (
    APP_ICON_STATE,
    HUD_ICON_PATH,
    HUD_ICON_STATE,
    ICON_PATH,
    ICON_SIZES,
    write_app_icon,
)


def main() -> None:
    """Write both icons and say where they went."""
    sizes = ", ".join(str(size) for size in ICON_SIZES)
    for path, state in ((ICON_PATH, APP_ICON_STATE), (HUD_ICON_PATH, HUD_ICON_STATE)):
        written = write_app_icon(path, state)
        print(f"Wrote {written} ({sizes} px).")


if __name__ == "__main__":
    main()
