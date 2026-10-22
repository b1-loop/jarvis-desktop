"""Regenerate ``src/jarvis/app/assets/jarvis.ico`` from the tray icon artwork.

Run after changing the state palette in ``jarvis.ui.view_model``:

    uv run python scripts/make_icon.py

The drawing itself lives in :mod:`jarvis.app.icon` so it is type-checked
and tested with the rest of the package; this script only calls it.
"""

from __future__ import annotations

from jarvis.app.icon import ICON_PATH, ICON_SIZES, write_app_icon


def main() -> None:
    """Write the icon and say where it went."""
    path = write_app_icon(ICON_PATH)
    sizes = ", ".join(str(size) for size in ICON_SIZES)
    print(f"Wrote {path} ({sizes} px).")


if __name__ == "__main__":
    main()
