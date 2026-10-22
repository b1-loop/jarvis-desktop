"""Install Jarvis as a desktop app: desktop, Start-menu and autostart shortcuts.

Run from the checkout:

    uv run python scripts/install_shortcuts.py --autostart

Each **Jarvis** shortcut starts ``uvw run --project <checkout> jarvis
--background`` (no console window; the tray icon is the app). The **Jarvis
HUD** shortcut, on the desktop and in the Start menu, adds ``--show-hud``:
it starts Jarvis if it is not running and otherwise brings the HUD forward
and replays its boot animation, without opening the dashboard. Options say
what should exist afterwards: ``--no-desktop`` removes desktop shortcuts
this script made earlier, ``--no-hud`` removes the HUD shortcuts,
``--uninstall`` removes every shortcut it made. Only files recorded in
``data/desktop-shortcuts.json`` are ever deleted.

Windows only. The logic lives in :mod:`jarvis.app.shortcuts`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from jarvis.app.icon import HUD_ICON_PATH, ICON_PATH
from jarvis.app.shortcuts import (
    MANIFEST_FILENAME,
    Launcher,
    Manifest,
    PowerShellShortcutWriter,
    ShortcutError,
    ShortcutKind,
    ShortcutLocation,
    ShortcutPlan,
    ShortcutSlot,
    ShortcutSpec,
    apply_plan,
    build_shortcut_spec,
    resolve_launcher,
    uninstall,
)

REPO = Path(__file__).resolve().parents[1]
MANIFEST = REPO / "data" / MANIFEST_FILENAME
LOG_PATH = REPO / "data" / "logs" / "jarvis.log"
ICONS = {ShortcutKind.JARVIS: ICON_PATH, ShortcutKind.HUD: HUD_ICON_PATH}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Command-line options; every location is an on/off switch."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    toggle = argparse.BooleanOptionalAction
    parser.add_argument("--desktop", action=toggle, default=True, help="desktop shortcut")
    parser.add_argument("--start-menu", action=toggle, default=True, help="Start-menu entry")
    parser.add_argument(
        "--autostart", action=toggle, default=False, help="start with Windows (Startup folder)"
    )
    parser.add_argument(
        "--hud",
        action=toggle,
        default=True,
        help="'Jarvis HUD' shortcuts (desktop, Start menu): show the HUD and its animation",
    )
    parser.add_argument(
        "--uninstall", action="store_true", help="remove every shortcut this script created"
    )
    return parser.parse_args(argv)


def plan_from(args: argparse.Namespace) -> ShortcutPlan:
    """The wanted state of each slot.

    The HUD shortcut exists where a Jarvis shortcut is wanted on the desktop
    or in the Start menu, never in the Startup folder: starting with Windows
    already plays the boot animation.
    """
    jarvis, hud = ShortcutKind.JARVIS, ShortcutKind.HUD
    return ShortcutPlan(
        wanted={
            ShortcutSlot(jarvis, ShortcutLocation.DESKTOP): args.desktop,
            ShortcutSlot(jarvis, ShortcutLocation.START_MENU): args.start_menu,
            ShortcutSlot(jarvis, ShortcutLocation.STARTUP): args.autostart,
            ShortcutSlot(hud, ShortcutLocation.DESKTOP): args.desktop and args.hud,
            ShortcutSlot(hud, ShortcutLocation.START_MENU): args.start_menu and args.hud,
        }
    )


def spec_for(launcher: Launcher, slot: ShortcutSlot, folder: Path) -> ShortcutSpec:
    """The ``.lnk`` for ``slot`` in ``folder``, with that kind's icon."""
    return build_shortcut_spec(folder, launcher, REPO, ICONS[slot.kind], slot.kind)


def install(args: argparse.Namespace) -> None:
    """Create or remove shortcuts as asked and print what exists now."""
    launcher = resolve_launcher(REPO)
    entries = apply_plan(
        plan_from(args),
        PowerShellShortcutWriter(),
        Manifest(MANIFEST),
        lambda slot, folder: spec_for(launcher, slot, folder),
    )
    print(f"Launcher: {launcher.target} {' '.join(launcher.arguments)}")
    for slot, path in entries.items():
        print(f"  {slot.location.name.lower():<10} {path}")
    print(f"Log file: {LOG_PATH}")
    print("Remove with: uv run python scripts/install_shortcuts.py --uninstall")


def main(argv: list[str] | None = None) -> int:
    """Run the command; returns the exit code."""
    if sys.platform != "win32":
        print("Shortcuts are a Windows feature; nothing to do here.", file=sys.stderr)
        return 1
    args = parse_args(argv)
    try:
        if args.uninstall:
            for path in uninstall(Manifest(MANIFEST)):
                print(f"Removed {path}")
            return 0
        install(args)
    except ShortcutError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
