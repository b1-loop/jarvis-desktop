"""The HUD's "Start with Windows" switch over the existing shortcut installer.

``scripts/install_shortcuts.py --autostart`` creates a ``Jarvis.lnk`` in the
user's Startup folder and records it in ``data/desktop-shortcuts.json``
(ADR-0021). :class:`StartupShortcut` drives the same code
(:func:`jarvis.app.shortcuts.apply_plan`) for that one location, so the HUD
and the script always agree, and the manifest still lists exactly what
``--uninstall`` will remove.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Final

from jarvis.app.icon import ICON_PATH
from jarvis.app.shortcuts import (
    MANIFEST_FILENAME,
    Launcher,
    Manifest,
    PowerShellShortcutWriter,
    ShortcutLocation,
    ShortcutPlan,
    ShortcutWriter,
    apply_plan,
    build_shortcut_spec,
    resolve_launcher,
)
from jarvis.utils.logging import get_logger

log = get_logger(__name__)

REPO_ROOT: Final = Path(__file__).resolve().parents[3]
"""The checkout: src/jarvis/app/autostart.py is three levels below it."""


class StartupShortcut:
    """``Autostart`` (ui.hud_settings) backed by the Startup-folder shortcut.

    Args:
        data_dir: Where the shortcut manifest lives (``Settings.data_dir``).
        repo: The checkout the shortcut starts Jarvis from.
        writer: Creates ``.lnk`` files; PowerShell + WScript.Shell by default.
        launcher: Picks ``uvw.exe`` or the venv's ``pythonw.exe``.
        icon: The shortcut's icon.
    """

    def __init__(
        self,
        data_dir: Path,
        *,
        repo: Path = REPO_ROOT,
        writer: ShortcutWriter | None = None,
        launcher: Callable[[Path], Launcher] = resolve_launcher,
        icon: Path = ICON_PATH,
    ) -> None:
        self._manifest = Manifest(data_dir / MANIFEST_FILENAME)
        self._repo = repo
        self._writer = writer or PowerShellShortcutWriter()
        self._launcher = launcher
        self._icon = icon

    def enabled(self) -> bool:
        """Whether a recorded Startup shortcut exists on disk."""
        path = self._manifest.load().get(ShortcutLocation.STARTUP)
        return path is not None and path.is_file()

    def set_enabled(self, enabled: bool) -> None:
        """Create or remove the Startup shortcut; other shortcuts are untouched.

        Blocking (PowerShell runs for about a second): call it off the event
        loop, as the settings controller does.

        Raises:
            ShortcutError: If the launcher cannot be found or the shortcut
                cannot be written.
        """
        launcher = self._launcher(self._repo)
        apply_plan(
            ShortcutPlan(wanted={ShortcutLocation.STARTUP: enabled}),
            self._writer,
            self._manifest,
            lambda folder: build_shortcut_spec(folder, launcher, self._repo, self._icon),
        )
        log.info("autostart.changed", enabled=enabled)


# Design notes:
# - A plan with only the STARTUP location: apply_plan leaves locations that
#   are not in the plan alone, so the desktop and Start-menu shortcuts the
#   user installed are never touched from the HUD.
# - The state is read from the manifest and the file on disk rather than
#   cached, so a shortcut deleted in Explorer shows as "off" next time.
