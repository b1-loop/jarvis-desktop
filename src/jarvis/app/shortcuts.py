"""Windows shortcuts (desktop, Start menu, Startup folder) for the windowless app.

``scripts/install_shortcuts.py`` is the user-facing command; this module
holds everything it does so it is type-checked and tested:

* :func:`resolve_launcher` picks the program the shortcut runs: ``uvw.exe``
  next to the ``uv`` on PATH (``uvw run --project <repo> jarvis
  --background``), or the checkout's ``.venv`` ``pythonw.exe``
  (``-m jarvis --background``) when uv is not on PATH.
* :func:`build_shortcut_spec` is a pure function from folder, launcher and
  repository to the ``.lnk`` fields.
* :class:`ShortcutWriter` is the effect: :class:`PowerShellShortcutWriter`
  drives the ``WScript.Shell`` COM object through PowerShell, so no Python
  COM dependency is needed.
* A manifest under ``data/`` records exactly which files were created, so
  ``--uninstall`` removes those and nothing else.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Final, Protocol

from jarvis.utils.errors import JarvisError

SHORTCUT_FILENAME: Final = "Jarvis.lnk"
WINDOW_STYLE_MINIMIZED: Final = 7
"""``IWshShortcut.WindowStyle``: 1 normal, 3 maximised, 7 minimised."""
DESCRIPTION: Final = "Jarvis voice assistant (runs in the notification area)"
MANIFEST_FILENAME: Final = "desktop-shortcuts.json"
UV_EXECUTABLE: Final = "uv"
UVW_FILENAME: Final = "uvw.exe"
VENV_PYTHONW: Final = Path(".venv") / "Scripts" / "pythonw.exe"
BACKGROUND_FLAG: Final = "--background"
_POWERSHELL: Final = ("powershell.exe", "-NoProfile", "-NonInteractive", "-Command")
_ENV_PREFIX: Final = "JARVIS_LNK_"


class ShortcutError(JarvisError):
    """A shortcut could not be resolved, created or removed."""


class ShortcutLocation(StrEnum):
    """Where a shortcut goes; values are ``Environment.SpecialFolder`` names."""

    DESKTOP = "Desktop"
    START_MENU = "Programs"
    STARTUP = "Startup"


@dataclass(frozen=True, slots=True)
class Launcher:
    """The program a shortcut starts and its arguments (unquoted)."""

    target: Path
    arguments: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ShortcutSpec:
    """Every field of one ``.lnk`` file.

    Attributes:
        path: The shortcut file itself.
        target: Program it runs.
        arguments: Command line after the program, quoted for Windows.
        working_directory: The checkout, so ``.env`` and ``data/`` resolve.
        icon: ``.ico`` shown for the shortcut.
        description: Tooltip in Explorer.
        window_style: ``WINDOW_STYLE_MINIMIZED``; ``uvw`` shows no window
            anyway, the minimised style keeps any fallback console out of
            the way.
    """

    path: Path
    target: Path
    arguments: str
    working_directory: Path
    icon: Path
    description: str = DESCRIPTION
    window_style: int = WINDOW_STYLE_MINIMIZED


def resolve_launcher(repo: Path, which: Callable[[str], str | None] = shutil.which) -> Launcher:
    """``uvw.exe`` beside the real ``uv``, else the checkout's ``pythonw.exe``.

    ``uv`` on PATH is often a WinGet link; the link is resolved so the
    shortcut points at the installed file, which is where ``uvw.exe`` lives.

    Raises:
        ShortcutError: If neither launcher exists.
    """
    uv = which(UV_EXECUTABLE)
    if uv is not None:
        uvw = Path(uv).resolve().parent / UVW_FILENAME
        if uvw.is_file():
            return Launcher(uvw, ("run", "--project", str(repo), "jarvis", BACKGROUND_FLAG))
    pythonw = repo / VENV_PYTHONW
    if pythonw.is_file():
        return Launcher(pythonw, ("-m", "jarvis", BACKGROUND_FLAG))
    raise ShortcutError(
        f"found neither {UVW_FILENAME} next to uv on PATH nor {pythonw}; "
        "install uv, or run `uv sync --all-extras` in the checkout first"
    )


def build_shortcut_spec(folder: Path, launcher: Launcher, repo: Path, icon: Path) -> ShortcutSpec:
    """The shortcut ``folder/Jarvis.lnk`` that starts ``launcher`` in ``repo``."""
    return ShortcutSpec(
        path=folder / SHORTCUT_FILENAME,
        target=launcher.target,
        arguments=subprocess.list2cmdline(launcher.arguments),
        working_directory=repo,
        icon=icon,
    )


class ShortcutWriter(Protocol):
    """The Windows side: special folders and ``.lnk`` creation."""

    def folder(self, location: ShortcutLocation) -> Path:
        """Absolute path of ``location`` for the current user."""

    def create(self, spec: ShortcutSpec) -> None:
        """Write ``spec.path``, replacing an existing shortcut."""


_CREATE_SCRIPT: Final = (
    "$ErrorActionPreference = 'Stop'; "
    "$link = (New-Object -ComObject WScript.Shell).CreateShortcut($env:JARVIS_LNK_PATH); "
    "$link.TargetPath = $env:JARVIS_LNK_TARGET; "
    "$link.Arguments = $env:JARVIS_LNK_ARGUMENTS; "
    "$link.WorkingDirectory = $env:JARVIS_LNK_WORKDIR; "
    "$link.IconLocation = $env:JARVIS_LNK_ICON + ',0'; "
    "$link.Description = $env:JARVIS_LNK_DESCRIPTION; "
    "$link.WindowStyle = [int]$env:JARVIS_LNK_STYLE; "
    "$link.Save()"
)
"""Values arrive as environment variables, never spliced into the script text."""

_FOLDER_SCRIPT: Final = "[Environment]::GetFolderPath($env:JARVIS_LNK_FOLDER)"


def shortcut_environment(spec: ShortcutSpec) -> dict[str, str]:
    """The variables :data:`_CREATE_SCRIPT` reads for ``spec``."""
    values = {
        "PATH": str(spec.path),
        "TARGET": str(spec.target),
        "ARGUMENTS": spec.arguments,
        "WORKDIR": str(spec.working_directory),
        "ICON": str(spec.icon),
        "DESCRIPTION": spec.description,
        "STYLE": str(spec.window_style),
    }
    return {_ENV_PREFIX + key: value for key, value in values.items()}


class PowerShellShortcutWriter:
    """:class:`ShortcutWriter` through PowerShell and ``WScript.Shell``."""

    def folder(self, location: ShortcutLocation) -> Path:
        """Ask .NET for the folder, which follows OneDrive and policy redirection."""
        output = self._run(_FOLDER_SCRIPT, {_ENV_PREFIX + "FOLDER": location.value})
        if not output:
            raise ShortcutError(f"Windows reported no {location.value} folder")
        return Path(output)

    def create(self, spec: ShortcutSpec) -> None:
        """Create the ``.lnk`` with the COM object."""
        spec.path.parent.mkdir(parents=True, exist_ok=True)
        self._run(_CREATE_SCRIPT, shortcut_environment(spec))

    @staticmethod
    def _run(script: str, variables: Mapping[str, str]) -> str:
        completed = subprocess.run(  # noqa: S603 - fixed PowerShell, values via environment
            [*_POWERSHELL, script],
            env={**os.environ, **variables},
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            raise ShortcutError(f"PowerShell failed: {completed.stderr.strip()}")
        return completed.stdout.strip()


@dataclass(frozen=True, slots=True)
class ShortcutPlan:
    """What the user asked for: each location is wanted or not."""

    wanted: Mapping[ShortcutLocation, bool]


class Manifest:
    """The record of shortcuts this tool created, keyed by location.

    Args:
        path: The JSON file, normally ``data/desktop-shortcuts.json``.
    """

    def __init__(self, path: Path) -> None:
        self._path = path

    def load(self) -> dict[ShortcutLocation, Path]:
        """Recorded shortcuts; empty when nothing was installed yet.

        Raises:
            ShortcutError: If the file exists but is not a manifest.
        """
        if not self._path.is_file():
            return {}
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            return {ShortcutLocation(key): Path(value) for key, value in raw.items()}
        except (ValueError, AttributeError) as error:
            raise ShortcutError(f"{self._path} is not a shortcut manifest: {error}") from error

    def save(self, entries: Mapping[ShortcutLocation, Path]) -> None:
        """Replace the record; an empty record deletes the file."""
        if not entries:
            self._path.unlink(missing_ok=True)
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        data = {location.value: str(path) for location, path in entries.items()}
        self._path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def apply_plan(
    plan: ShortcutPlan,
    writer: ShortcutWriter,
    manifest: Manifest,
    make_spec: Callable[[Path], ShortcutSpec],
) -> dict[ShortcutLocation, Path]:
    """Create the wanted shortcuts and remove recorded ones that are no longer wanted.

    Returns:
        The shortcuts that exist afterwards, as recorded in the manifest.
    """
    entries = manifest.load()
    for location, wanted in plan.wanted.items():
        if wanted:
            spec = make_spec(writer.folder(location))
            writer.create(spec)
            entries[location] = spec.path
        elif location in entries:
            entries.pop(location).unlink(missing_ok=True)
    manifest.save(entries)
    return entries


def uninstall(manifest: Manifest) -> list[Path]:
    """Delete every shortcut in the manifest, then the manifest.

    Returns:
        The paths that were recorded (deleted, or already gone).
    """
    entries = manifest.load()
    removed = _remove(entries.values())
    manifest.save({})
    return removed


def _remove(paths: Iterable[Path]) -> list[Path]:
    removed = list(paths)
    for path in removed:
        path.unlink(missing_ok=True)
    return removed


# Design notes:
# - PowerShell + WScript.Shell instead of pywin32 or comtypes: the COM
#   object ships with every Windows, and creating three shortcuts once does
#   not justify a native Python dependency (CLAUDE.md: least dependencies).
# - Values are handed to PowerShell as environment variables, not
#   formatted into the script, so a repository path with quotes, `$` or
#   spaces cannot change what the script does.
# - Autostart is a shortcut in the user's Startup folder, never a Run key
#   in the registry: the user can see and delete it in Explorer (and Task
#   Manager's Startup apps lists it), and uninstall is a file delete.
# - The manifest is the only source of what to delete. Uninstall never
#   globs for "Jarvis.lnk", so a shortcut the user made by hand survives.
# - Arguments are quoted with subprocess.list2cmdline, the same rules the
#   C runtime uses to split them again in uvw/pythonw.
