"""Relaunching from the HUD and the "Start with Windows" shortcut. No process is started."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from jarvis.app.autostart import StartupShortcut
from jarvis.app.restart import Relaunch, relaunch_command
from jarvis.app.shortcuts import (
    Launcher,
    Manifest,
    ShortcutKind,
    ShortcutLocation,
    ShortcutSlot,
    ShortcutSpec,
)


def test_the_relaunch_command_restarts_the_windowless_app() -> None:
    assert relaunch_command(wake_word=True, executable="py") == [
        "py",
        "-m",
        "jarvis",
        "--background",
    ]
    assert relaunch_command(wake_word=False, executable="py")[-1] == "--no-wakeword"


def test_relaunch_spawns_only_when_requested(tmp_path: Path) -> None:
    spawned: list[tuple[list[str], Path]] = []

    def spawn(command: Sequence[str], cwd: Path) -> None:
        spawned.append((list(command), cwd))

    relaunch = Relaunch(wake_word=True, spawn=spawn, cwd=tmp_path)
    assert not relaunch.run_if_requested()

    relaunch.request()
    assert relaunch.requested
    assert relaunch.run_if_requested()
    assert spawned[0][0][-1] == "--background"
    assert spawned[0][1] == tmp_path


class FakeWriter:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.created: list[ShortcutSpec] = []

    def folder(self, location: ShortcutLocation) -> Path:
        folder = self.root / location.value
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    def create(self, spec: ShortcutSpec) -> None:
        spec.path.write_text("lnk", encoding="utf-8")
        self.created.append(spec)


def shortcut(tmp_path: Path) -> tuple[StartupShortcut, FakeWriter]:
    writer = FakeWriter(tmp_path / "shell")
    built = StartupShortcut(
        tmp_path / "data",
        repo=tmp_path,
        writer=writer,
        launcher=lambda repo: Launcher(repo / "uvw.exe", ("run", "jarvis", "--background")),
        icon=tmp_path / "jarvis.ico",
    )
    return built, writer


def test_start_with_windows_creates_and_removes_only_the_startup_shortcut(tmp_path: Path) -> None:
    autostart, writer = shortcut(tmp_path)
    desktop = writer.folder(ShortcutLocation.DESKTOP) / "Jarvis.lnk"
    desktop.write_text("lnk", encoding="utf-8")
    manifest = Manifest(tmp_path / "data" / "desktop-shortcuts.json")
    desktop_slot = ShortcutSlot(ShortcutKind.JARVIS, ShortcutLocation.DESKTOP)
    manifest.save({desktop_slot: desktop})
    assert not autostart.enabled()

    autostart.set_enabled(True)
    assert autostart.enabled()
    assert writer.created[0].path.parent.name == "Startup"

    autostart.set_enabled(False)
    assert not autostart.enabled()
    assert desktop.exists()
    assert manifest.load() == {desktop_slot: desktop}


def test_a_shortcut_deleted_in_explorer_reads_as_off(tmp_path: Path) -> None:
    autostart, writer = shortcut(tmp_path)
    autostart.set_enabled(True)

    writer.created[0].path.unlink()

    assert not autostart.enabled()
