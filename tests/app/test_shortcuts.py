import subprocess
import sys
from pathlib import Path

import pytest

from jarvis.app.icon import ICON_PATH
from jarvis.app.shortcuts import (
    SHORTCUT_FILENAME,
    WINDOW_STYLE_MINIMIZED,
    Launcher,
    Manifest,
    PowerShellShortcutWriter,
    ShortcutError,
    ShortcutLocation,
    ShortcutPlan,
    ShortcutSpec,
    apply_plan,
    build_shortcut_spec,
    resolve_launcher,
    shortcut_environment,
    uninstall,
)


def touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")
    return path


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    return tmp_path / "Jarvis Checkout"


# -- launcher -----------------------------------------------------------------------


def test_uvw_next_to_uv_is_preferred(tmp_path: Path, repo: Path) -> None:
    uv = touch(tmp_path / "uv-install" / "uv.exe")
    uvw = touch(tmp_path / "uv-install" / "uvw.exe")
    touch(repo / ".venv" / "Scripts" / "pythonw.exe")

    launcher = resolve_launcher(repo, lambda _name: str(uv))

    assert launcher.target == uvw.resolve()
    assert launcher.arguments == ("run", "--project", str(repo), "jarvis", "--background")


@pytest.mark.skipif(sys.platform == "win32", reason="symlinks need developer mode on Windows")
def test_a_linked_uv_is_resolved_to_its_install_directory(tmp_path: Path, repo: Path) -> None:
    uv = touch(tmp_path / "package" / "uv.exe")
    uvw = touch(tmp_path / "package" / "uvw.exe")
    link = tmp_path / "Links" / "uv.exe"
    link.parent.mkdir()
    link.symlink_to(uv)

    assert resolve_launcher(repo, lambda _name: str(link)).target == uvw.resolve()


def test_venv_pythonw_is_the_fallback(tmp_path: Path, repo: Path) -> None:
    uv = touch(tmp_path / "uv-without-uvw" / "uv.exe")
    pythonw = touch(repo / ".venv" / "Scripts" / "pythonw.exe")

    launcher = resolve_launcher(repo, lambda _name: str(uv))

    assert launcher == Launcher(pythonw, ("-m", "jarvis", "--background"))


def test_no_launcher_is_an_error_with_the_remedy(repo: Path) -> None:
    with pytest.raises(ShortcutError, match="uv sync"):
        resolve_launcher(repo, lambda _name: None)


# -- spec -----------------------------------------------------------------------------


def test_spec_quotes_arguments_and_starts_minimised_in_the_checkout(
    tmp_path: Path, repo: Path
) -> None:
    uvw = tmp_path / "uvw.exe"
    launcher = Launcher(uvw, ("run", "--project", str(repo), "jarvis", "--background"))

    spec = build_shortcut_spec(tmp_path / "Desktop", launcher, repo, ICON_PATH)

    assert spec.path == tmp_path / "Desktop" / SHORTCUT_FILENAME
    assert spec.target == uvw
    assert spec.arguments == f'run --project "{repo}" jarvis --background'
    assert spec.working_directory == repo
    assert spec.icon == ICON_PATH
    assert spec.window_style == WINDOW_STYLE_MINIMIZED


def test_values_reach_powershell_only_through_the_environment(tmp_path: Path) -> None:
    spec = ShortcutSpec(
        path=tmp_path / "a'b$c.lnk",
        target=tmp_path / "uvw.exe",
        arguments='run "x"',
        working_directory=tmp_path,
        icon=ICON_PATH,
    )

    env = shortcut_environment(spec)

    assert env["JARVIS_LNK_PATH"] == str(spec.path)
    assert env["JARVIS_LNK_ARGUMENTS"] == 'run "x"'
    assert env["JARVIS_LNK_STYLE"] == "7"
    assert set(env) == {
        "JARVIS_LNK_PATH",
        "JARVIS_LNK_TARGET",
        "JARVIS_LNK_ARGUMENTS",
        "JARVIS_LNK_WORKDIR",
        "JARVIS_LNK_ICON",
        "JARVIS_LNK_DESCRIPTION",
        "JARVIS_LNK_STYLE",
    }


# -- install / uninstall with a fake writer -------------------------------------------------


class FakeWriter:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.created: list[ShortcutSpec] = []

    def folder(self, location: ShortcutLocation) -> Path:
        return self.root / location.value

    def create(self, spec: ShortcutSpec) -> None:
        touch(spec.path)
        self.created.append(spec)


def make_spec(folder: Path) -> ShortcutSpec:
    return build_shortcut_spec(folder, Launcher(Path("uvw.exe"), ()), Path("repo"), ICON_PATH)


def plan(*, desktop: bool, start_menu: bool, startup: bool) -> ShortcutPlan:
    return ShortcutPlan(
        wanted={
            ShortcutLocation.DESKTOP: desktop,
            ShortcutLocation.START_MENU: start_menu,
            ShortcutLocation.STARTUP: startup,
        }
    )


def test_install_creates_the_wanted_shortcuts_and_records_them(tmp_path: Path) -> None:
    writer = FakeWriter(tmp_path / "shell")
    manifest = Manifest(tmp_path / "data" / "desktop-shortcuts.json")

    entries = apply_plan(
        plan(desktop=True, start_menu=True, startup=True), writer, manifest, make_spec
    )

    assert set(entries) == set(ShortcutLocation)
    assert entries[ShortcutLocation.STARTUP] == tmp_path / "shell" / "Startup" / "Jarvis.lnk"
    assert all(path.is_file() for path in entries.values())
    assert manifest.load() == entries


def test_turning_a_location_off_removes_only_the_recorded_shortcut(tmp_path: Path) -> None:
    writer = FakeWriter(tmp_path / "shell")
    manifest = Manifest(tmp_path / "manifest.json")
    apply_plan(plan(desktop=True, start_menu=False, startup=True), writer, manifest, make_spec)
    users_own = touch(tmp_path / "shell" / "Programs" / "Jarvis.lnk")

    entries = apply_plan(
        plan(desktop=True, start_menu=False, startup=False), writer, manifest, make_spec
    )

    assert set(entries) == {ShortcutLocation.DESKTOP}
    assert not (tmp_path / "shell" / "Startup" / "Jarvis.lnk").exists()
    assert users_own.exists()  # never recorded, so never deleted


def test_uninstall_removes_exactly_the_recorded_files_and_the_manifest(tmp_path: Path) -> None:
    writer = FakeWriter(tmp_path / "shell")
    manifest_path = tmp_path / "manifest.json"
    manifest = Manifest(manifest_path)
    entries = apply_plan(
        plan(desktop=True, start_menu=True, startup=False), writer, manifest, make_spec
    )
    unrelated = touch(tmp_path / "shell" / "Desktop" / "Other.lnk")
    entries[ShortcutLocation.DESKTOP].unlink()  # the user already deleted one by hand

    removed = uninstall(manifest)

    assert sorted(removed) == sorted(entries.values())
    assert not any(path.exists() for path in entries.values())
    assert unrelated.exists()
    assert not manifest_path.exists()
    assert uninstall(manifest) == []


def test_a_corrupt_manifest_is_reported(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text('{"Nowhere": "x"}', encoding="utf-8")

    with pytest.raises(ShortcutError, match="not a shortcut manifest"):
        Manifest(path).load()


def test_powershell_failure_is_a_shortcut_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess([], 1, stdout="", stderr="COM error")

    monkeypatch.setattr(subprocess, "run", fail)

    with pytest.raises(ShortcutError, match="COM error"):
        PowerShellShortcutWriter().folder(ShortcutLocation.DESKTOP)


# -- the real COM call (Windows desktop only) -------------------------------------------------


@pytest.mark.hardware
@pytest.mark.skipif(sys.platform != "win32", reason="WScript.Shell is a Windows COM object")
def test_powershell_writer_creates_a_real_shortcut(tmp_path: Path) -> None:
    writer = PowerShellShortcutWriter()
    spec = build_shortcut_spec(
        tmp_path / "Shortcuts",
        Launcher(Path(sys.executable), ("-c", "pass")),
        tmp_path,
        ICON_PATH,
    )

    writer.create(spec)

    assert spec.path.is_file()
    assert writer.folder(ShortcutLocation.STARTUP).name == "Startup"
    spec.path.unlink()
    assert not spec.path.exists()
