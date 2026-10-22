import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

from jarvis.app.icon import HUD_ICON_PATH, ICON_PATH
from jarvis.app.shortcuts import Launcher, ShortcutKind, ShortcutLocation, ShortcutSlot

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "install_shortcuts.py"


def load_script() -> ModuleType:
    # scripts/ is not a package, so the module is loaded from its file path.
    spec = importlib.util.spec_from_file_location("install_shortcuts", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


install_shortcuts = load_script()

DESKTOP, START_MENU, STARTUP = (
    ShortcutLocation.DESKTOP,
    ShortcutLocation.START_MENU,
    ShortcutLocation.STARTUP,
)


def jarvis(location: ShortcutLocation) -> ShortcutSlot:
    return ShortcutSlot(ShortcutKind.JARVIS, location)


def hud(location: ShortcutLocation) -> ShortcutSlot:
    return ShortcutSlot(ShortcutKind.HUD, location)


def test_defaults_are_jarvis_and_hud_on_desktop_and_start_menu_without_autostart() -> None:
    plan = install_shortcuts.plan_from(install_shortcuts.parse_args([]))

    assert plan.wanted == {
        jarvis(DESKTOP): True,
        jarvis(START_MENU): True,
        jarvis(STARTUP): False,
        hud(DESKTOP): True,
        hud(START_MENU): True,
    }


def test_every_location_can_be_switched() -> None:
    args = install_shortcuts.parse_args(["--autostart", "--no-desktop", "--no-start-menu"])

    assert install_shortcuts.plan_from(args).wanted == {
        jarvis(DESKTOP): False,
        jarvis(START_MENU): False,
        jarvis(STARTUP): True,
        hud(DESKTOP): False,
        hud(START_MENU): False,
    }
    assert not args.uninstall


def test_no_hud_removes_only_the_hud_shortcuts() -> None:
    wanted = install_shortcuts.plan_from(install_shortcuts.parse_args(["--no-hud"])).wanted

    assert wanted[jarvis(DESKTOP)] is True
    assert wanted[hud(DESKTOP)] is False
    assert wanted[hud(START_MENU)] is False
    assert hud(STARTUP) not in wanted  # never made, never touched


def test_each_kind_gets_its_own_icon_and_the_hud_its_flag(tmp_path: Path) -> None:
    launcher = Launcher(tmp_path / "uvw.exe", ("jarvis", "--background"))

    app = install_shortcuts.spec_for(launcher, jarvis(DESKTOP), tmp_path)
    show_hud = install_shortcuts.spec_for(launcher, hud(DESKTOP), tmp_path)

    assert (app.icon, show_hud.icon) == (ICON_PATH, HUD_ICON_PATH)
    assert app.arguments == "jarvis --background"
    assert show_hud.arguments == "jarvis --background --show-hud"


def test_uninstall_flag_is_parsed() -> None:
    assert install_shortcuts.parse_args(["--uninstall"]).uninstall


@pytest.mark.skipif(sys.platform == "win32", reason="checks the non-Windows refusal")
def test_other_platforms_are_refused(capsys: pytest.CaptureFixture[str]) -> None:
    assert install_shortcuts.main([]) == 1
    assert "Windows" in capsys.readouterr().err
