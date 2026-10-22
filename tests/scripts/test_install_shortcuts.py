import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

from jarvis.app.shortcuts import ShortcutLocation

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


def test_defaults_are_desktop_and_start_menu_without_autostart() -> None:
    plan = install_shortcuts.plan_from(install_shortcuts.parse_args([]))

    assert plan.wanted == {
        ShortcutLocation.DESKTOP: True,
        ShortcutLocation.START_MENU: True,
        ShortcutLocation.STARTUP: False,
    }


def test_every_location_can_be_switched() -> None:
    args = install_shortcuts.parse_args(["--autostart", "--no-desktop", "--no-start-menu"])

    assert install_shortcuts.plan_from(args).wanted == {
        ShortcutLocation.DESKTOP: False,
        ShortcutLocation.START_MENU: False,
        ShortcutLocation.STARTUP: True,
    }
    assert not args.uninstall


def test_uninstall_flag_is_parsed() -> None:
    assert install_shortcuts.parse_args(["--uninstall"]).uninstall


@pytest.mark.skipif(sys.platform == "win32", reason="checks the non-Windows refusal")
def test_other_platforms_are_refused(capsys: pytest.CaptureFixture[str]) -> None:
    assert install_shortcuts.main([]) == 1
    assert "Windows" in capsys.readouterr().err
