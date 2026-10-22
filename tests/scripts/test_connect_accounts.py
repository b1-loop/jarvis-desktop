import importlib.util
from pathlib import Path
from types import ModuleType

from jarvis.integrations.connect import cli

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "connect_accounts.py"


def load_script() -> ModuleType:
    # scripts/ is not a package, so the module is loaded from its file path.
    spec = importlib.util.spec_from_file_location("connect_accounts", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_script_is_a_thin_wrapper_over_the_connect_cli() -> None:
    assert load_script().main is cli.main
