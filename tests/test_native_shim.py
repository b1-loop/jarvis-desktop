import importlib

import pytest

from jarvis import _native
from jarvis.utils.errors import NativeUnavailableError


def test_module_names_are_the_four_native_components() -> None:
    assert _native.MODULE_NAMES == (
        "jarvis_audio",
        "jarvis_os",
        "jarvis_inference",
        "jarvis_overlay",
    )


def test_loading_an_unknown_name_is_refused() -> None:
    with pytest.raises(NativeUnavailableError) as excinfo:
        _native.load("jarvis_toaster")

    assert excinfo.value.module == "jarvis_toaster"
    assert "not a Jarvis native module" in excinfo.value.reason


def test_import_failure_is_wrapped_with_the_cause(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken_import(name: str) -> None:
        raise ImportError(f"DLL load failed for {name}")

    monkeypatch.setattr(importlib, "import_module", broken_import)

    with pytest.raises(NativeUnavailableError) as excinfo:
        _native.load("jarvis_audio")

    assert excinfo.value.module == "jarvis_audio"
    assert "DLL load failed" in excinfo.value.reason
    assert isinstance(excinfo.value.__cause__, ImportError)


def test_error_message_tells_the_user_how_to_fix_it() -> None:
    error = NativeUnavailableError("jarvis_os", "missing")

    assert "scripts/build_native.py" in str(error)
    assert "--text-only" in str(error)


def test_available_modules_is_a_subset_of_module_names() -> None:
    available = _native.available_modules()

    assert set(available) <= set(_native.MODULE_NAMES)


def test_available_modules_skips_modules_that_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    def only_audio_loads(name: str) -> object:
        if name.endswith("jarvis_audio"):
            return object()
        raise ImportError(name)

    monkeypatch.setattr(importlib, "import_module", only_audio_loads)

    assert _native.available_modules() == ("jarvis_audio",)


@pytest.mark.native
def test_built_modules_report_version_and_status() -> None:
    if not _native.available_modules():
        pytest.skip("native modules not built on this machine")
    for name in _native.available_modules():
        module = _native.load(name)
        assert module.__version__.count(".") == 2
        assert isinstance(module.IMPLEMENTATION_STATUS, str)
