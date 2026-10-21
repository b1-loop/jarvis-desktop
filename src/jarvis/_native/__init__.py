"""Loader shim for the compiled pybind11 extension modules.

The ``.pyd`` files produced by ``scripts/build_native.py`` land in this
package directory (git-ignored). Python code never imports them directly;
it goes through :func:`load` so that a missing or broken build turns into one
well-explained :class:`~jarvis.utils.errors.NativeUnavailableError` instead
of a bare ``ImportError`` deep inside the audio pipeline.

See ``docs/adr/0002-decoupled-native-build.md`` for why the native build is
separate from the Python package.
"""

from __future__ import annotations

import importlib
from types import ModuleType
from typing import Final

from jarvis.utils.errors import NativeUnavailableError

MODULE_NAMES: Final[tuple[str, ...]] = (
    "jarvis_audio",
    "jarvis_os",
    "jarvis_inference",
    "jarvis_overlay",
)
"""Every extension module the native build can produce, in build order."""


def load(name: str) -> ModuleType:
    """Import and return the extension module called ``name``.

    Args:
        name: One of :data:`MODULE_NAMES`.

    Raises:
        NativeUnavailableError: If ``name`` is not a Jarvis module, the
            ``.pyd`` has not been built, or it fails to load (for example
            because ``onnxruntime.dll`` is missing from ``PATH``).
    """
    if name not in MODULE_NAMES:
        raise NativeUnavailableError(name, "not a Jarvis native module")
    try:
        return importlib.import_module(f"{__name__}.{name}")
    except ImportError as exc:
        raise NativeUnavailableError(name, str(exc)) from exc


def available_modules() -> tuple[str, ...]:
    """Return the names of the modules that actually load on this machine.

    Loads each module to answer, because a ``.pyd`` that exists but cannot
    resolve its DLL dependencies is not available in any useful sense.
    """
    return tuple(name for name in MODULE_NAMES if _loads(name))


def _loads(name: str) -> bool:
    try:
        load(name)
    except NativeUnavailableError:
        return False
    return True


# Design notes:
# - Modules are imported as submodules of this package (``jarvis._native.
#   jarvis_audio``) so no sys.path manipulation is needed and the ``.pyd``
#   files cannot collide with anything else on the path.
# - available_modules() is the single probe CI uses to prove the build and the
#   import path agree.
