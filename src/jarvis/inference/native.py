"""Loads ``jarvis_inference`` and checks LOCAL-mode model files, with remedies.

Both LOCAL-mode providers that sit on the native module (whisper.cpp STT
and the llama.cpp LLM) need the same two preconditions: the extension is
built *with* inference enabled, and the model file exists. Checking them
here, at provider construction, turns a failure deep in the first
utterance into one :class:`~jarvis.utils.errors.ConfigError` at startup
that says exactly which command fixes it.
"""

from __future__ import annotations

from pathlib import Path
from types import ModuleType
from typing import Final

from jarvis import _native
from jarvis.utils.errors import ConfigError, NativeUnavailableError

MODULE_NAME: Final = "jarvis_inference"
DISABLED_MARKER: Final = "disabled"
"""Substring of ``IMPLEMENTATION_STATUS`` in a ``JARVIS_BUILD_INFERENCE=OFF`` build."""

BUILD_REMEDY: Final = "run 'uv run python scripts/build_native.py'"
DOWNLOAD_REMEDY: Final = "run 'uv run python scripts/download_models.py --local'"


def load_inference_module() -> ModuleType:
    """Return the real ``jarvis_inference`` module.

    Raises:
        ConfigError: If the extension is not built, fails to load, or was
            built as the stub (``JARVIS_BUILD_INFERENCE=OFF``).
    """
    try:
        module = _native.load(MODULE_NAME)
    except NativeUnavailableError as error:
        raise ConfigError(
            f"LOCAL mode needs the native {MODULE_NAME} module ({error.reason}); {BUILD_REMEDY}"
        ) from error
    status = str(getattr(module, "IMPLEMENTATION_STATUS", ""))
    if DISABLED_MARKER in status:
        raise ConfigError(
            f"{MODULE_NAME} was built without whisper.cpp/llama.cpp ({status}); "
            f"rebuild with JARVIS_BUILD_INFERENCE=ON: {BUILD_REMEDY}"
        )
    return module


def require_model_file(path: Path, *, setting: str) -> Path:
    """Return ``path`` if it exists.

    Args:
        path: The configured model file.
        setting: The environment variable that configures it, for the message.

    Raises:
        ConfigError: If the file is missing, naming the download command and
            the setting that points elsewhere.
    """
    if not path.is_file():
        raise ConfigError(
            f"LOCAL-mode model file not found: {path}. {DOWNLOAD_REMEDY.capitalize()} "
            f"or point {setting} at an existing file."
        )
    return path


# Design notes:
# - The stub build loads fine but has no classes; detecting it by its status
#   string is the contract the native module documents (module_info.hpp),
#   and it is cheaper than probing for attributes one by one.
# - Paths are checked with is_file() rather than exists() so a directory
#   named like a model gives the same clear message instead of a native
#   "failed to load" further down.
