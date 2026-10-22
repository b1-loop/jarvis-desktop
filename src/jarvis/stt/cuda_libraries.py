"""Make the pip-installed CUDA runtime DLLs visible to CTranslate2 on Windows.

faster-whisper runs on CTranslate2, whose CUDA backend loads cuBLAS 12 and
cuDNN 9 lazily (on model load or on the first transcription). Those DLLs
normally come with a system-wide CUDA toolkit. The ``gpu`` extra installs
them as the NVIDIA wheels ``nvidia-cublas-cu12`` and ``nvidia-cudnn-cu12``
instead, into ``site-packages/nvidia/<library>/bin``, a directory Windows
does not search. :func:`register_cuda_libraries` adds those directories to
the process's DLL search path before CTranslate2 is imported, so the GPU
works after ``uv sync --extra gpu`` with nothing else installed.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from collections.abc import Callable, Iterable, MutableMapping
from pathlib import Path
from typing import Final

from jarvis.utils.logging import get_logger

log = get_logger(__name__)

NVIDIA_LIBRARY_PACKAGES: Final = ("nvidia.cublas", "nvidia.cudnn")
"""The wheels of the ``gpu`` extra, as importable (namespace) packages."""

WINDOWS_DLL_SUBDIR: Final = "bin"
_WINDOWS_PATH_SEPARATOR: Final = ";"
"""PATH entries on Windows; ``os.pathsep`` would be wrong when tests run elsewhere."""

DllDirectoryAdder = Callable[[str], object]
"""``os.add_dll_directory``; injected so tests run on any OS."""

PackageLocator = Callable[[str], Iterable[str]]
"""Directories of an importable package, empty when it is not installed."""

_registered: set[Path] = set()
_handles: list[object] = []
"""Keeps the ``add_dll_directory`` handles alive for the life of the process."""


def package_locations(name: str) -> tuple[str, ...]:
    """Where package ``name`` is installed; empty when it is not.

    ``nvidia`` is a namespace package shared by every NVIDIA wheel, so a
    missing ``nvidia`` parent and a missing sub-package both mean "absent".
    """
    try:
        spec = importlib.util.find_spec(name)
    except ModuleNotFoundError:
        return ()
    if spec is None or spec.submodule_search_locations is None:
        return ()
    return tuple(spec.submodule_search_locations)


def cuda_library_dirs(locate: PackageLocator = package_locations) -> tuple[Path, ...]:
    """The existing ``bin`` directories of the installed NVIDIA runtime wheels."""
    found: list[Path] = []
    for package in NVIDIA_LIBRARY_PACKAGES:
        for location in locate(package):
            candidate = Path(location) / WINDOWS_DLL_SUBDIR
            if candidate.is_dir():
                found.append(candidate)
    return tuple(found)


def register_cuda_libraries(
    *,
    platform: str = sys.platform,
    add_dll_directory: DllDirectoryAdder | None = None,
    environ: MutableMapping[str, str] | None = None,
    locate: PackageLocator = package_locations,
) -> tuple[Path, ...]:
    """Add the NVIDIA wheels' DLL directories to the search path, once per process.

    Both mechanisms are used: ``os.add_dll_directory`` covers libraries loaded
    with the safe search flags (CTranslate2's own loader), and prepending to
    ``PATH`` covers plain ``LoadLibrary`` calls inside the CUDA runtime.
    Does nothing off Windows, where the loader reads ``LD_LIBRARY_PATH`` only
    at process start.

    Returns:
        The directories newly registered by this call (empty if none).
    """
    if platform != "win32":
        return ()
    adder = add_dll_directory or getattr(os, "add_dll_directory", None)
    env = os.environ if environ is None else environ
    added = tuple(path for path in cuda_library_dirs(locate) if path not in _registered)
    for path in added:
        if adder is not None:
            _handles.append(adder(str(path)))
        env["PATH"] = _WINDOWS_PATH_SEPARATOR.join([str(path), env.get("PATH", "")])
        _registered.add(path)
    if added:
        log.info("stt.cuda_libraries_registered", directories=[str(path) for path in added])
    return added


def reset_registration() -> None:
    """Forget what was registered; for tests that exercise registration twice."""
    _registered.clear()
    _handles.clear()


# Design notes:
# - Wheels instead of the CUDA toolkit: the toolkit is a multi-gigabyte
#   system install with admin rights, the wheels are a per-venv dependency
#   that uv pins and removes cleanly. Only the two libraries CTranslate2
#   needs are installed; the driver already ships the CUDA driver API.
# - Located through importlib rather than a site-packages path guess, so a
#   user install, a venv and a frozen launcher all resolve the same way.
# - Missing wheels are not an error: the provider's CPU fallback (ADR-0020
#   era) still handles "cuBLAS not found", which is exactly what happens
#   when the gpu extra is not installed.
# - Registered once per process: os.add_dll_directory returns a handle
#   whose lifetime is the process here, and repeating it would only grow
#   PATH on every model reload.
