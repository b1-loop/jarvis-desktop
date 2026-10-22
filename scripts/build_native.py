"""Build the native extension modules into ``src/jarvis/_native/``.

Wraps the CMake presets in ``native/CMakePresets.json`` so a contributor
needs one command and does not have to know where pybind11, the Python
interpreter or the MSVC environment live. On Windows the MSVC developer
environment is entered automatically through ``VsDevCmd.bat`` when ``cl`` is
not already on ``PATH``.

Usage::

    uv run python scripts/build_native.py                 # windows-release
    uv run python scripts/build_native.py --preset windows-debug --clean
    uv run python scripts/build_native.py --skip-tests

Why this is a script and not the package build backend: see
``docs/adr/0002-decoupled-native-build.md``.
"""

from __future__ import annotations

import argparse
import os
import shutil
import stat
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import pybind11

ROOT = Path(__file__).resolve().parent.parent
NATIVE_DIR = ROOT / "native"
OUTPUT_DIR = ROOT / "src" / "jarvis" / "_native"
VC_TOOLS_COMPONENT = "Microsoft.VisualStudio.Component.VC.Tools.x86.x64"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--preset", default="windows-release", help="CMake preset name")
    parser.add_argument("--clean", action="store_true", help="delete the build tree first")
    parser.add_argument("--skip-tests", action="store_true", help="build only, do not run ctest")
    parser.add_argument(
        "--tidy", action="store_true", help="run clang-tidy over the native sources after building"
    )
    return parser.parse_args(argv)


def build_dir(preset: str) -> Path:
    """Build tree for ``preset``; mirrors ``binaryDir`` in CMakePresets.json."""
    return NATIVE_DIR / "build" / preset


def native_sources(suffix: str = ".cpp") -> list[Path]:
    """Our own C++ files under native/, excluding build trees and submodules.

    Only translation units are handed to clang-tidy; headers are covered
    through ``HeaderFilterRegex`` in ``.clang-tidy`` because they have no
    entry of their own in the compile database.
    """
    excluded = {"build", "third_party"}
    return sorted(
        path
        for path in NATIVE_DIR.rglob(f"*{suffix}")
        if excluded.isdisjoint(path.relative_to(NATIVE_DIR).parts)
    )


def pybind11_cmake_dir() -> Path:
    """Locate the CMake config shipped inside the ``pybind11`` wheel.

    Using the pip-installed pybind11 (pinned by ``uv.lock``) keeps the Python
    and C++ sides on the same version without a network fetch at configure
    time.
    """
    return Path(pybind11.get_cmake_dir())


def vswhere_path() -> Path:
    """Location of ``vswhere.exe``, which every Visual Studio installer provides."""
    program_files_x86 = os.environ.get("PROGRAMFILES(X86)", r"C:\Program Files (x86)")
    return Path(program_files_x86) / "Microsoft Visual Studio" / "Installer" / "vswhere.exe"


def find_vsdevcmd() -> Path | None:
    """Return the newest Visual Studio ``VsDevCmd.bat`` that ships the C++ tools."""
    vswhere = vswhere_path()
    if not vswhere.exists():
        return None
    query = [
        str(vswhere),
        "-latest",
        "-prerelease",
        "-products",
        "*",
        "-requires",
        VC_TOOLS_COMPONENT,
        "-property",
        "installationPath",
    ]
    install_path = subprocess.run(query, capture_output=True, text=True, check=False).stdout
    if not install_path.strip():
        return None
    return Path(install_path.strip()) / "Common7" / "Tools" / "VsDevCmd.bat"


def msvc_environment_needed() -> bool:
    """True when we are on Windows and the compiler is not already reachable."""
    return sys.platform == "win32" and shutil.which("cl") is None


class CommandRunner:
    """Runs commands inside ``native/``, optionally wrapped in the MSVC environment."""

    def __init__(self, vsdevcmd: Path | None) -> None:
        self._vsdevcmd = vsdevcmd

    def run(self, args: list[str]) -> None:
        """Run ``args`` and raise ``CalledProcessError`` on a non-zero exit."""
        print("+", subprocess.list2cmdline(args))
        if self._vsdevcmd is None:
            subprocess.run(args, check=True, cwd=NATIVE_DIR)
            return
        # Passed as one string: a list would make subprocess escape the inner
        # quotes with backslashes, which cmd.exe does not understand. /s makes
        # cmd strip exactly the outer pair of quotes. VsDevCmd expects vswhere
        # on PATH and prints a harmless but confusing error otherwise.
        inner = (
            f'set "PATH=%PATH%;{vswhere_path().parent}" && '
            f'call "{self._vsdevcmd}" -arch=x64 -no_logo && {subprocess.list2cmdline(args)}'
        )
        subprocess.run(f'cmd.exe /d /s /c "{inner}"', check=True, cwd=NATIVE_DIR)


def clear_readonly_and_retry(
    function: Callable[[str], object], path: str, error: BaseException
) -> None:
    """``shutil.rmtree`` ``onexc`` hook that makes read-only files deletable.

    Git marks the objects in a clone's ``.git/objects/pack`` read-only, and
    on Windows ``os.unlink`` refuses to delete a read-only file. FetchContent
    clones (Catch2, Dear ImGui) live inside the build tree, so ``--clean``
    hits exactly that. Anything other than a permission error is re-raised
    untouched, and a retry that still fails propagates as well.
    """
    if not isinstance(error, PermissionError):
        raise error
    Path(path).chmod(stat.S_IWRITE)
    function(path)


def remove_tree(path: Path) -> None:
    """Delete ``path`` recursively, including read-only files."""
    shutil.rmtree(path, onexc=clear_readonly_and_retry)


def configure(runner: CommandRunner, preset: str, *, clean: bool) -> None:
    """Run the CMake configure step for ``preset``."""
    if clean and build_dir(preset).exists():
        remove_tree(build_dir(preset))
    runner.run(
        [
            "cmake",
            "--preset",
            preset,
            f"-DPython_EXECUTABLE={sys.executable}",
            f"-Dpybind11_DIR={pybind11_cmake_dir()}",
            f"-DJARVIS_PYTHON_NATIVE_DIR={OUTPUT_DIR}",
        ]
    )


def run_clang_tidy(runner: CommandRunner, preset: str) -> None:
    """Lint every native source against the compile database of ``preset``.

    Runs inside the MSVC environment so clang-tidy sees the same include
    paths as the compiler. Unused-argument diagnostics are silenced because
    pybind11 adds MSVC-only flags (``/MP``, ``/GL``) that clang-cl ignores.
    """
    runner.run(
        [
            "clang-tidy",
            "-p",
            str(build_dir(preset)),
            "--extra-arg=-Wno-unused-command-line-argument",
            *map(str, native_sources()),
        ]
    )


def main(argv: list[str] | None = None) -> int:
    """Configure, build, test and optionally lint; return a process exit code."""
    args = parse_args(argv)
    vsdevcmd = find_vsdevcmd() if msvc_environment_needed() else None
    if msvc_environment_needed() and vsdevcmd is None:
        print("error: no Visual Studio with the C++ workload found", file=sys.stderr)
        return 1
    runner = CommandRunner(vsdevcmd)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    try:
        configure(runner, args.preset, clean=args.clean)
        runner.run(["cmake", "--build", "--preset", args.preset])
        if not args.skip_tests:
            runner.run(["ctest", "--preset", args.preset, "--output-on-failure"])
        if args.tidy:
            run_clang_tidy(runner, args.preset)
    except subprocess.CalledProcessError as exc:
        print(f"error: step failed with exit code {exc.returncode}", file=sys.stderr)
        return exc.returncode
    print(f"native modules written to {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())


# Design notes:
# - The MSVC environment is entered per command through cmd.exe because
#   VsDevCmd.bat mutates environment variables that cannot be imported into a
#   running Python process portably.
# - The preset owns generator, build type and warnings; this script only adds
#   the three machine-specific paths (interpreter, pybind11, output directory).
