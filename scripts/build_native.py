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
    uv run python scripts/build_native.py --tidy               # build, test, then clang-tidy
    uv run python scripts/build_native.py --tidy-only --jobs 4 # configure + clang-tidy only
    uv run python scripts/build_native.py --tidy-only --tidy-changed-since origin/main

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
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pybind11

ROOT = Path(__file__).resolve().parent.parent
NATIVE_DIR = ROOT / "native"
OUTPUT_DIR = ROOT / "src" / "jarvis" / "_native"
VC_TOOLS_COMPONENT = "Microsoft.VisualStudio.Component.VC.Tools.x86.x64"
# Outside native/, the files whose change means every translation unit is
# linted again rather than only the changed ones.
FULL_TIDY_TRIGGERS = frozenset({".clang-tidy", "scripts/build_native.py"})


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--preset", default="windows-release", help="CMake preset name")
    parser.add_argument("--clean", action="store_true", help="delete the build tree first")
    parser.add_argument("--skip-tests", action="store_true", help="build only, do not run ctest")
    add_tidy_arguments(parser)
    return parser.parse_args(argv)


def add_tidy_arguments(parser: argparse.ArgumentParser) -> None:
    """Options that select what clang-tidy lints and how it is spread out."""
    parser.add_argument(
        "--tidy", action="store_true", help="run clang-tidy over the native sources after building"
    )
    parser.add_argument(
        "--tidy-only",
        action="store_true",
        help="configure and run clang-tidy without building or testing (CI's lint job)",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=os.cpu_count() or 1,
        help="clang-tidy processes to run in parallel (default: CPU count)",
    )
    parser.add_argument(
        "--tidy-changed-since",
        metavar="REF",
        help="lint only .cpp files changed since REF; everything if a header, CMake file "
        "or .clang-tidy changed",
    )
    parser.add_argument(
        "--tidy-shard",
        type=parse_shard,
        default=(1, 1),
        metavar="I/N",
        help="lint only the I-th of N interleaved slices of the files (CI matrix)",
    )


def parse_shard(spec: str) -> tuple[int, int]:
    """Parse ``I/N`` (1-based) for ``--tidy-shard``."""
    index_text, _, count_text = spec.partition("/")
    try:
        index, count = int(index_text), int(count_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected I/N, got {spec!r}") from exc
    if not 1 <= index <= count:
        raise argparse.ArgumentTypeError(f"shard {index} is outside 1..{count}")
    return index, count


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
        subprocess.run(self._command(args), check=True, cwd=NATIVE_DIR)

    def run_captured(self, args: list[str]) -> subprocess.CompletedProcess[str]:
        """Run ``args`` with stdout and stderr captured together; never raises.

        Used for commands that run concurrently, so their output can be
        printed as one block per command instead of interleaved lines.
        """
        return subprocess.run(
            self._command(args),
            check=False,
            cwd=NATIVE_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

    def _command(self, args: list[str]) -> list[str] | str:
        """``args`` as given, or wrapped in ``cmd.exe`` after ``VsDevCmd.bat``."""
        if self._vsdevcmd is None:
            return args
        # Passed as one string: a list would make subprocess escape the inner
        # quotes with backslashes, which cmd.exe does not understand. /s makes
        # cmd strip exactly the outer pair of quotes. VsDevCmd expects vswhere
        # on PATH and prints a harmless but confusing error otherwise.
        inner = (
            f'set "PATH=%PATH%;{vswhere_path().parent}" && '
            f'call "{self._vsdevcmd}" -arch=x64 -no_logo && {subprocess.list2cmdline(args)}'
        )
        return f'cmd.exe /d /s /c "{inner}"'


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


def clang_tidy_command(preset: str, source: Path) -> list[str]:
    """clang-tidy invocation for one translation unit of ``preset``.

    Unused-argument diagnostics are silenced because pybind11 adds MSVC-only
    flags (``/MP``, ``/GL``) that clang-cl ignores. ``--quiet`` drops only
    the "N warnings generated" summaries of suppressed third-party warnings;
    every diagnostic from our code is still printed and still an error.
    """
    return [
        "clang-tidy",
        "-p",
        str(build_dir(preset)),
        "--quiet",
        "--extra-arg=-Wno-unused-command-line-argument",
        str(source),
    ]


def changed_files(ref: str) -> list[str]:
    """Repository-relative paths that differ between ``ref`` and ``HEAD``."""
    result = subprocess.run(
        ["git", "diff", "--name-only", ref, "HEAD"],
        capture_output=True,
        text=True,
        check=True,
        cwd=ROOT,
    )
    return result.stdout.splitlines()


def needs_full_tidy(changed: list[str]) -> bool:
    """True when a change can alter diagnostics in files it does not touch.

    A header is linted through every translation unit that includes it, and
    a CMake file, the preset or ``.clang-tidy`` changes flags or checks for
    all of them; only an edit confined to our ``.cpp`` files is local. A
    submodule bump (``native/third_party/llama.cpp`` ends in ``.cpp`` too)
    changes headers every wrapper includes.
    """
    return any(name in FULL_TIDY_TRIGGERS or not is_local_change(name) for name in changed)


def is_local_change(name: str) -> bool:
    """True for a path whose change can only affect its own diagnostics."""
    if not name.startswith("native/"):
        return True
    return name.endswith(".cpp") and not name.startswith("native/third_party/")


def tidy_sources(changed: list[str] | None, shard: tuple[int, int]) -> list[Path]:
    """The translation units this run lints.

    ``changed`` is ``None`` for a full run. ``shard`` ``(i, n)`` keeps every
    n-th file starting at the i-th, so n CI jobs split the work evenly.
    """
    sources = native_sources()
    if changed is not None and not needs_full_tidy(changed):
        touched = {(ROOT / name).resolve() for name in changed}
        sources = [source for source in sources if source.resolve() in touched]
    index, count = shard
    return sources[index - 1 :: count]


def run_clang_tidy(runner: CommandRunner, preset: str, sources: list[Path], jobs: int) -> None:
    """Lint ``sources`` against the compile database of ``preset``.

    Runs inside the MSVC environment so clang-tidy sees the same include
    paths as the compiler. One process per translation unit, ``jobs`` at a
    time: a single clang-tidy process is single-threaded and took 42
    minutes for the whole tree in CI. Each file's output is printed as one
    block, and any failing file fails the step after all files ran.
    """
    print(f"+ clang-tidy over {len(sources)} files, {jobs} at a time")
    failures: list[int] = []
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = {
            pool.submit(runner.run_captured, clang_tidy_command(preset, source)): source
            for source in sources
        }
        for future in as_completed(futures):
            result = future.result()
            status = "ok" if result.returncode == 0 else f"FAILED ({result.returncode})"
            print(f"clang-tidy {futures[future].relative_to(NATIVE_DIR)}: {status}")
            print(result.stdout, end="", flush=True)
            if result.returncode != 0:
                failures.append(result.returncode)
    if failures:
        raise subprocess.CalledProcessError(failures[0], "clang-tidy")


def build_and_test(runner: CommandRunner, preset: str, *, skip_tests: bool) -> None:
    """Build every target of ``preset`` and run the Catch2 suite through ctest."""
    runner.run(["cmake", "--build", "--preset", preset])
    if not skip_tests:
        runner.run(["ctest", "--preset", preset, "--output-on-failure"])
    print(f"native modules written to {OUTPUT_DIR}")


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
        if not args.tidy_only:
            build_and_test(runner, args.preset, skip_tests=args.skip_tests)
        if args.tidy or args.tidy_only:
            changed = changed_files(args.tidy_changed_since) if args.tidy_changed_since else None
            sources = tidy_sources(changed, args.tidy_shard)
            run_clang_tidy(runner, args.preset, sources, args.jobs)
    except subprocess.CalledProcessError as exc:
        print(f"error: step failed with exit code {exc.returncode}", file=sys.stderr)
        return exc.returncode
    return 0


if __name__ == "__main__":
    sys.exit(main())


# Design notes:
# - The MSVC environment is entered per command through cmd.exe because
#   VsDevCmd.bat mutates environment variables that cannot be imported into a
#   running Python process portably.
# - The preset owns generator, build type and warnings; this script only adds
#   the three machine-specific paths (interpreter, pybind11, output directory).
# - --tidy-only exists so CI can lint in a job parallel to the build: clang-tidy
#   needs only the compile database and the sources FetchContent puts in place
#   at configure time, never a compiled object.
# - --tidy-changed-since narrows a PR's lint to the .cpp files it touched, and
#   falls back to every file when a header, CMake file or .clang-tidy changed,
#   since those alter diagnostics elsewhere. Pushes to main always lint all.
