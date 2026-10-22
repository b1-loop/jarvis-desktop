import importlib.util
import stat
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "build_native.py"


def load_build_native() -> ModuleType:
    # scripts/ is not a package, so the module is loaded from its file path.
    spec = importlib.util.spec_from_file_location("build_native", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("build_native", module)
    spec.loader.exec_module(module)
    return module


build_native = load_build_native()


def make_readonly_tree(root: Path) -> Path:
    # Mirrors a FetchContent clone: .git/objects/pack/*.pack is read-only.
    pack = root / "_deps" / "catch2-src" / ".git" / "objects" / "pack" / "pack-1.pack"
    pack.parent.mkdir(parents=True)
    pack.write_bytes(b"packed objects")
    pack.chmod(stat.S_IREAD)
    return pack


def test_remove_tree_deletes_read_only_files(tmp_path: Path) -> None:
    tree = tmp_path / "build"
    pack = make_readonly_tree(tree)
    assert not pack.stat().st_mode & stat.S_IWRITE

    build_native.remove_tree(tree)

    assert not tree.exists()


def test_onexc_hook_reraises_errors_that_are_not_permission_errors(tmp_path: Path) -> None:
    missing = tmp_path / "missing"
    error = FileNotFoundError(str(missing))

    def unreachable(_path: str) -> None:
        pytest.fail("the hook must not retry a non-permission error")

    with pytest.raises(FileNotFoundError):
        build_native.clear_readonly_and_retry(unreachable, str(missing), error)


def test_onexc_hook_clears_read_only_bit_before_retrying(tmp_path: Path) -> None:
    target = tmp_path / "locked.txt"
    target.write_text("x")
    target.chmod(stat.S_IREAD)
    retried: list[str] = []

    def retry(path: str) -> None:
        retried.append(path)
        Path(path).unlink()

    build_native.clear_readonly_and_retry(retry, str(target), PermissionError(str(target)))

    assert retried == [str(target)]
    assert not target.exists()


class FakeRunner:
    def __init__(self, failing: set[str]) -> None:
        self.failing = failing
        self.commands: list[list[str]] = []

    def run_captured(self, args: list[str]) -> subprocess.CompletedProcess[str]:
        self.commands.append(args)
        code = 1 if Path(args[-1]).name in self.failing else 0
        return subprocess.CompletedProcess(args, code, stdout=f"output of {args[-1]}\n")


def test_clang_tidy_runs_one_process_per_source(monkeypatch: pytest.MonkeyPatch) -> None:
    sources = [build_native.NATIVE_DIR / "a.cpp", build_native.NATIVE_DIR / "b.cpp"]
    monkeypatch.setattr(build_native, "native_sources", lambda: sources)
    runner = FakeRunner(failing=set())

    build_native.run_clang_tidy(runner, "ci-windows", jobs=2)

    assert sorted(command[-1] for command in runner.commands) == [str(s) for s in sources]
    assert all("--quiet" in command for command in runner.commands)


def test_clang_tidy_lints_every_file_before_reporting_a_failure(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    sources = [build_native.NATIVE_DIR / name for name in ("a.cpp", "b.cpp", "c.cpp")]
    monkeypatch.setattr(build_native, "native_sources", lambda: sources)
    runner = FakeRunner(failing={"a.cpp"})

    with pytest.raises(subprocess.CalledProcessError):
        build_native.run_clang_tidy(runner, "ci-windows", jobs=1)

    assert len(runner.commands) == len(sources)
    output = capsys.readouterr().out
    assert "a.cpp: FAILED (1)" in output
    assert "output of" in output


def test_tidy_only_flag_parses() -> None:
    args = build_native.parse_args(["--tidy-only", "--jobs", "3"])

    assert args.tidy_only
    assert args.jobs == 3
