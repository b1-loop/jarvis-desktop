import os
import sys
from collections.abc import Callable
from pathlib import Path

import pytest

from jarvis.config import Settings
from jarvis.utils.errors import JarvisError
from jarvis.utils.sandbox import (
    CommandNotFoundError,
    CommandRefusedError,
    SandboxPolicy,
    Verdict,
    check_command,
    normalize_binary,
    parse_command,
    resolve_working_directory,
    run_sandboxed,
    scrub_environment,
)

PYTHON = f'"{sys.executable}"'


@pytest.fixture
def policy(tmp_path: Path) -> SandboxPolicy:
    return SandboxPolicy(
        allowed_binaries=frozenset({"python", "git", "ls"}),
        working_root=tmp_path,
        timeout_seconds=10.0,
        max_output_bytes=1_000,
    )


# --- parsing and classification ---------------------------------------------


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("python", "python"),
        ("Python.EXE", "python"),
        ("C:\\Tools\\Git\\git.exe", "git"),
        ("/usr/bin/git", "git"),
        ("npm.cmd", "npm"),
        ('"python.exe"', "python"),
    ],
)
def test_normalize_binary_strips_path_case_and_extension(name: str, expected: str) -> None:
    assert normalize_binary(name) == expected


def test_parse_command_honours_quotes() -> None:
    argv = parse_command('python -c "print(1 + 1)"')

    assert argv == ("python", "-c", "print(1 + 1)")


@pytest.mark.parametrize(
    "command",
    [
        "rm -rf build",
        "rm -fr .",
        "rm -r /",
        "del /s /q C:\\Users",
        "rmdir /s /q build",
        "format C:",
        "diskpart",
        "shutdown /s /t 0",
        "reg delete HKLM\\Software\\Foo /f",
        "cmd /c dir",
        "powershell -Command Remove-Item x",
        "bash -c ls",
        "sudo rm x",
        "git push --force origin main",
        "icacls C:\\ /grant everyone:F",
        "dd if=/dev/zero of=/dev/sda",
        "git status && rm x",
        "git log | head",
        "python script.py > out.txt",
        "",
        "   ",
        'python -c "unbalanced',
    ],
)
def test_destructive_or_unparsable_commands_are_refused(
    command: str, policy: SandboxPolicy
) -> None:
    check = check_command(command, policy)

    assert check.verdict is Verdict.REFUSED, command
    assert check.reason


@pytest.mark.parametrize(
    "command",
    [
        "rm notes.txt",
        "del notes.txt",
        "git reset --hard HEAD~1",
        "git clean -fd",
        "git push --force origin feature",
        "pip uninstall requests",
        "npm uninstall left-pad",
        "taskkill /im notepad.exe",
        "mv a b",
    ],
)
def test_risky_commands_need_confirmation(command: str, policy: SandboxPolicy) -> None:
    check = check_command(command, policy)

    assert check.verdict is Verdict.NEEDS_CONFIRMATION, command


def test_unlisted_binary_needs_confirmation(policy: SandboxPolicy) -> None:
    check = check_command("cargo build", policy)

    assert check.verdict is Verdict.NEEDS_CONFIRMATION
    assert "cargo" in check.reason
    assert check.binary == "cargo"


def test_allowlisted_binary_is_allowed_regardless_of_path_and_case(policy: SandboxPolicy) -> None:
    check = check_command('"C:\\Program Files\\Git\\bin\\GIT.EXE" status', policy)

    assert check.verdict is Verdict.ALLOWED
    assert check.argv[1:] == ("status",)


def test_quoted_operators_inside_an_argument_are_plain_text(policy: SandboxPolicy) -> None:
    check = check_command('python -c "print(1 | 2); print(3 > 2)"', policy)

    assert check.verdict is Verdict.ALLOWED


# --- environment and working directory --------------------------------------


def test_scrub_environment_keeps_only_safe_keys() -> None:
    env = scrub_environment(
        {
            "PATH": "/usr/bin",
            "Path": "C:\\Windows",
            "ANTHROPIC_API_KEY": "sk-ant-secret",
            "SLACK_BOT_TOKEN": "xoxb",
            "RANDOM_THING": "x",
            "USERPROFILE": "C:\\Users\\me",
        }
    )

    assert set(env) == {"PATH", "Path", "USERPROFILE", "PYTHONUTF8", "PYTHONIOENCODING"}


def test_working_directory_defaults_to_the_root(policy: SandboxPolicy, tmp_path: Path) -> None:
    assert resolve_working_directory(None, policy) == tmp_path.resolve()


def test_working_directory_below_the_root_is_allowed(policy: SandboxPolicy, tmp_path: Path) -> None:
    child = tmp_path / "project"
    child.mkdir()

    assert resolve_working_directory(child, policy) == child.resolve()


def test_working_directory_outside_the_root_is_refused(
    policy: SandboxPolicy, tmp_path: Path
) -> None:
    with pytest.raises(CommandRefusedError, match="outside"):
        resolve_working_directory(tmp_path.parent, policy)


def test_working_directory_escaping_with_dotdot_is_refused(
    policy: SandboxPolicy, tmp_path: Path
) -> None:
    with pytest.raises(CommandRefusedError, match="outside"):
        resolve_working_directory(tmp_path / ".." / "elsewhere", policy)


def test_missing_working_directory_is_refused(policy: SandboxPolicy, tmp_path: Path) -> None:
    with pytest.raises(CommandRefusedError, match="does not exist"):
        resolve_working_directory(tmp_path / "nope", policy)


# --- execution --------------------------------------------------------------


async def test_run_sandboxed_captures_output_and_exit_code(policy: SandboxPolicy) -> None:
    result = await run_sandboxed(f'{PYTHON} -c "print(1); import sys; sys.exit(3)"', policy)

    assert result.stdout.strip() == "1"
    assert result.exit_code == 3
    assert not result.ok
    assert not result.timed_out
    assert result.duration_ms >= 0
    assert Path(result.argv[0]).name == Path(sys.executable).name


async def test_run_sandboxed_reports_success(policy: SandboxPolicy) -> None:
    result = await run_sandboxed(f"{PYTHON} -c \"print('ok')\"", policy)

    assert result.ok
    assert result.stdout.strip() == "ok"


async def test_run_sandboxed_runs_in_the_requested_directory(
    policy: SandboxPolicy, tmp_path: Path
) -> None:
    child = tmp_path / "work"
    child.mkdir()

    result = await run_sandboxed(f'{PYTHON} -c "import os; print(os.getcwd())"', policy, cwd=child)

    assert os.path.normcase(result.stdout.strip()) == os.path.normcase(str(child))


async def test_run_sandboxed_withholds_secrets_from_the_child(
    policy: SandboxPolicy, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-secret")

    result = await run_sandboxed(
        f"{PYTHON} -c \"import os; print(os.environ.get('ANTHROPIC_API_KEY'))\"", policy
    )

    assert result.stdout.strip() == "None"


async def test_run_sandboxed_kills_on_timeout(tmp_path: Path) -> None:
    policy = SandboxPolicy(frozenset({"python"}), tmp_path, timeout_seconds=0.5)

    result = await run_sandboxed(f'{PYTHON} -c "import time; time.sleep(30)"', policy)

    assert result.timed_out
    assert result.exit_code is None
    assert not result.ok
    assert result.duration_ms < 10_000


async def test_run_sandboxed_caps_output(policy: SandboxPolicy) -> None:
    result = await run_sandboxed(f"{PYTHON} -c \"print('x' * 50_000)\"", policy)

    assert result.ok
    assert result.truncated
    assert len(result.stdout.encode()) <= policy.max_output_bytes


async def test_run_sandboxed_refuses_before_spawning(policy: SandboxPolicy) -> None:
    with pytest.raises(CommandRefusedError) as info:
        await run_sandboxed("rm -rf /", policy)

    assert isinstance(info.value, JarvisError)
    assert info.value.reason == "matches a destructive pattern"


async def test_run_sandboxed_reports_a_missing_binary(policy: SandboxPolicy) -> None:
    with pytest.raises(CommandNotFoundError, match="definitely-not-a-real-binary"):
        await run_sandboxed("definitely-not-a-real-binary --version", policy)


# --- settings ---------------------------------------------------------------


def test_policy_from_settings_normalises_the_allowlist(
    make_settings: Callable[..., Settings], tmp_path: Path
) -> None:
    settings = make_settings(
        shell_allowlist=["Git.exe", "python"], shell_timeout_seconds=5, shell_working_root=tmp_path
    )

    policy = SandboxPolicy.from_settings(settings)

    assert policy.allowed_binaries == frozenset({"git", "python"})
    assert policy.timeout_seconds == 5.0
    assert policy.working_root == tmp_path


def test_policy_from_settings_defaults_the_root_to_home(
    make_settings: Callable[..., Settings],
) -> None:
    policy = SandboxPolicy.from_settings(make_settings())

    assert policy.working_root == Path.home()
    assert "git" in policy.allowed_binaries
    assert "cmd" not in policy.allowed_binaries


def test_parse_command_on_windows_keeps_backslashes() -> None:
    if os.name != "nt":
        pytest.skip("Windows path semantics")

    assert parse_command("python C:\\work\\script.py") == ("python", "C:\\work\\script.py")
