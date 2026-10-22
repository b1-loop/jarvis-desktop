import sys
from collections.abc import Callable
from pathlib import Path

import pytest

from jarvis.agent.confirmation import DECLINED_SPOKEN, ConfirmationPolicy, NotNeeded, Required
from jarvis.agent.registry import ToolRegistry
from jarvis.agent.tool import ToolContext
from jarvis.config import Settings
from jarvis.tools.shell_command import ShellCommandInput, ShellCommandTool
from tests.agent.fakes import FakeConfirmation, make_context

PYTHON = f'"{sys.executable}"'


@pytest.fixture
def settings(make_settings: Callable[..., Settings], tmp_path: Path) -> Settings:
    return make_settings(shell_working_root=tmp_path, shell_timeout_seconds=5)


@pytest.fixture
def context(settings: Settings) -> ToolContext:
    return make_context(settings)


def test_is_discovered() -> None:
    tool = ToolRegistry.discover().get("shell_command")

    assert isinstance(tool, ShellCommandTool)
    assert tool.requires_confirmation is False  # decided per command instead


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ("git status", False),
        ("python -V", False),
        ("cargo build", False),  # on the default allowlist
        ("ping localhost", True),  # not on the allowlist
        ("rm notes.txt", True),  # risky pattern
        ("rm -rf /", False),  # refused outright: nothing to confirm, run() fails
    ],
)
def test_needs_confirmation_follows_the_sandbox_verdict(
    context: ToolContext, command: str, expected: bool
) -> None:
    tool = ShellCommandTool()

    assert tool.needs_confirmation(ShellCommandInput(command=command), context) is expected


def test_allowlist_from_settings_drives_the_decision(
    make_settings: Callable[..., Settings],
) -> None:
    context = make_context(make_settings(shell_allowlist=["ping"]))
    tool = ShellCommandTool()

    assert tool.needs_confirmation(ShellCommandInput(command="ping x"), context) is False
    assert tool.needs_confirmation(ShellCommandInput(command="git status"), context) is True


def test_policy_integration_decides_from_the_command(context: ToolContext) -> None:
    policy = ConfirmationPolicy()
    tool = ShellCommandTool()

    safe = policy.decide(tool, ShellCommandInput(command="git status"), context)
    risky = policy.decide(tool, ShellCommandInput(command="rm notes.txt"), context)

    assert isinstance(safe, NotNeeded)
    assert risky == Required(
        "Do you want me to run the command: rm notes.txt?", "the tool requires confirmation"
    )


async def test_runs_an_allowlisted_command_and_reports_output(context: ToolContext) -> None:
    result = await ShellCommandTool().call({"command": f'{PYTHON} -c "print(42)"'}, context)

    assert result.ok
    assert result.spoken == "Done. It printed: 42"
    assert "exit code: 0" in result.detail
    assert "--- stdout ---\n42" in result.detail
    assert result.data is not None
    assert result.data["exit_code"] == 0
    assert result.data["stdout"] == "42"


async def test_silent_success(context: ToolContext) -> None:
    result = await ShellCommandTool().call({"command": f'{PYTHON} -c "pass"'}, context)

    assert result.spoken == "Done, with no output."


async def test_failure_reports_the_exit_code_and_stderr(context: ToolContext) -> None:
    command = f"{PYTHON} -c \"import sys; print('bad', file=sys.stderr); sys.exit(2)\""

    result = await ShellCommandTool().call({"command": command}, context)

    assert not result.ok
    assert result.spoken == "The command failed with exit code 2."
    assert "--- stderr ---\nbad" in result.detail


async def test_timeout_is_reported(make_settings: Callable[..., Settings], tmp_path: Path) -> None:
    context = make_context(make_settings(shell_working_root=tmp_path, shell_timeout_seconds=1))

    result = await ShellCommandTool().call(
        {"command": f'{PYTHON} -c "import time; time.sleep(30)"'}, context
    )

    assert not result.ok
    assert result.spoken == "The command timed out after 1 seconds, so I stopped it."
    assert "killed on timeout" in result.detail
    assert result.data is not None
    assert result.data["timed_out"] is True


async def test_runs_in_the_requested_directory(context: ToolContext, tmp_path: Path) -> None:
    (tmp_path / "sub").mkdir()

    result = await ShellCommandTool().call(
        {
            "command": f'{PYTHON} -c "import os; print(os.path.basename(os.getcwd()))"',
            "working_directory": str(tmp_path / "sub"),
        },
        context,
    )

    assert result.ok
    assert result.data is not None
    assert result.data["stdout"] == "sub"


async def test_directory_outside_the_root_is_refused(context: ToolContext, tmp_path: Path) -> None:
    result = await ShellCommandTool().call(
        {"command": "git status", "working_directory": str(tmp_path.parent)}, context
    )

    assert not result.ok
    assert result.spoken.startswith("I can't run that:")
    assert "outside" in result.detail


async def test_destructive_command_is_refused_even_when_confirmed(settings: Settings) -> None:
    asker = FakeConfirmation(answers=[True])
    context = make_context(settings, confirmation=asker)

    result = await ConfirmationPolicy().execute_with_policy(
        ShellCommandTool(), {"command": "rm -rf /"}, context
    )

    assert not result.ok
    assert "destructive" in result.detail
    assert asker.prompts == []  # refused outright, never asked


async def test_unknown_binary_is_not_run_when_declined(settings: Settings) -> None:
    asker = FakeConfirmation(answers=[False])
    context = make_context(settings, confirmation=asker)

    result = await ConfirmationPolicy().execute_with_policy(
        ShellCommandTool(), {"command": "definitely-not-a-binary --version"}, context
    )

    assert result.spoken == DECLINED_SPOKEN
    assert asker.prompts == [
        "Do you want me to run the command: definitely-not-a-binary --version?"
    ]


async def test_missing_binary_is_a_failed_result(context: ToolContext) -> None:
    result = await ShellCommandTool().call({"command": "git-but-not-really status"}, context)

    assert not result.ok
    assert "command not found" in result.detail


async def test_long_output_is_tailed_in_the_detail(context: ToolContext) -> None:
    command = f"{PYTHON} -c \"print('a' * 3000); print('END')\""

    result = await ShellCommandTool().call({"command": command}, context)

    assert result.ok
    assert result.detail.endswith("END\n--- stderr ---\n")
    assert result.spoken == "Done. It printed: " + "a" * 79 + "…"


async def test_invalid_arguments_are_rejected(context: ToolContext) -> None:
    result = await ShellCommandTool().call({"command": ""}, context)

    assert not result.ok
    assert "command: String should have at least 1 character" in result.detail
