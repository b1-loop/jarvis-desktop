from collections.abc import Callable
from pathlib import Path

import pytest

from jarvis.config import Settings
from jarvis.tools import open_in_vscode
from jarvis.tools.open_in_vscode import OpenInVSCodeTool
from jarvis.utils.sandbox import CommandNotFoundError, CommandResult, SandboxPolicy
from tests.tools.local_fakes import local_context


class RecordingRunner:
    """Stands in for run_sandboxed so no editor is launched."""

    def __init__(self, exit_code: int = 0, missing: bool = False) -> None:
        self.calls: list[tuple[str, Path | None]] = []
        self.exit_code = exit_code
        self.missing = missing

    async def __call__(
        self, command: str, policy: SandboxPolicy, *, cwd: Path | None = None
    ) -> CommandResult:
        self.calls.append((command, cwd))
        if self.missing:
            raise CommandNotFoundError("code")
        return CommandResult(("code",), self.exit_code, "", "boom", 1.0, False, False)


@pytest.fixture
def settings(make_settings: Callable[..., Settings], tmp_path: Path) -> Settings:
    (tmp_path / "repo").mkdir()
    (tmp_path / "repo" / "main.py").write_text("", encoding="utf-8")
    return make_settings(shell_working_root=tmp_path)


async def test_opens_a_folder_from_inside_it(
    settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = RecordingRunner()
    monkeypatch.setattr(open_in_vscode, "run_sandboxed", runner)

    result = await OpenInVSCodeTool().call({"repo_path": "repo"}, local_context(settings))

    assert result.ok
    assert result.spoken == "Opening repo in VS Code."
    assert runner.calls == [("code .", tmp_path / "repo")]


async def test_opens_a_file_by_bare_name_from_its_folder(
    settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = RecordingRunner()
    monkeypatch.setattr(open_in_vscode, "run_sandboxed", runner)

    result = await OpenInVSCodeTool().call({"repo_path": "repo/main.py"}, local_context(settings))

    assert result.data == {"path": str(tmp_path / "repo" / "main.py"), "is_file": True}
    assert runner.calls == [('code "main.py"', tmp_path / "repo")]


async def test_reports_a_missing_target_or_launcher(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(open_in_vscode, "run_sandboxed", RecordingRunner(missing=True))
    context = local_context(settings)

    missing_target = await OpenInVSCodeTool().call({"repo_path": "nope"}, context)
    missing_code = await OpenInVSCodeTool().call({"repo_path": "repo"}, context)

    assert missing_target.spoken == "I can't find nope."
    assert missing_code.spoken == "VS Code's 'code' command isn't on the PATH."


async def test_reports_a_failed_launch(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(open_in_vscode, "run_sandboxed", RecordingRunner(exit_code=1))

    result = await OpenInVSCodeTool().call({"repo_path": "repo"}, local_context(settings))

    assert not result.ok
    assert result.spoken == "Opening VS Code failed: boom"


async def test_refuses_file_names_with_quotes(settings: Settings, tmp_path: Path) -> None:
    odd = tmp_path / "repo" / 'say "hi".txt'
    try:
        odd.write_text("", encoding="utf-8")
    except OSError:
        pytest.skip("this filesystem does not allow quotes in names")

    result = await OpenInVSCodeTool().call({"repo_path": str(odd)}, local_context(settings))

    assert not result.ok
