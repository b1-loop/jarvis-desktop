from collections.abc import Callable
from pathlib import Path

import pytest

from jarvis.config import Settings
from jarvis.tools.git_push import GitPushInput, GitPushTool
from tests.tools.local_fakes import local_context
from tests.tools.local_git import git, make_repo, requires_git


def test_always_confirms_and_says_what_goes_where() -> None:
    tool = GitPushTool()

    default = tool.confirmation_prompt(GitPushInput())
    named = tool.confirmation_prompt(GitPushInput(repo_path="jarvis", branch="main"))

    assert tool.requires_confirmation
    assert default == "Do you want me to push the current branch to origin?"
    assert named == "Do you want me to push main of jarvis to origin?"


@pytest.mark.parametrize(
    "raw", [{"remote": "--force"}, {"branch": "-f"}, {"branch": "main;rm"}, {"force": True}]
)
async def test_option_smuggling_is_rejected_by_validation(
    make_settings: Callable[..., Settings], raw: dict[str, object]
) -> None:
    result = await GitPushTool().call(raw, local_context(make_settings()))

    assert not result.ok
    assert "missing something" in result.spoken


@requires_git
async def test_pushes_the_current_branch_to_a_local_remote(
    make_settings: Callable[..., Settings], tmp_path: Path
) -> None:
    repo = await make_repo(tmp_path)
    await git(tmp_path, tmp_path, "init -q --bare remote.git")
    await git(tmp_path, repo, f"remote add origin {(tmp_path / 'remote.git').as_posix()}")
    context = local_context(make_settings(shell_working_root=tmp_path))

    result = await GitPushTool().call({"repo_path": "repo", "set_upstream": True}, context)

    assert result.ok, result.detail
    assert result.spoken == "Pushed to origin."
    branches = await git(tmp_path, tmp_path / "remote.git", "branch --list main")
    assert "main" in branches


@requires_git
async def test_a_failed_push_reports_gits_reason(
    make_settings: Callable[..., Settings], tmp_path: Path
) -> None:
    await make_repo(tmp_path)
    context = local_context(make_settings(shell_working_root=tmp_path))

    result = await GitPushTool().call({"repo_path": "repo", "remote": "nowhere"}, context)

    assert not result.ok
    assert result.spoken.startswith("The push failed:")
