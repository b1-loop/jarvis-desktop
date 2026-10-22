from collections.abc import Callable
from pathlib import Path

from jarvis.config import Settings
from jarvis.tools.git_log import Commit, GitLogTool, parse_log
from tests.tools.local_fakes import local_context
from tests.tools.local_git import git, make_repo, requires_git


def test_parse_log_splits_on_the_unit_separator() -> None:
    output = "abc123\x1fAda\x1f2 hours ago\x1ffix: a | b: c\nbroken line\n"

    assert parse_log(output) == [Commit("abc123", "Ada", "2 hours ago", "fix: a | b: c")]


@requires_git
async def test_lists_recent_commits_newest_first(
    make_settings: Callable[..., Settings], tmp_path: Path
) -> None:
    repo = await make_repo(tmp_path)
    (repo / "a.txt").write_text("a", encoding="utf-8")
    await git(tmp_path, repo, "add a.txt")
    await git(tmp_path, repo, "commit -q -m second")
    context = local_context(make_settings(shell_working_root=tmp_path))

    result = await GitLogTool().call({"repo_path": "repo", "count": 1}, context)

    assert result.ok, result.detail
    assert result.spoken.startswith("The last commit is 'second' by Jarvis, ")
    assert result.data is not None
    commits = result.data["commits"]
    assert isinstance(commits, list)
    assert len(commits) == 1


@requires_git
async def test_an_empty_repository_fails_with_gits_reason(
    make_settings: Callable[..., Settings], tmp_path: Path
) -> None:
    repo = tmp_path / "empty"
    repo.mkdir()
    await git(tmp_path, repo, "init -q")
    context = local_context(make_settings(shell_working_root=tmp_path))

    result = await GitLogTool().call({"repo_path": "empty"}, context)

    assert not result.ok
    assert result.spoken.startswith("git log failed:")
