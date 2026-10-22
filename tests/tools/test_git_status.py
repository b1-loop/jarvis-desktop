from collections.abc import Callable
from pathlib import Path

import pytest

from jarvis.config import Settings
from jarvis.tools.git_status import GitStatusTool, StatusSummary, parse_status, spoken_status
from tests.tools.local_fakes import local_context
from tests.tools.local_git import make_repo, requires_git

PORCELAIN = """\
## feature/x...origin/feature/x [ahead 2, behind 1]
M  staged.py
 M changed.py
MM both.py
?? new.txt
"""


def test_parse_status_reads_branch_tracking_and_entries() -> None:
    summary = parse_status(PORCELAIN)

    assert summary.branch == "feature/x"
    assert (summary.ahead, summary.behind) == (2, 1)
    assert summary.staged == ["staged.py", "both.py"]
    assert summary.modified == ["changed.py", "both.py"]
    assert summary.untracked == ["new.txt"]


@pytest.mark.parametrize(
    ("header", "branch"),
    [
        ("## main", "main"),
        ("## No commits yet on dev", "dev"),
        ("## release-1.2...up/r", "release-1.2"),
    ],
)
def test_parse_status_branch_header_variants(header: str, branch: str) -> None:
    assert parse_status(header).branch == branch


def test_spoken_status_phrasing() -> None:
    assert spoken_status(StatusSummary(branch="main")) == "On main: working tree clean."
    assert (
        spoken_status(parse_status(PORCELAIN))
        == "On feature/x: 2 staged, 2 modified, 1 untracked; 2 ahead and 1 behind of upstream."
    )


@requires_git
async def test_reports_a_real_repository(
    make_settings: Callable[..., Settings], tmp_path: Path
) -> None:
    repo = await make_repo(tmp_path)
    (repo / "README.md").write_text("changed\n", encoding="utf-8")
    (repo / "new.txt").write_text("x", encoding="utf-8")
    context = local_context(make_settings(shell_working_root=tmp_path))

    result = await GitStatusTool().call({"repo_path": "repo"}, context)

    assert result.ok, result.detail
    assert result.spoken == "On main: 1 modified, 1 untracked."
    assert result.data is not None
    assert result.data["modified"] == ["README.md"]


@requires_git
async def test_a_folder_that_is_not_a_repository_fails_with_gits_message(
    make_settings: Callable[..., Settings], tmp_path: Path
) -> None:
    (tmp_path / "plain").mkdir()
    context = local_context(make_settings(shell_working_root=tmp_path))

    result = await GitStatusTool().call({"repo_path": str(tmp_path / "plain")}, context)

    assert not result.ok
    assert result.spoken.startswith("git status failed:")
    assert "not a git repository" in result.detail.lower()


async def test_a_path_outside_the_working_root_is_refused(
    make_settings: Callable[..., Settings], tmp_path: Path
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    context = local_context(make_settings(shell_working_root=root))

    result = await GitStatusTool().call({"repo_path": str(tmp_path)}, context)

    assert not result.ok
    assert "outside" in result.detail
