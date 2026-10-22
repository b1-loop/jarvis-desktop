from collections.abc import Callable
from pathlib import Path

import pytest

from jarvis.config import Settings
from jarvis.tools.git_diff_summary import GitDiffSummaryTool, parse_diff_stat
from tests.tools.local_fakes import local_context
from tests.tools.local_git import git, make_repo, requires_git

STAT = """\
 src/app.py      | 12 ++++++++----
 logo.png        | Bin 0 -> 1024 bytes
 2 files changed, 8 insertions(+), 4 deletions(-)
"""


def test_parse_diff_stat_reads_files_and_totals() -> None:
    stat = parse_diff_stat(STAT)

    assert stat.files == (("src/app.py", "12"), ("logo.png", "Bin 0 -> 1024 bytes"))
    assert (stat.files_changed, stat.insertions, stat.deletions) == (2, 8, 4)


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        (" 1 file changed, 1 insertion(+)", (1, 1, 0)),
        (" 1 file changed, 3 deletions(-)", (1, 0, 3)),
        ("", (0, 0, 0)),
    ],
)
def test_parse_diff_stat_totals_variants(line: str, expected: tuple[int, int, int]) -> None:
    stat = parse_diff_stat(line)

    assert (stat.files_changed, stat.insertions, stat.deletions) == expected


@requires_git
async def test_summarises_staged_unstaged_and_all_changes(
    make_settings: Callable[..., Settings], tmp_path: Path
) -> None:
    repo = await make_repo(tmp_path)
    (repo / "README.md").write_text("hello\nworld\n", encoding="utf-8")
    context = local_context(make_settings(shell_working_root=tmp_path))
    tool = GitDiffSummaryTool()

    unstaged = await tool.call({"repo_path": "repo", "scope": "unstaged"}, context)
    staged_before = await tool.call({"repo_path": "repo", "scope": "staged"}, context)
    await git(tmp_path, repo, "add README.md")
    everything = await tool.call({"repo_path": "repo"}, context)

    assert unstaged.spoken == "1 file changed: 1 lines added and 0 removed."
    assert staged_before.spoken == "Nothing is staged."
    assert everything.data is not None
    assert everything.data["files"] == [{"path": "README.md", "changes": "1"}]


@requires_git
async def test_clean_tree_and_failures(
    make_settings: Callable[..., Settings], tmp_path: Path
) -> None:
    await make_repo(tmp_path)
    (tmp_path / "plain").mkdir()
    context = local_context(make_settings(shell_working_root=tmp_path))

    clean = await GitDiffSummaryTool().call({"repo_path": "repo"}, context)
    broken = await GitDiffSummaryTool().call({"repo_path": "plain"}, context)

    assert clean.spoken == "No changes since the last commit."
    assert not broken.ok
    assert broken.spoken.startswith("git diff failed:")
