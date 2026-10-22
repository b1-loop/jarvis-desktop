"""Tool: how much has changed, per file, from ``git diff --stat``.

"How big is my change", "what have I touched". The stat output is parsed
into per-file line counts and the totals line, so the spoken answer is the
totals and the LLM still sees which files moved.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final, Literal

from pydantic import Field, JsonValue

from jarvis.agent.tool import Tool, ToolCategory, ToolContext, ToolResult
from jarvis.tools._dev import (
    GIT_TIMEOUT_SECONDS,
    RepoInput,
    command_failure,
    dev_policy,
    resolve_repo,
)
from jarvis.utils.sandbox import run_sandboxed

DiffScope = Literal["unstaged", "staged", "all"]

_SCOPE_ARGUMENTS: Final[dict[str, str]] = {"unstaged": "", "staged": " --cached", "all": " HEAD"}
_FILE_LINE: Final = re.compile(r"^\s*(?P<path>.+?)\s+\|\s+(?P<changes>\d+|Bin.*)")
_TOTALS: Final = re.compile(
    r"(?P<files>\d+) files? changed(?:, (?P<ins>\d+) insertions?\(\+\))?"
    r"(?:, (?P<dels>\d+) deletions?\(-\))?"
)


@dataclass(frozen=True, slots=True)
class DiffStat:
    """Totals and per-file counts from ``git diff --stat``."""

    files: tuple[tuple[str, str], ...]
    files_changed: int
    insertions: int
    deletions: int


class GitDiffSummaryInput(RepoInput):
    """Arguments for :class:`GitDiffSummaryTool`."""

    scope: DiffScope = Field(
        default="all",
        description=(
            "'unstaged' for changes not yet added, 'staged' for what the next commit "
            "contains, 'all' for everything since the last commit."
        ),
    )


class GitDiffSummaryTool(Tool[GitDiffSummaryInput]):
    """Summarises the size of uncommitted changes."""

    name = "git_diff_summary"
    description = (
        "Summarise uncommitted changes in a repository: files changed and lines added "
        "and removed, overall and per file. Does not return the diff itself."
    )
    Input = GitDiffSummaryInput
    category = ToolCategory.DEVELOPER

    async def run(self, params: GitDiffSummaryInput, context: ToolContext) -> ToolResult:
        """Run ``git diff --stat`` for the scope and phrase the totals."""
        repo = await resolve_repo(context.settings, params.repo_path)
        policy = dev_policy(context.settings, timeout_seconds=GIT_TIMEOUT_SECONDS)
        command = f"git diff --stat --no-color{_SCOPE_ARGUMENTS[params.scope]}"
        result = await run_sandboxed(command, policy, cwd=repo)
        if not result.ok:
            return command_failure("git diff", result)
        stat = parse_diff_stat(result.stdout)
        files: list[JsonValue] = [{"path": path, "changes": count} for path, count in stat.files]
        return ToolResult.success(
            _spoken(stat, params.scope),
            detail=result.stdout.strip() or "no changes",
            data={
                "scope": params.scope,
                "files_changed": stat.files_changed,
                "insertions": stat.insertions,
                "deletions": stat.deletions,
                "files": files,
            },
        )


def parse_diff_stat(output: str) -> DiffStat:
    """Per-file lines and the trailing totals line of ``--stat`` output."""
    files = []
    totals = None
    for line in output.splitlines():
        if (match := _TOTALS.search(line)) and "|" not in line:
            totals = match
        elif file_match := _FILE_LINE.match(line):
            files.append((file_match.group("path"), file_match.group("changes")))
    if totals is None:
        return DiffStat(files=(), files_changed=0, insertions=0, deletions=0)
    return DiffStat(
        files=tuple(files),
        files_changed=int(totals.group("files")),
        insertions=int(totals.group("ins") or 0),
        deletions=int(totals.group("dels") or 0),
    )


def _spoken(stat: DiffStat, scope: DiffScope) -> str:
    if stat.files_changed == 0:
        return {"unstaged": "No unstaged changes.", "staged": "Nothing is staged."}.get(
            scope, "No changes since the last commit."
        )
    files = "1 file" if stat.files_changed == 1 else f"{stat.files_changed} files"
    return f"{files} changed: {stat.insertions} lines added and {stat.deletions} removed."
