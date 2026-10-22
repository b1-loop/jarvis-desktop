"""Tool: summarise ``git status`` for one repository.

"What's the state of the repo", "anything uncommitted?". The porcelain
output is parsed into branch, ahead/behind and counts of staged, modified
and untracked files, which is what a spoken answer needs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

from pydantic import JsonValue

from jarvis.agent.tool import Tool, ToolCategory, ToolContext, ToolResult
from jarvis.tools._dev import (
    GIT_TIMEOUT_SECONDS,
    RepoInput,
    command_failure,
    dev_policy,
    resolve_repo,
)
from jarvis.utils.sandbox import run_sandboxed

GIT_STATUS_COMMAND: Final = "git status --porcelain=v1 --branch"
_HEADER_PREFIX: Final = "## "
_NO_COMMITS_PREFIX: Final = "No commits yet on "
_UPSTREAM_SEPARATOR: Final = "..."
_TRACKING_START: Final = " ["
_PATH_OFFSET: Final = 3
"""Entries are "XY path": two status letters and a space precede the path."""
_AHEAD: Final = re.compile(r"ahead (\d+)")
_BEHIND: Final = re.compile(r"behind (\d+)")
_UNTRACKED: Final = "??"


@dataclass
class StatusSummary:
    """Parsed ``git status --porcelain --branch``."""

    branch: str = "unknown"
    ahead: int = 0
    behind: int = 0
    staged: list[str] = field(default_factory=list)
    modified: list[str] = field(default_factory=list)
    untracked: list[str] = field(default_factory=list)

    @property
    def clean(self) -> bool:
        """True when nothing is staged, modified or untracked."""
        return not (self.staged or self.modified or self.untracked)


def parse_status(output: str) -> StatusSummary:
    """Parse porcelain v1 output with the ``## branch`` header line."""
    summary = StatusSummary()
    for line in output.splitlines():
        if line.startswith(_HEADER_PREFIX):
            _parse_branch(line.removeprefix(_HEADER_PREFIX), summary)
        elif line.startswith(_UNTRACKED):
            summary.untracked.append(line[_PATH_OFFSET:])
        elif len(line) > _PATH_OFFSET:
            if line[0] != " ":
                summary.staged.append(line[_PATH_OFFSET:])
            if line[1] != " ":
                summary.modified.append(line[_PATH_OFFSET:])
    return summary


def _parse_branch(header: str, summary: StatusSummary) -> None:
    """``main...origin/main [ahead 1, behind 2]`` -> branch and counts."""
    head, _, tracking = header.partition(_TRACKING_START)
    head = head.removeprefix(_NO_COMMITS_PREFIX)
    summary.branch = head.partition(_UPSTREAM_SEPARATOR)[0]
    if ahead := _AHEAD.search(tracking):
        summary.ahead = int(ahead.group(1))
    if behind := _BEHIND.search(tracking):
        summary.behind = int(behind.group(1))


class GitStatusTool(Tool[RepoInput]):
    """Reports branch and working-tree state."""

    name = "git_status"
    description = (
        "Summarise git status for a repository: current branch, commits ahead/behind "
        "the upstream, and the staged, modified and untracked files."
    )
    Input = RepoInput
    category = ToolCategory.DEVELOPER

    async def run(self, params: RepoInput, context: ToolContext) -> ToolResult:
        """Run git status in the sandbox and phrase the parsed result."""
        repo = await resolve_repo(context.settings, params.repo_path)
        policy = dev_policy(context.settings, timeout_seconds=GIT_TIMEOUT_SECONDS)
        result = await run_sandboxed(GIT_STATUS_COMMAND, policy, cwd=repo)
        if not result.ok:
            return command_failure("git status", result)
        summary = parse_status(result.stdout)
        return ToolResult.success(
            spoken_status(summary), detail=result.stdout.strip(), data=_data(summary, repo)
        )


def spoken_status(summary: StatusSummary) -> str:
    """One sentence: branch, then what is pending, then sync state."""
    counts = [
        f"{len(items)} {label}"
        for items, label in (
            (summary.staged, "staged"),
            (summary.modified, "modified"),
            (summary.untracked, "untracked"),
        )
        if items
    ]
    state = "working tree clean" if summary.clean else ", ".join(counts)
    sync = []
    if summary.ahead:
        sync.append(f"{summary.ahead} ahead")
    if summary.behind:
        sync.append(f"{summary.behind} behind")
    tail = f"; {' and '.join(sync)} of upstream" if sync else ""
    return f"On {summary.branch}: {state}{tail}."


def _data(summary: StatusSummary, repo: Path) -> dict[str, JsonValue]:
    return {
        "repo": str(repo),
        "branch": summary.branch,
        "ahead": summary.ahead,
        "behind": summary.behind,
        "staged": list(summary.staged),
        "modified": list(summary.modified),
        "untracked": list(summary.untracked),
    }


# Design notes:
# - Porcelain v1 is git's stable machine format; the human format changes
#   wording between versions and locales.
# - A file both staged and changed again counts in both lists, matching
#   what ``git status`` shows in its two sections.
