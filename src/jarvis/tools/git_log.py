"""Tool: the last N commits of a repository.

"What did I commit yesterday", "who changed this last". The log is asked for
in a unit-separator-delimited format so subjects containing any printable
character parse unambiguously.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

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

DEFAULT_COUNT: Final = 5
MAX_COUNT: Final = 50
FIELD_SEPARATOR: Final = "\x1f"
_FORMAT: Final = "%h%x1f%an%x1f%ar%x1f%s"
_FIELDS: Final = 4


@dataclass(frozen=True, slots=True)
class Commit:
    """One line of the log."""

    sha: str
    author: str
    when: str
    subject: str


class GitLogInput(RepoInput):
    """Arguments for :class:`GitLogTool`."""

    count: int = Field(
        default=DEFAULT_COUNT, ge=1, le=MAX_COUNT, description="How many recent commits."
    )


class GitLogTool(Tool[GitLogInput]):
    """Lists recent commits with author and relative date."""

    name = "git_log"
    description = (
        "List the most recent commits of a repository (short hash, author, relative "
        "date, subject), newest first."
    )
    Input = GitLogInput
    category = ToolCategory.DEVELOPER

    async def run(self, params: GitLogInput, context: ToolContext) -> ToolResult:
        """Run git log in the sandbox and phrase the newest commit."""
        repo = await resolve_repo(context.settings, params.repo_path)
        policy = dev_policy(context.settings, timeout_seconds=GIT_TIMEOUT_SECONDS)
        command = f"git log -n {params.count} --no-color --pretty=format:{_FORMAT}"
        result = await run_sandboxed(command, policy, cwd=repo)
        if not result.ok:
            return command_failure("git log", result)
        commits = parse_log(result.stdout)
        if not commits:
            return ToolResult.success("There are no commits yet.", data={"commits": []})
        newest = commits[0]
        rows: list[JsonValue] = [
            {"sha": c.sha, "author": c.author, "when": c.when, "subject": c.subject}
            for c in commits
        ]
        return ToolResult.success(
            f"The last commit is '{newest.subject}' by {newest.author}, {newest.when}.",
            detail="\n".join(f"{c.sha} {c.when} {c.author}: {c.subject}" for c in commits),
            data={"commits": rows},
        )


def parse_log(output: str) -> list[Commit]:
    """Split the unit-separated log lines; malformed lines are skipped."""
    commits = []
    for line in output.splitlines():
        parts = line.split(FIELD_SEPARATOR)
        if len(parts) == _FIELDS:
            commits.append(Commit(*parts))
    return commits


# Design notes:
# - ``%x1f`` (ASCII unit separator) is emitted by git itself, so the command
#   line stays free of quotes and the separator can never occur in a name
#   or subject.
# - An empty repository makes ``git log`` exit non-zero; that surfaces as a
#   failure with git's own message, which is accurate.
