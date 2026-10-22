"""Tool: push the current branch to a remote, after the user confirmed.

"Push to GitHub". Pushing publishes work to an external service and cannot
be taken back quietly, so the tool always confirms, never forces, and only
accepts remote and branch names that cannot smuggle in an option.
"""

from __future__ import annotations

from typing import Final

from pydantic import Field

from jarvis.agent.tool import Tool, ToolCategory, ToolContext, ToolResult
from jarvis.tools._common import truncate
from jarvis.tools._dev import RepoInput, command_failure, dev_policy, resolve_repo
from jarvis.utils.sandbox import run_sandboxed

PUSH_TIMEOUT_SECONDS: Final = 120.0
"""Pushes cross the network and may run server-side hooks; 30 s is too tight."""
REF_NAME_PATTERN: Final = r"^[A-Za-z0-9_][A-Za-z0-9._/-]{0,199}$"
"""A safe subset of git ref names; a leading ``-`` would be read as an option."""
MAX_REF_CHARS: Final = 200
_SPOKEN_OUTPUT_CHARS: Final = 120


class GitPushInput(RepoInput):
    """Arguments for :class:`GitPushTool`."""

    remote: str = Field(
        default="origin",
        max_length=MAX_REF_CHARS,
        pattern=REF_NAME_PATTERN,
        description="Remote to push to; almost always 'origin'.",
    )
    branch: str | None = Field(
        default=None,
        max_length=MAX_REF_CHARS,
        pattern=REF_NAME_PATTERN,
        description="Branch to push. Omit to push the currently checked-out branch.",
    )
    set_upstream: bool = Field(
        default=False,
        description="Set the remote branch as upstream (-u); use for a brand-new branch.",
    )


class GitPushTool(Tool[GitPushInput]):
    """Publishes commits to a remote; never with --force."""

    name = "git_push"
    description = (
        "Push commits of a repository to a remote (default 'origin'), the current branch "
        "unless one is named. Never force-pushes. Requires the user's confirmation."
    )
    Input = GitPushInput
    category = ToolCategory.DEVELOPER
    requires_confirmation = True

    def confirmation_prompt(self, params: GitPushInput) -> str:
        """Say what goes where; the repository path is in the dashboard detail."""
        branch = params.branch or "the current branch"
        where = f" of {params.repo_path}" if params.repo_path else ""
        return f"Do you want me to push {branch}{where} to {params.remote}?"

    async def run(self, params: GitPushInput, context: ToolContext) -> ToolResult:
        """Push through the sandbox; git reports progress on stderr."""
        repo = await resolve_repo(context.settings, params.repo_path)
        policy = dev_policy(context.settings, timeout_seconds=PUSH_TIMEOUT_SECONDS)
        upstream = " -u" if params.set_upstream else ""
        command = f"git push{upstream} {params.remote} {params.branch or 'HEAD'}"
        result = await run_sandboxed(command, policy, cwd=repo)
        if not result.ok:
            return command_failure("The push", result)
        lines = [line.strip() for line in result.stderr.splitlines() if line.strip()]
        summary = truncate(lines[-1], _SPOKEN_OUTPUT_CHARS) if lines else "done"
        return ToolResult.success(
            f"Pushed to {params.remote}.",
            detail=result.stderr.strip() or result.stdout.strip(),
            data={"remote": params.remote, "branch": params.branch, "summary": summary},
        )


# Design notes:
# - ``HEAD`` as the refspec pushes the checked-out branch to the same name
#   on the remote without relying on push.default configuration.
# - No ``force`` field exists, so no phrasing can produce a force-push; the
#   sandbox denylist refuses one to main/master anyway. Rewriting history
#   stays a deliberate shell_command the user confirms word for word.
