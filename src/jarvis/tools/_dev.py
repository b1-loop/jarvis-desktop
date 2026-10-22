"""Shared plumbing for the dev-mode tools: repository paths and sandbox policies.

Private module (ignored by the registry). Every dev tool runs its fixed
command line through ``jarvis.utils.sandbox`` with a policy derived from the
user's settings, so the working-root restriction, scrubbed environment,
timeout and output cap apply exactly as they do to shell_command.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Final

from pydantic import Field

from jarvis.agent.tool import ToolInput, ToolResult
from jarvis.tools._common import truncate
from jarvis.utils.sandbox import SandboxPolicy

if TYPE_CHECKING:
    from jarvis.config import Settings
    from jarvis.utils.sandbox import CommandResult

MAX_PATH_CHARS: Final = 1024
GIT_TIMEOUT_SECONDS: Final = 30.0
_SPOKEN_ERROR_CHARS: Final = 120


class RepoInput(ToolInput):
    """Base input for tools that act on one repository or project folder."""

    repo_path: str | None = Field(
        default=None,
        max_length=MAX_PATH_CHARS,
        description=(
            "Path of the repository or project folder, absolute or relative to the "
            "configured working root. Omit to use the working root itself."
        ),
    )


def dev_policy(settings: Settings, *, timeout_seconds: float) -> SandboxPolicy:
    """The user's sandbox policy with a tool-specific timeout."""
    return replace(SandboxPolicy.from_settings(settings), timeout_seconds=timeout_seconds)


async def resolve_repo(settings: Settings, repo_path: str | None) -> Path:
    """Absolute project directory; relative paths are taken from the working root.

    Whether it lies inside the root is checked by the sandbox itself, which
    refuses the command otherwise; this only anchors relative paths.
    """
    root = settings.shell_working_root
    if root is None:
        root = await asyncio.to_thread(Path.home)
    if repo_path is None:
        return root
    candidate = Path(repo_path)
    return candidate if candidate.is_absolute() else root / candidate


def first_error_line(result: CommandResult) -> str:
    """The most telling line of a failed command, for speech."""
    for stream in (result.stderr, result.stdout):
        lines = [line.strip() for line in stream.splitlines() if line.strip()]
        if lines:
            return truncate(lines[0], _SPOKEN_ERROR_CHARS)
    return f"exit code {result.exit_code}"


def command_failure(what: str, result: CommandResult) -> ToolResult:
    """Uniform failed result for a dev command that ran but did not succeed."""
    if result.timed_out:
        spoken = f"{what} took too long, so I stopped it."
    else:
        spoken = f"{what} failed: {first_error_line(result)}"
    return ToolResult.failure(
        spoken,
        detail=f"$ {' '.join(result.argv)}\nexit code: {result.exit_code}\n"
        f"--- stdout ---\n{result.stdout.strip()}\n--- stderr ---\n{result.stderr.strip()}",
        data={"exit_code": result.exit_code, "timed_out": result.timed_out},
    )


# Design notes:
# - The commands these tools build are fixed strings plus validated values,
#   never LLM text, yet they still go through run_sandboxed: one execution
#   path means one place to audit, and the environment scrubbing keeps API
#   keys away from test suites and git hooks.
# - Paths are passed as the working directory rather than as arguments, so
#   no path ever needs quoting inside a command line.
