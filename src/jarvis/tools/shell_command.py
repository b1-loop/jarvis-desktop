"""Tool: run a command line through the sandbox and report what it printed.

The policy in ``jarvis.utils.sandbox`` decides what happens: allowlisted
programs run straight away, risky ones (deletions, unknown binaries) need
the user's spoken confirmation, and destructive ones are refused. The tool
never spawns a shell and never bypasses that policy.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

from pydantic import Field, JsonValue

from jarvis.agent.tool import Tool, ToolCategory, ToolContext, ToolInput, ToolResult
from jarvis.utils.sandbox import (
    CommandResult,
    SandboxError,
    SandboxPolicy,
    Verdict,
    check_command,
    run_sandboxed,
)

MAX_COMMAND_CHARS: Final = 2000
MAX_DETAIL_CHARS: Final = 4000
"""Per-stream cap on what the LLM sees; the tail is kept because errors come last."""
_SPOKEN_OUTPUT_CHARS: Final = 80


class ShellCommandInput(ToolInput):
    """Arguments for :class:`ShellCommandTool`."""

    command: str = Field(
        min_length=1,
        max_length=MAX_COMMAND_CHARS,
        description=(
            "The command line to run, for example 'git status' or 'uv run pytest -q'. "
            "One program and its arguments; pipes, redirects and && are not allowed."
        ),
    )
    working_directory: str | None = Field(
        default=None,
        description=(
            "Directory to run in. Must be inside the configured working root; "
            "defaults to the root itself."
        ),
    )


class ShellCommandTool(Tool[ShellCommandInput]):
    """Runs developer commands under the sandbox policy."""

    name = "shell_command"
    description = (
        "Run a command-line program such as git, python, uv, npm or cargo and return "
        "its output. Destructive commands are refused; risky ones ask the user first."
    )
    Input = ShellCommandInput
    category = ToolCategory.DEVELOPER
    keywords = ("terminal", "console", "command", "script")

    def needs_confirmation(self, params: ShellCommandInput, context: ToolContext) -> bool:
        """Ask when the policy says so: unknown binary or a risky-but-allowed pattern."""
        policy = SandboxPolicy.from_settings(context.settings)
        return check_command(params.command, policy).verdict is Verdict.NEEDS_CONFIRMATION

    def confirmation_prompt(self, params: ShellCommandInput) -> str:
        """Quote the exact command so the user confirms what will actually run."""
        return f"Do you want me to run the command: {params.command}?"

    async def run(self, params: ShellCommandInput, context: ToolContext) -> ToolResult:
        """Execute under the policy built from settings and summarise the outcome."""
        policy = SandboxPolicy.from_settings(context.settings)
        cwd = Path(params.working_directory) if params.working_directory else None
        try:
            result = await run_sandboxed(params.command, policy, cwd=cwd)
        except SandboxError as error:
            return ToolResult.failure(f"I can't run that: {error}.", detail=str(error))
        return _to_tool_result(result, policy)


def _to_tool_result(result: CommandResult, policy: SandboxPolicy) -> ToolResult:
    return ToolResult(
        ok=result.ok,
        spoken=_spoken(result, policy),
        detail=_detail(result),
        data=_data(result),
    )


def _spoken(result: CommandResult, policy: SandboxPolicy) -> str:
    if result.timed_out:
        return f"The command timed out after {policy.timeout_seconds:.0f} seconds, so I stopped it."
    if not result.ok:
        return f"The command failed with exit code {result.exit_code}."
    first_line = result.stdout.strip().splitlines()[:1]
    if not first_line:
        return "Done, with no output."
    return f"Done. It printed: {_truncate(first_line[0], _SPOKEN_OUTPUT_CHARS)}"


def _detail(result: CommandResult) -> str:
    lines = [
        f"$ {' '.join(result.argv)}",
        f"exit code: {result.exit_code}" + (" (killed on timeout)" if result.timed_out else ""),
        f"duration: {result.duration_ms:.0f} ms",
    ]
    if result.truncated:
        lines.append("note: output was truncated by the sandbox")
    lines.append("--- stdout ---")
    lines.append(_tail(result.stdout))
    lines.append("--- stderr ---")
    lines.append(_tail(result.stderr))
    return "\n".join(lines)


def _data(result: CommandResult) -> dict[str, JsonValue]:
    return {
        "argv": list(result.argv),
        "exit_code": result.exit_code,
        "timed_out": result.timed_out,
        "truncated": result.truncated,
        "duration_ms": round(result.duration_ms),
        "stdout": _tail(result.stdout),
        "stderr": _tail(result.stderr),
    }


def _tail(text: str) -> str:
    text = text.strip()
    if len(text) <= MAX_DETAIL_CHARS:
        return text
    return "…" + text[-MAX_DETAIL_CHARS:]


def _truncate(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


# Design notes:
# - ``requires_confirmation`` stays False at class level because the right
#   answer depends on the command; ``needs_confirmation`` consults the same
#   ``check_command`` the sandbox applies at run time, so the user is asked
#   about exactly what will execute and never asked about ``git status``.
# - Refusals surface as failed results with the sandbox's reason, which the
#   LLM relays, rather than exceptions the agent loop would have to special-case.
# - The detail keeps the tail of each stream: compiler and test failures put
#   the useful part last.
