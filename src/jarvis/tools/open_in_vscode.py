"""Tool: open a folder or file in Visual Studio Code.

"Open the Jarvis repo in VS Code". Runs VS Code's ``code`` command-line
launcher through the sandbox with the target's folder as the working
directory, so at most a bare file name (never a path) appears in the
command line.
"""

from __future__ import annotations

import asyncio
from typing import Final

from jarvis.agent.tool import Tool, ToolCategory, ToolContext, ToolResult
from jarvis.tools._dev import RepoInput, command_failure, dev_policy, resolve_repo
from jarvis.utils.sandbox import CommandNotFoundError, run_sandboxed

LAUNCH_TIMEOUT_SECONDS: Final = 20.0
"""``code`` hands off to the running editor and exits; this only bounds a hang."""
_UNSAFE_NAME_CHARACTERS: Final = frozenset('"')


class OpenInVSCodeTool(Tool[RepoInput]):
    """Opens a project folder or a single file in VS Code."""

    name = "open_in_vscode"
    description = (
        "Open a folder (typically a repository) or a single file in Visual Studio Code "
        "using its 'code' command. Pass the folder or file as repo_path."
    )
    Input = RepoInput
    category = ToolCategory.DEVELOPER

    async def run(self, params: RepoInput, context: ToolContext) -> ToolResult:
        """Resolve the target, then run ``code`` from its folder."""
        target = await resolve_repo(context.settings, params.repo_path)
        is_file = await asyncio.to_thread(target.is_file)
        if not is_file and not await asyncio.to_thread(target.is_dir):
            return ToolResult.failure(f"I can't find {target.name}.", detail=f"missing: {target}")
        if _UNSAFE_NAME_CHARACTERS & set(target.name):
            return ToolResult.failure("That file name has a quote in it, which I can't pass on.")
        folder, argument = (target.parent, f'"{target.name}"') if is_file else (target, ".")
        policy = dev_policy(context.settings, timeout_seconds=LAUNCH_TIMEOUT_SECONDS)
        try:
            result = await run_sandboxed(f"code {argument}", policy, cwd=folder)
        except CommandNotFoundError as error:
            return ToolResult.failure(
                "VS Code's 'code' command isn't on the PATH.", detail=str(error)
            )
        if not result.ok:
            return command_failure("Opening VS Code", result)
        return ToolResult.success(
            f"Opening {target.name} in VS Code.", data={"path": str(target), "is_file": is_file}
        )


# Design notes:
# - Not a Platform.launch: ``code`` is a developer CLI that already lives on
#   PATH, and going through the sandbox keeps the working-root restriction
#   (only projects under the configured root) for every dev-mode tool.
# - A missing ``code`` launcher gets its own spoken message because the fix
#   (VS Code's "Install 'code' command in PATH") is something the user can do.
