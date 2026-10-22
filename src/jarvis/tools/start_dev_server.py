"""Tool: start a project's dev server in the background and return its pid.

"Start the dev server". The command comes from the project (a ``dev`` or
``start`` script in package.json, Django's manage.py, ``cargo run``) or, if
the user names one, from the request; either way it goes through the
sandbox's refusal rules and is confirmed by the user, because it runs
project code for an unbounded time. Output is appended to a log file under
``<data_dir>/logs`` and the process is handed to the background-job tracker
so its exit is announced later.
"""

from __future__ import annotations

import asyncio
import json
from asyncio.subprocess import Process
from pathlib import Path
from typing import Final

from pydantic import Field

from jarvis.agent.tool import Tool, ToolCategory, ToolContext, ToolResult
from jarvis.tools._dev import RepoInput, dev_policy, resolve_repo
from jarvis.tools._jobs import BackgroundJobs
from jarvis.utils.sandbox import spawn_sandboxed

MAX_COMMAND_CHARS: Final = 300
LOG_DIRECTORY: Final = "logs"
_NPM_SCRIPTS: Final = ("dev", "start", "serve")
_LOG_NAME_FORMAT: Final = "dev-server-%Y%m%d-%H%M%S.log"


class StartDevServerInput(RepoInput):
    """Arguments for :class:`StartDevServerTool`."""

    command: str | None = Field(
        default=None,
        min_length=1,
        max_length=MAX_COMMAND_CHARS,
        description=(
            "Exact command to start the server, only if the user said one (e.g. "
            "'npm run dev'). Omit to detect it from the project files."
        ),
    )


class StartDevServerTool(Tool[StartDevServerInput]):
    """Spawns a long-running dev server after confirmation."""

    name = "start_dev_server"
    description = (
        "Start a project's development server in the background (detected from "
        "package.json dev/start scripts, manage.py or Cargo.toml, or a command the user "
        "gave). Returns the process id and the log file path. Needs confirmation."
    )
    Input = StartDevServerInput
    category = ToolCategory.DEVELOPER
    requires_confirmation = True

    def __init__(self, fallback_jobs: BackgroundJobs | None = None) -> None:
        self._fallback_jobs = fallback_jobs or BackgroundJobs()

    @property
    def fallback_jobs(self) -> BackgroundJobs:
        """Tracker used when the context has none; reaps processes nobody reports."""
        return self._fallback_jobs

    def confirmation_prompt(self, params: StartDevServerInput) -> str:
        """Quote the command when known; detection happens after the yes."""
        what = f"'{params.command}'" if params.command else "the project's dev server"
        where = f" in {params.repo_path}" if params.repo_path else ""
        return f"Do you want me to start {what}{where}?"

    async def run(self, params: StartDevServerInput, context: ToolContext) -> ToolResult:
        """Choose the command, spawn it with a log file and register it."""
        project = await resolve_repo(context.settings, params.repo_path)
        command = params.command or await asyncio.to_thread(detect_dev_command, project)
        if command is None:
            return ToolResult.failure(
                "I couldn't tell how to start a dev server there. Which command should I use?",
                detail=f"no dev/start script, manage.py or Cargo.toml in {project}",
            )
        log_path = (
            context.settings.data_dir
            / LOG_DIRECTORY
            / context.clock.now().strftime(_LOG_NAME_FORMAT)
        )
        policy = dev_policy(
            context.settings, timeout_seconds=context.settings.shell_timeout_seconds
        )
        process = await spawn_sandboxed(command, policy, cwd=project, log_path=log_path)
        job_id = self._register(context, command, process)
        promise = " I'll let you know if it stops." if job_id else ""
        return ToolResult.success(
            f"Started {command}.{promise}",
            detail=f"pid {process.pid}, output in {log_path}, job {job_id or 'unreported'}",
            data={"pid": process.pid, "command": command, "log": str(log_path), "job": job_id},
        )

    def _register(self, context: ToolContext, command: str, process: Process) -> str | None:
        """Hand the process to the context's tracker; the fallback only reaps it."""
        if context.jobs is not None:
            return context.jobs.track(command, process)
        self._fallback_jobs.track(command, process)
        return None


def detect_dev_command(project: Path) -> str | None:
    """The usual dev-server command for the project, or ``None``."""
    script = _npm_script(project / "package.json")
    if script is not None:
        return "npm start" if script == "start" else f"npm run {script}"
    if (project / "manage.py").is_file():
        return "python manage.py runserver"
    if (project / "Cargo.toml").is_file():
        return "cargo run"
    return None


def _npm_script(package_json: Path) -> str | None:
    if not package_json.is_file():
        return None
    try:
        scripts = json.loads(package_json.read_text(encoding="utf-8")).get("scripts", {})
    except (json.JSONDecodeError, AttributeError):
        return None
    if not isinstance(scripts, dict):
        return None
    return next((name for name in _NPM_SCRIPTS if name in scripts), None)


# Design notes:
# - requires_confirmation is unconditional: even a detected command runs
#   project code indefinitely, and a user-named command is LLM-relayed text.
#   The sandbox still refuses shells, operators and destructive patterns.
# - "I'll let you know if it stops" is only said when the context's job
#   tracker (which feeds LongRunningCommandTrigger) is wired in. Without it
#   the tool's own fallback tracker still awaits the process, so it is reaped
#   and never left as a zombie or an unclosed transport, but nobody announces
#   the exit and the result says "unreported".
# - The policy timeout is irrelevant to spawn_sandboxed (the process is
#   meant to outlive it) and is passed only to satisfy the policy shape.
