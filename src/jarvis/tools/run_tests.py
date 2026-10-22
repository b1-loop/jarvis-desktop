"""Tool: detect a project's test runner and run its tests.

"Run the tests". The runner is chosen from the files in the project folder
(Cargo.toml, a package.json test script, Python project markers), so the
LLM never composes the command line. Suites can be slow, so the timeout is
much higher than for other dev commands, and the result is spoken as the
runner's own summary line.
"""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from pydantic import Field

from jarvis.agent.tool import Tool, ToolCategory, ToolContext, ToolResult
from jarvis.tools._common import truncate
from jarvis.tools._dev import RepoInput, command_failure, dev_policy, resolve_repo
from jarvis.utils.sandbox import run_sandboxed

DEFAULT_TIMEOUT_SECONDS: Final = 900
MIN_TIMEOUT_SECONDS: Final = 10
MAX_TIMEOUT_SECONDS: Final = 3600
_PYTHON_MARKERS: Final = ("pyproject.toml", "pytest.ini", "setup.cfg", "tox.ini", "setup.py")
_SUMMARY_LINE: Final = re.compile(r"\b(passed|failed|test result:|Tests:|error)\b", re.IGNORECASE)
_SPOKEN_SUMMARY_CHARS: Final = 120
_DETAIL_TAIL_CHARS: Final = 4000


@dataclass(frozen=True, slots=True)
class RunnerCommand:
    """A detected test runner.

    Attributes:
        runner: Short name for speech ("pytest", "npm", "cargo").
        command: The fixed command line to run in the project folder.
    """

    runner: str
    command: str


RunnerDetector = Callable[[Path], RunnerCommand | None]
"""Blocking project inspection; called in a worker thread."""


def detect_test_command(project: Path) -> RunnerCommand | None:
    """Pick the test runner from the project's marker files, or ``None``."""
    if (project / "Cargo.toml").is_file():
        return RunnerCommand("cargo", "cargo test")
    if _has_npm_test_script(project / "package.json"):
        return RunnerCommand("npm", "npm test")
    if (
        any((project / marker).is_file() for marker in _PYTHON_MARKERS)
        or (project / "tests").is_dir()
    ):
        if (project / "uv.lock").is_file():
            return RunnerCommand("pytest", "uv run pytest -q")
        return RunnerCommand("pytest", "python -m pytest -q")
    return None


def _has_npm_test_script(package_json: Path) -> bool:
    if not package_json.is_file():
        return False
    try:
        scripts = json.loads(package_json.read_text(encoding="utf-8")).get("scripts", {})
    except (json.JSONDecodeError, AttributeError):
        return False
    return isinstance(scripts, dict) and "test" in scripts


class RunTestsInput(RepoInput):
    """Arguments for :class:`RunTestsTool`."""

    timeout_seconds: int = Field(
        default=DEFAULT_TIMEOUT_SECONDS,
        ge=MIN_TIMEOUT_SECONDS,
        le=MAX_TIMEOUT_SECONDS,
        description="Stop the suite after this many seconds.",
    )


class RunTestsTool(Tool[RunTestsInput]):
    """Runs the project's own test suite and reports the outcome."""

    name = "run_tests"
    description = (
        "Run a project's test suite. The runner is detected automatically: cargo test "
        "for Rust, npm test when package.json has a test script, pytest for Python. "
        "Returns the runner's summary line and the tail of its output."
    )
    Input = RunTestsInput
    category = ToolCategory.DEVELOPER
    keywords = ("test", "tester", "testerna")

    def __init__(self, detector: RunnerDetector = detect_test_command) -> None:
        self._detect = detector

    async def run(self, params: RunTestsInput, context: ToolContext) -> ToolResult:
        """Detect, run under the sandbox with the long timeout, and summarise."""
        project = await resolve_repo(context.settings, params.repo_path)
        detected = await asyncio.to_thread(self._detect, project)
        if detected is None:
            return ToolResult.failure(
                "I couldn't tell how to run the tests in that project.",
                detail=f"no Cargo.toml, package.json test script or Python markers in {project}",
            )
        policy = dev_policy(context.settings, timeout_seconds=params.timeout_seconds)
        result = await run_sandboxed(detected.command, policy, cwd=project)
        if not result.ok:
            return command_failure(f"The {detected.runner} run", result)
        summary = summary_line(result.stdout + "\n" + result.stderr)
        return ToolResult.success(
            f"Tests passed: {summary}",
            detail=f"$ {detected.command}\n{result.stdout.strip()[-_DETAIL_TAIL_CHARS:]}",
            data={"runner": detected.runner, "command": detected.command, "summary": summary},
        )


def summary_line(output: str) -> str:
    """The last line that looks like a runner's summary, else the last line."""
    lines = [line.strip().strip("=").strip() for line in output.splitlines() if line.strip()]
    for line in reversed(lines):
        if _SUMMARY_LINE.search(line):
            return truncate(line, _SPOKEN_SUMMARY_CHARS)
    return truncate(lines[-1], _SPOKEN_SUMMARY_CHARS) if lines else "no output"


# Design notes:
# - No confirmation: running the user's own test suite in their own project
#   is the request. It goes through the sandbox like every dev command, so
#   the suite never sees API keys from Jarvis's environment.
# - The detector is injectable so tests can substitute a harmless command;
#   the real detector is a pure function of the folder's files and is tested
#   on generated project layouts.
# - A failing suite is reported through command_failure, whose spoken line
#   is the first error line; the full output tail is in detail for the LLM.
