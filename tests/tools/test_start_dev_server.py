import asyncio
import json
import sys
from collections.abc import Callable
from pathlib import Path

import pytest

from jarvis.config import Settings
from jarvis.tools.start_dev_server import (
    StartDevServerInput,
    StartDevServerTool,
    detect_dev_command,
)
from tests.tools.local_fakes import RecordingJobs, local_context

PYTHON = f'"{sys.executable}"'


@pytest.fixture
def settings(make_settings: Callable[..., Settings], tmp_path: Path) -> Settings:
    return make_settings(shell_working_root=tmp_path, data_dir=tmp_path / "data")


def test_always_confirms_and_quotes_a_named_command() -> None:
    tool = StartDevServerTool()

    named = tool.confirmation_prompt(StartDevServerInput(command="npm run dev", repo_path="web"))
    detected = tool.confirmation_prompt(StartDevServerInput())

    assert tool.requires_confirmation
    assert named == "Do you want me to start 'npm run dev' in web?"
    assert detected == "Do you want me to start the project's dev server?"


@pytest.mark.parametrize(
    ("files", "expected"),
    [
        ({"package.json": json.dumps({"scripts": {"start": "x", "dev": "y"}})}, "npm run dev"),
        ({"package.json": json.dumps({"scripts": {"start": "x"}})}, "npm start"),
        ({"package.json": json.dumps({"scripts": []})}, None),
        ({"package.json": "{"}, None),
        ({"manage.py": ""}, "python manage.py runserver"),
        ({"Cargo.toml": ""}, "cargo run"),
        ({}, None),
    ],
)
def test_detects_the_usual_command(
    tmp_path: Path, files: dict[str, str], expected: str | None
) -> None:
    for name, content in files.items():
        (tmp_path / name).write_text(content, encoding="utf-8")

    assert detect_dev_command(tmp_path) == expected


async def test_spawns_logs_and_registers_the_process(settings: Settings, tmp_path: Path) -> None:
    jobs = RecordingJobs()
    command = f"{PYTHON} -c \"print('listening on 5173')\""

    result = await StartDevServerTool().call(
        {"command": command}, local_context(settings, jobs=jobs)
    )

    assert result.ok, result.detail
    assert result.spoken.endswith("I'll let you know if it stops.")
    ((tracked_command, process),) = jobs.tracked
    assert tracked_command == command
    assert await process.wait() == 0
    assert result.data is not None
    assert result.data["pid"] == process.pid
    assert result.data["job"] == "job-1"
    log = Path(str(result.data["log"]))
    assert log.parent == tmp_path / "data" / "logs"
    assert "listening on 5173" in await asyncio.to_thread(log.read_text, encoding="utf-8")


async def test_without_a_tracker_it_promises_nothing_but_still_reaps(settings: Settings) -> None:
    tool = StartDevServerTool()

    result = await tool.call({"command": f'{PYTHON} -c "pass"'}, local_context(settings))

    assert result.ok
    assert "let you know" not in result.spoken
    assert "unreported" in result.detail
    async with asyncio.timeout(10):
        await tool.fallback_jobs.join()
    assert [job.exit_code for job in await tool.fallback_jobs.pop_finished()] == [0]


async def test_refused_commands_never_spawn(settings: Settings) -> None:
    jobs = RecordingJobs()

    result = await StartDevServerTool().call(
        {"command": "bash -c 'npm run dev'"}, local_context(settings, jobs=jobs)
    )

    assert not result.ok
    assert jobs.tracked == []


async def test_asks_for_a_command_when_none_can_be_detected(settings: Settings) -> None:
    result = await StartDevServerTool().call({}, local_context(settings))

    assert not result.ok
    assert result.spoken.endswith("Which command should I use?")
