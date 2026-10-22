import json
import sys
from collections.abc import Callable
from pathlib import Path

import pytest

from jarvis.config import Settings
from jarvis.tools.run_tests import RunnerCommand, RunTestsTool, detect_test_command, summary_line
from tests.tools.local_fakes import local_context

PYTHON = f'"{sys.executable}"'


def project(tmp_path: Path, files: dict[str, str]) -> Path:
    for name, content in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return tmp_path


@pytest.mark.parametrize(
    ("files", "expected"),
    [
        ({"Cargo.toml": ""}, RunnerCommand("cargo", "cargo test")),
        (
            {"package.json": json.dumps({"scripts": {"test": "jest"}})},
            RunnerCommand("npm", "npm test"),
        ),
        ({"pyproject.toml": "", "uv.lock": ""}, RunnerCommand("pytest", "uv run pytest -q")),
        ({"tests/test_x.py": ""}, RunnerCommand("pytest", "python -m pytest -q")),
        ({"package.json": json.dumps({"scripts": {"build": "tsc"}})}, None),
        ({"package.json": "{not json"}, None),
        ({"README.md": ""}, None),
    ],
)
def test_detects_the_runner_from_project_files(
    tmp_path: Path, files: dict[str, str], expected: RunnerCommand | None
) -> None:
    assert detect_test_command(project(tmp_path, files)) == expected


@pytest.mark.parametrize(
    ("output", "expected"),
    [
        ("collected 3\n...\n===== 3 passed in 0.12s =====\n", "3 passed in 0.12s"),
        (
            "test result: ok. 4 passed; 0 failed\n\nDoc-tests\n",
            "test result: ok. 4 passed; 0 failed",
        ),
        ("all good\n", "all good"),
        ("", "no output"),
    ],
)
def test_summary_line(output: str, expected: str) -> None:
    assert summary_line(output) == expected


def fake_runner(code: str) -> Callable[[Path], RunnerCommand]:
    return lambda _: RunnerCommand("pytest", f'{PYTHON} -c "{code}"')


async def test_runs_the_detected_command_and_speaks_the_summary(
    make_settings: Callable[..., Settings], tmp_path: Path
) -> None:
    tool = RunTestsTool(detector=fake_runner("print('5 passed in 0.1s')"))
    context = local_context(make_settings(shell_working_root=tmp_path))

    result = await tool.call({}, context)

    assert result.ok, result.detail
    assert result.spoken == "Tests passed: 5 passed in 0.1s"
    assert result.data is not None
    assert result.data["runner"] == "pytest"


async def test_a_failing_suite_is_a_failed_result(
    make_settings: Callable[..., Settings], tmp_path: Path
) -> None:
    code = "import sys; print('1 failed'); sys.exit(1)"
    tool = RunTestsTool(detector=fake_runner(code))

    result = await tool.call({}, local_context(make_settings(shell_working_root=tmp_path)))

    assert not result.ok
    assert result.spoken == "The pytest run failed: 1 failed"


async def test_an_unknown_project_asks_for_help(
    make_settings: Callable[..., Settings], tmp_path: Path
) -> None:
    result = await RunTestsTool().call(
        {}, local_context(make_settings(shell_working_root=tmp_path))
    )

    assert not result.ok
    assert result.spoken == "I couldn't tell how to run the tests in that project."


async def test_timeout_bounds_are_validated(make_settings: Callable[..., Settings]) -> None:
    result = await RunTestsTool().call({"timeout_seconds": 5}, local_context(make_settings()))

    assert not result.ok
