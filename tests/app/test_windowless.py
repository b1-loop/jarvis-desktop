import io
import json
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
import structlog

from jarvis.app.windowless import LogWriter, is_windowless, log_file_for, log_path_for
from jarvis.config import Settings
from jarvis.utils.logging import LogFile, close_file_logging, configure_logging


@pytest.fixture(autouse=True)
def _close_log_file() -> Iterator[None]:
    yield
    close_file_logging()
    structlog.reset_defaults()


@pytest.mark.parametrize(
    ("background", "stdout", "expected"),
    [
        (False, io.StringIO(), False),
        (True, io.StringIO(), True),
        (False, None, True),
        (True, None, True),
    ],
)
def test_windowless_when_flagged_or_without_stdout(
    background: bool, stdout: object | None, expected: bool
) -> None:
    assert is_windowless(background=background, stdout=stdout) is expected


def test_log_file_lives_under_data_dir_with_the_configured_limits(
    make_settings: Callable[..., Settings], tmp_path: Path
) -> None:
    settings = make_settings(data_dir=tmp_path, log_file_max_bytes=50_000, log_file_backups=5)

    log_file = log_file_for(settings)

    assert log_file.path == tmp_path.resolve() / "logs" / "jarvis.log"
    assert log_file.path.is_absolute()
    assert (log_file.max_bytes, log_file.backups) == (50_000, 5)
    assert log_path_for(settings) == log_file.path


def _records(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_log_writer_emits_one_event_per_completed_line(tmp_path: Path) -> None:
    path = tmp_path / "logs" / "jarvis.log"
    configure_logging("INFO", log_file=LogFile(path))
    write = LogWriter()

    write("you> what time")
    write(" is it\nJarvis> It is ")
    write("noon.\n\n")
    write("trailing")
    write.flush()

    texts = [record["text"] for record in _records(path)]
    assert texts == ["you> what time is it", "Jarvis> It is noon.", "trailing"]
