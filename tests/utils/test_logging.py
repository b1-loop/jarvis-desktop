import io
import json
import logging
from collections.abc import Iterator
from pathlib import Path

import pytest
import structlog

from jarvis.utils.logging import (
    LogFile,
    close_file_logging,
    configure_logging,
    get_logger,
    redact_secrets,
)

created_at_import = get_logger("tests.logging.module_level")
"""Like every ``log = get_logger(__name__)``: made before any configuration."""


@pytest.fixture(autouse=True)
def _restore_structlog_defaults() -> Iterator[None]:
    """Each test configures structlog with pytest's captured stderr; undo that."""
    yield
    structlog.reset_defaults()


@pytest.mark.parametrize(
    "key",
    ["api_key", "ANTHROPIC_API_KEY", "refresh_token", "client_secret", "password"],
)
def test_credential_like_keys_are_redacted(key: str) -> None:
    event = redact_secrets(None, "info", {"event": "x", key: "hunter2"})

    assert event[key] == "***"


def test_ordinary_keys_are_untouched() -> None:
    event = redact_secrets(None, "info", {"event": "x", "mode": "cloud", "latency_ms": 12})

    assert event == {"event": "x", "mode": "cloud", "latency_ms": 12}


def test_json_output_contains_level_timestamp_and_logger_name(
    capsys: pytest.CaptureFixture[str],
) -> None:
    configure_logging("INFO", json_output=True)

    get_logger("tests.logging").info("hello", api_key="secret", mode="cloud")

    record = json.loads(capsys.readouterr().err.strip())
    assert record["event"] == "hello"
    assert record["level"] == "info"
    assert record["logger"] == "tests.logging"
    assert record["mode"] == "cloud"
    assert record["api_key"] == "***"
    assert "timestamp" in record


def test_messages_below_the_configured_level_are_dropped(
    capsys: pytest.CaptureFixture[str],
) -> None:
    configure_logging("WARNING", json_output=True)

    get_logger("tests.logging").info("quiet")
    get_logger("tests.logging").warning("loud")

    lines = capsys.readouterr().err.strip().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["event"] == "loud"


def test_a_logger_created_before_configuration_obeys_the_later_level(
    capsys: pytest.CaptureFixture[str],
) -> None:
    configure_logging("WARNING", json_output=True)

    created_at_import.info("quiet")
    created_at_import.warning("loud")

    lines = capsys.readouterr().err.strip().splitlines()
    assert [json.loads(line)["event"] for line in lines] == ["loud"]
    assert json.loads(lines[0])["logger"] == "tests.logging.module_level"


@pytest.mark.parametrize("json_output", [False, True])
def test_non_ascii_messages_survive_a_cp1252_stream(json_output: bool) -> None:
    raw = io.BytesIO()
    stream = io.TextIOWrapper(raw, encoding="cp1252", errors="strict")
    configure_logging("INFO", json_output=json_output, stream=stream)

    get_logger("tests.logging").warning("clipboard.failed", error="Åtkomst nekad → 0x5")
    stream.flush()

    written = raw.getvalue().decode("cp1252")
    assert "clipboard.failed" in written
    assert "tkomst nekad" in written


def test_unencodable_characters_are_escaped_not_dropped() -> None:
    raw = io.BytesIO()
    stream = io.TextIOWrapper(raw, encoding="cp1252", errors="strict")
    configure_logging("INFO", stream=stream)

    get_logger("tests.logging").info("arrow →")
    stream.flush()

    assert "\\u2192" in raw.getvalue().decode("cp1252")


def test_level_names_are_case_insensitive() -> None:
    configure_logging("debug")

    assert structlog.is_configured()


def test_unknown_level_fails_fast() -> None:
    with pytest.raises(KeyError):
        configure_logging("VERBOSE")


@pytest.mark.parametrize("key", ["input_tokens", "output_tokens", "keyboard_layout", "token_count"])
def test_counters_that_only_contain_a_marker_stay_readable(key: str) -> None:
    event = redact_secrets(None, "info", {"event": "x", key: 42})

    assert event[key] == 42


# -- rotating file (windowless app) ------------------------------------------------


@pytest.fixture
def log_path(tmp_path: Path) -> Iterator[Path]:
    yield tmp_path / "logs" / "jarvis.log"
    close_file_logging()


def test_file_logging_writes_redacted_json_lines_and_creates_the_directory(
    log_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    configure_logging("INFO", log_file=LogFile(log_path))

    get_logger("tests.file").info("started", api_key="secret", mode="cloud")
    get_logger("tests.file").debug("dropped")

    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["event"] == "started"
    assert record["logger"] == "tests.file"
    assert record["api_key"] == "***"
    assert "\x1b[" not in lines[0]  # never coloured
    assert capsys.readouterr().err == ""


def test_file_logging_rotates_at_the_size_cap_and_keeps_the_backups(log_path: Path) -> None:
    configure_logging("INFO", log_file=LogFile(log_path, max_bytes=2_000, backups=2))

    for index in range(200):
        get_logger("tests.file").info("filler", index=index, padding="x" * 40)

    names = sorted(path.name for path in log_path.parent.iterdir())
    assert names == ["jarvis.log", "jarvis.log.1", "jarvis.log.2"]
    assert all(path.stat().st_size <= 2_000 for path in log_path.parent.iterdir())


def test_exceptions_reach_the_file_with_one_traceback(log_path: Path) -> None:
    configure_logging("INFO", log_file=LogFile(log_path))

    try:
        raise ValueError("boom")
    except ValueError:
        get_logger("tests.file").exception("failed")

    record = json.loads(log_path.read_text(encoding="utf-8"))
    assert record["level"] == "error"
    assert record["exception"].count("Traceback") == 1


def test_reconfiguring_closes_the_previous_file_handler(log_path: Path, tmp_path: Path) -> None:
    configure_logging("INFO", log_file=LogFile(log_path))
    configure_logging("INFO", log_file=LogFile(tmp_path / "other.log"))

    assert len(logging.getLogger("jarvis.file").handlers) == 1
    close_file_logging()
    close_file_logging()  # idempotent
    assert logging.getLogger("jarvis.file").handlers == []
