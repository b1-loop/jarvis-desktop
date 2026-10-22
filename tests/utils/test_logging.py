import io
import json
from collections.abc import Iterator

import pytest
import structlog

from jarvis.utils.logging import configure_logging, get_logger, redact_secrets

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
