import json

import pytest
import structlog

from jarvis.utils.logging import configure_logging, get_logger, redact_secrets


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


def test_level_names_are_case_insensitive() -> None:
    configure_logging("debug")

    assert structlog.is_configured()


def test_unknown_level_fails_fast() -> None:
    with pytest.raises(KeyError):
        configure_logging("VERBOSE")
