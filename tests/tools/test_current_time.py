from collections.abc import Callable
from datetime import UTC, datetime

import pytest

from jarvis.agent.registry import ToolRegistry
from jarvis.config import Settings
from jarvis.tools.current_time import CurrentTimeTool, spoken_time
from tests.agent.fakes import make_context
from tests.fakes import FakeClock


@pytest.fixture
def settings(make_settings: Callable[..., Settings]) -> Settings:
    return make_settings()


def test_is_discovered() -> None:
    registry = ToolRegistry.discover()

    assert isinstance(registry.get("current_time"), CurrentTimeTool)
    assert not registry.get("current_time").requires_confirmation


@pytest.mark.parametrize(
    ("hour", "minute", "expected"),
    [
        (0, 5, "12:05 AM"),
        (9, 30, "9:30 AM"),
        (12, 0, "12:00 PM"),
        (15, 42, "3:42 PM"),
        (23, 59, "11:59 PM"),
    ],
)
def test_spoken_time_uses_a_twelve_hour_clock(hour: int, minute: int, expected: str) -> None:
    assert spoken_time(datetime(2026, 1, 1, hour, minute, tzinfo=UTC)) == expected


async def test_reports_local_time_from_the_injected_clock(settings: Settings) -> None:
    clock = FakeClock()  # 2026-10-21 15:42:07 at UTC+3, a Wednesday
    result = await CurrentTimeTool().call({}, make_context(settings, clock=clock))

    assert result.ok
    assert result.spoken == "It's 3:42 PM on Wednesday, 21 October 2026."
    assert result.detail.startswith("2026-10-21T15:42:07+03:00 (Wednesday, ")
    assert result.data is not None
    assert result.data["weekday"] == "Wednesday"
    assert result.data["utc_offset"] == "+0300"
    assert result.data["unix"] == int(clock.utc().timestamp())


async def test_converts_to_a_requested_zone(settings: Settings) -> None:
    result = await CurrentTimeTool().call(
        {"timezone": "America/New_York"}, make_context(settings, clock=FakeClock())
    )

    assert result.ok
    assert result.spoken == "It's 8:42 AM on Wednesday, 21 October 2026 in America/New_York."
    assert result.data is not None
    assert result.data["iso"] == "2026-10-21T08:42:07-04:00"


async def test_unknown_zone_is_a_failed_result_not_an_exception(settings: Settings) -> None:
    result = await CurrentTimeTool().call({"timezone": "Mars/Olympus"}, make_context(settings))

    assert not result.ok
    assert "Mars/Olympus" in result.spoken
    assert "unknown IANA time zone" in result.detail


async def test_extra_arguments_are_rejected(settings: Settings) -> None:
    result = await CurrentTimeTool().call({"format": "24h"}, make_context(settings))

    assert not result.ok
    assert "format: Extra inputs are not permitted" in result.detail


def test_llm_schema_describes_the_timezone_argument() -> None:
    schema = CurrentTimeTool.to_llm_schema()

    assert schema["name"] == "current_time"
    assert "IANA" in schema["input_schema"]["properties"]["timezone"]["description"]
