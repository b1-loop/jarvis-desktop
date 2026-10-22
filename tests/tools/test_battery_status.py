from collections.abc import Callable
from types import SimpleNamespace

import psutil
import pytest

from jarvis.config import Settings
from jarvis.tools.battery_status import (
    BatteryReading,
    BatteryStatusTool,
    PsutilBatteryProbe,
)
from tests.tools.local_fakes import local_context


class FakeProbe:
    def __init__(self, reading: BatteryReading | None) -> None:
        self.reading = reading

    def read(self) -> BatteryReading | None:
        return self.reading


@pytest.mark.parametrize(
    ("reading", "spoken"),
    [
        (BatteryReading(81.4, True, None), "Battery is at 81 percent and charging."),
        (BatteryReading(40.0, False, None), "Battery is at 40 percent, on battery."),
        (BatteryReading(40.0, False, 45 * 60), "Battery is at 40 percent, about 45 minutes left."),
        (BatteryReading(70.0, False, 3600), "Battery is at 70 percent, about 1 hour left."),
        (
            BatteryReading(90.0, False, 2 * 3600 + 15 * 60),
            "Battery is at 90 percent, about 2 hours 15 minutes left.",
        ),
    ],
)
async def test_phrases_the_reading(
    make_settings: Callable[..., Settings], reading: BatteryReading, spoken: str
) -> None:
    result = await BatteryStatusTool(FakeProbe(reading)).call({}, local_context(make_settings()))

    assert result.ok
    assert result.spoken == spoken


async def test_desktop_without_a_battery(make_settings: Callable[..., Settings]) -> None:
    result = await BatteryStatusTool(FakeProbe(None)).call({}, local_context(make_settings()))

    assert result.spoken == "This computer doesn't have a battery."
    assert result.data == {"has_battery": False}


@pytest.mark.parametrize(
    ("secsleft", "expected"),
    [(1800, 1800), (psutil.POWER_TIME_UNKNOWN, None), (psutil.POWER_TIME_UNLIMITED, None)],
)
def test_psutil_probe_maps_special_estimates_to_none(
    monkeypatch: pytest.MonkeyPatch, secsleft: int, expected: int | None
) -> None:
    battery = SimpleNamespace(percent=55, power_plugged=False, secsleft=secsleft)
    monkeypatch.setattr(psutil, "sensors_battery", lambda: battery)

    assert PsutilBatteryProbe().read() == BatteryReading(55.0, False, expected)


def test_psutil_probe_without_a_battery(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(psutil, "sensors_battery", lambda: None)

    assert PsutilBatteryProbe().read() is None
