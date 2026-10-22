from collections.abc import Callable
from dataclasses import dataclass

import pytest

from jarvis.config import Settings
from jarvis.os_integration import StubPlatform, UnsupportedFeatureError
from jarvis.tools.set_brightness import UNSUPPORTED_SPOKEN, SetBrightnessTool
from tests.tools.local_fakes import local_context


@dataclass
class DesktopPlatform(StubPlatform):
    """A desktop with external monitors: no software brightness."""

    async def get_brightness(self) -> int:
        raise UnsupportedFeatureError("no display supports brightness control")

    async def set_brightness(self, percent: int) -> None:
        raise UnsupportedFeatureError("no display supports brightness control")


@pytest.mark.parametrize(
    ("raw", "expected"), [({"percent": 70}, 70), ({"change": -20}, 30), ({"change": 80}, 100)]
)
async def test_sets_an_absolute_or_relative_level(
    make_settings: Callable[..., Settings], raw: dict[str, int], expected: int
) -> None:
    platform = StubPlatform(brightness=50)

    result = await SetBrightnessTool().call(raw, local_context(make_settings(), platform=platform))

    assert result.ok
    assert result.spoken == f"Brightness set to {expected} percent."
    assert platform.brightness == expected


@pytest.mark.parametrize("raw", [{"percent": 40}, {"change": 10}])
async def test_unsupported_hardware_gets_a_clear_spoken_answer(
    make_settings: Callable[..., Settings], raw: dict[str, int]
) -> None:
    result = await SetBrightnessTool().call(
        raw, local_context(make_settings(), platform=DesktopPlatform())
    )

    assert not result.ok
    assert result.spoken == UNSUPPORTED_SPOKEN
    assert "no display supports" in result.detail


async def test_requires_exactly_one_field(make_settings: Callable[..., Settings]) -> None:
    result = await SetBrightnessTool().call({}, local_context(make_settings()))

    assert not result.ok
    assert "exactly one" in result.detail
