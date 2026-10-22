from collections.abc import Callable

import pytest

from jarvis.agent.tool import ToolContext
from jarvis.config import Settings
from jarvis.os_integration import StubPlatform
from jarvis.tools.volume import GetVolumeTool, MuteTool, SetVolumeTool
from tests.tools.local_fakes import local_context


@pytest.fixture
def platform() -> StubPlatform:
    return StubPlatform(volume=30)


@pytest.fixture
def context(make_settings: Callable[..., Settings], platform: StubPlatform) -> ToolContext:
    return local_context(make_settings(), platform=platform)


async def test_get_volume_reports_level_and_mute(
    context: ToolContext, platform: StubPlatform
) -> None:
    unmuted = await GetVolumeTool().call({}, context)
    platform.muted = True
    muted = await GetVolumeTool().call({}, context)

    assert unmuted.spoken == "Volume is at 30 percent."
    assert muted.spoken == "Volume is at 30 percent, but sound is muted."
    assert muted.data == {"volume": 30, "muted": True}


async def test_set_volume_to_an_absolute_level_unmutes(
    context: ToolContext, platform: StubPlatform
) -> None:
    platform.muted = True

    result = await SetVolumeTool().call({"percent": 40}, context)

    assert result.spoken == "Volume set to 40 percent."
    assert platform.volume == 40
    assert platform.muted is False


@pytest.mark.parametrize(("change", "expected"), [(15, 45), (-50, 0), (90, 100)])
async def test_set_volume_by_a_step_is_clamped(
    context: ToolContext, platform: StubPlatform, change: int, expected: int
) -> None:
    result = await SetVolumeTool().call({"change": change}, context)

    assert result.data == {"volume": expected}
    assert platform.volume == expected


async def test_setting_zero_does_not_unmute(context: ToolContext, platform: StubPlatform) -> None:
    platform.muted = True

    await SetVolumeTool().call({"percent": 0}, context)

    assert platform.muted is True


@pytest.mark.parametrize("raw", [{}, {"percent": 10, "change": 5}, {"percent": 101}])
async def test_set_volume_rejects_ambiguous_or_out_of_range_input(
    context: ToolContext, raw: dict[str, int]
) -> None:
    result = await SetVolumeTool().call(raw, context)

    assert not result.ok


async def test_mute_and_unmute(context: ToolContext, platform: StubPlatform) -> None:
    muted = await MuteTool().call({}, context)
    assert platform.muted is True
    unmuted = await MuteTool().call({"muted": False}, context)

    assert muted.spoken == "Muted."
    assert unmuted.spoken == "Sound is back on."
    assert platform.muted is False
