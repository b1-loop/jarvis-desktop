import asyncio
from collections.abc import Callable
from pathlib import Path

import pytest

from jarvis.agent.tool import ToolContext
from jarvis.config import Settings
from jarvis.os_integration import StubPlatform
from jarvis.os_integration.stub import STUB_PNG
from jarvis.tools.take_screenshot import TakeScreenshotTool
from tests.tools.local_fakes import CODE, local_context


@pytest.fixture
def platform() -> StubPlatform:
    return StubPlatform(windows=[CODE], active=CODE)


@pytest.fixture
def context(
    make_settings: Callable[..., Settings], tmp_path: Path, platform: StubPlatform
) -> ToolContext:
    return local_context(make_settings(data_dir=tmp_path), platform=platform)


async def test_saves_a_png_under_the_data_directory(
    context: ToolContext, tmp_path: Path, platform: StubPlatform
) -> None:
    result = await TakeScreenshotTool().call({}, context)

    assert result.ok
    assert result.data is not None
    path = Path(str(result.data["path"]))
    assert path.parent == (tmp_path / "screenshots").resolve()
    assert path.name == "screenshot-20261021-154207.png"
    assert await asyncio.to_thread(path.read_bytes) == STUB_PNG
    assert result.spoken == f"Screenshot saved as {path.name}."
    assert platform.calls == [("screenshot", (False,))]


async def test_never_overwrites_an_earlier_capture(context: ToolContext) -> None:
    first = await TakeScreenshotTool().call({}, context)
    second = await TakeScreenshotTool().call({}, context)

    assert first.data is not None
    assert second.data is not None
    assert first.data["path"] != second.data["path"]
    assert str(second.data["path"]).endswith("-1.png")


async def test_can_capture_all_monitors_or_the_focused_window(
    context: ToolContext, platform: StubPlatform
) -> None:
    await TakeScreenshotTool().call({"all_monitors": True}, context)
    await TakeScreenshotTool().call({"active_window_only": True}, context)

    assert ("screenshot", (True,)) in platform.calls
    assert ("screenshot_window", (CODE.handle,)) in platform.calls


async def test_focused_window_capture_without_focus_fails_clearly(
    context: ToolContext, platform: StubPlatform
) -> None:
    platform.active = None

    result = await TakeScreenshotTool().call({"active_window_only": True}, context)

    assert not result.ok
    assert result.spoken == "There's no focused window to capture."
