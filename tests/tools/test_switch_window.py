from collections.abc import Callable

from jarvis.config import Settings
from jarvis.os_integration import StubPlatform
from jarvis.tools.switch_window import SwitchWindowTool
from tests.tools.local_fakes import CODE, NOTEPAD, local_context


async def test_focuses_the_first_title_match(make_settings: Callable[..., Settings]) -> None:
    platform = StubPlatform(windows=[NOTEPAD, CODE])

    result = await SwitchWindowTool().call(
        {"title": "visual studio"}, local_context(make_settings(), platform=platform)
    )

    assert result.ok
    assert result.spoken == f"Switched to {CODE.title}."
    assert platform.active == CODE
    assert result.data == {"handle": 202, "title": CODE.title, "pid": 4004}


async def test_a_miss_is_a_recoverable_failure(make_settings: Callable[..., Settings]) -> None:
    result = await SwitchWindowTool().call(
        {"title": "Excel"}, local_context(make_settings(), platform=StubPlatform())
    )

    assert not result.ok
    assert result.spoken == "I can't find a window called Excel."
