from collections.abc import Callable
from dataclasses import replace

from jarvis.config import Settings
from jarvis.os_integration import StubPlatform
from jarvis.tools.list_windows import ListWindowsTool
from tests.tools.local_fakes import CODE, NOTEPAD, local_context


async def test_lists_windows_front_most_first(make_settings: Callable[..., Settings]) -> None:
    platform = StubPlatform(windows=[CODE, NOTEPAD])

    result = await ListWindowsTool().call({}, local_context(make_settings(), platform=platform))

    assert result.ok
    assert result.spoken == f"2 windows: {CODE.title}, {NOTEPAD.title}."
    assert result.data is not None
    assert result.data["windows"] == [
        {"handle": 202, "title": CODE.title, "process_name": "Code.exe", "pid": 4004},
        {"handle": 101, "title": NOTEPAD.title, "process_name": "notepad.exe", "pid": 4000},
    ]
    assert "notepad.exe, pid 4000, handle 101" in result.detail


async def test_filters_by_title_or_executable(make_settings: Callable[..., Settings]) -> None:
    platform = StubPlatform(windows=[CODE, NOTEPAD])
    context = local_context(make_settings(), platform=platform)

    by_exe = await ListWindowsTool().call({"contains": "NOTEPAD.EXE"}, context)
    by_title = await ListWindowsTool().call({"contains": "visual studio"}, context)

    assert by_exe.spoken == f"1 windows: {NOTEPAD.title}."
    assert by_title.spoken == f"1 windows: {CODE.title}."


async def test_mentions_how_many_more_there_are(make_settings: Callable[..., Settings]) -> None:
    windows = [replace(NOTEPAD, handle=n, title=f"Window {n}") for n in range(5)]

    result = await ListWindowsTool().call(
        {"limit": 4}, local_context(make_settings(), platform=StubPlatform(windows=windows))
    )

    assert result.spoken == "5 windows: Window 0, Window 1, Window 2, and 2 more."
    assert result.data is not None
    assert isinstance(result.data["windows"], list)
    assert len(result.data["windows"]) == 4


async def test_says_so_when_nothing_matches(make_settings: Callable[..., Settings]) -> None:
    result = await ListWindowsTool().call(
        {"contains": "excel"}, local_context(make_settings(), platform=StubPlatform())
    )

    assert result.ok
    assert result.spoken == "I don't see any matching windows."
