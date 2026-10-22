import asyncio

import pytest

from jarvis.os_integration import (
    InvalidHotkeyError,
    Platform,
    ProcessNotFoundError,
    StubPlatform,
    WindowInfo,
    WindowNotFoundError,
)
from jarvis.os_integration.stub import STUB_PNG

EDITOR = WindowInfo(1, "Editor - main.py", "Code.exe", r"C:\Code.exe", 10)
MAIL = WindowInfo(2, "Inbox", "Mail.exe", r"C:\Mail.exe", 11)


@pytest.fixture
def stub() -> StubPlatform:
    return StubPlatform(windows=[EDITOR, MAIL])


def test_stub_is_a_platform(stub: StubPlatform) -> None:
    assert isinstance(stub, Platform)


async def test_every_call_is_recorded(stub: StubPlatform) -> None:
    await stub.set_volume(20)
    await stub.lock_screen()

    assert stub.calls == [("set_volume", (20,)), ("lock_screen", ())]


async def test_hotkeys_can_be_pressed_from_tests(stub: StubPlatform) -> None:
    hotkey_id = await stub.register_hotkey("Ctrl + Alt + J")
    stub.press(hotkey_id)

    stream = stub.hotkey_events(poll_interval=0.001)
    event = await asyncio.wait_for(anext(stream), 1.0)

    assert event.hotkey == "ctrl+alt+j"
    assert stub.poll_hotkeys() == []
    await stub.unregister_hotkey(hotkey_id)
    assert stub.hotkeys == {}


async def test_empty_hotkey_is_rejected(stub: StubPlatform) -> None:
    with pytest.raises(InvalidHotkeyError):
        await stub.register_hotkey("  ")


async def test_window_queries_and_focus(stub: StubPlatform) -> None:
    assert await stub.active_window() is None
    assert await stub.list_windows() == [EDITOR, MAIL]

    await stub.focus_window(2)
    assert await stub.active_window() == MAIL

    assert await stub.focus_window_by_title("editor") == EDITOR
    assert stub.active == EDITOR


async def test_window_management_validates_handles(stub: StubPlatform) -> None:
    await stub.minimize_window(1)
    await stub.maximize_window(1)
    await stub.restore_window(1)
    await stub.close_window(1)

    assert await stub.list_windows() == [MAIL]
    with pytest.raises(WindowNotFoundError):
        await stub.focus_window(1)
    with pytest.raises(WindowNotFoundError):
        await stub.focus_window_by_title("nothing like this")
    with pytest.raises(WindowNotFoundError):
        await stub.screenshot_window(1)


async def test_processes(stub: StubPlatform) -> None:
    pid = await stub.launch("notepad")

    await stub.close_process(pid)
    assert await stub.close_process_by_name("mail.exe") == 1
    with pytest.raises(ProcessNotFoundError):
        await stub.close_process(pid + 4)
    with pytest.raises(ProcessNotFoundError):
        await stub.close_process_by_name("ghost.exe")


async def test_input_clipboard_and_notifications(stub: StubPlatform) -> None:
    await stub.type_text("hello", press_enter=True)
    await stub.set_clipboard_text("copied")
    await stub.notify("Jarvis", "done")

    assert stub.typed == ["hello\n"]
    assert await stub.get_clipboard_text() == "copied"
    assert stub.notifications == [("Jarvis", "done")]


async def test_system_controls_clamp_and_persist(stub: StubPlatform) -> None:
    await stub.set_volume(150)
    await stub.set_brightness(-10)
    await stub.set_muted(True)

    assert await stub.get_volume() == 100
    assert await stub.get_brightness() == 0
    assert await stub.is_muted() is True


async def test_screenshots_are_real_png_files(stub: StubPlatform) -> None:
    png = await stub.screenshot(all_monitors=True)

    assert png == STUB_PNG
    assert png.startswith(b"\x89PNG\r\n\x1a\n")
    assert await stub.screenshot_window(1) == STUB_PNG


async def test_context_manager_closes_cleanly(stub: StubPlatform) -> None:
    async with stub as platform:
        assert platform is stub
