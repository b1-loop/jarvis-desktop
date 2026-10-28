import asyncio

import pytest

from jarvis import _native
from jarvis.os_integration import (
    ClipboardBusyError,
    InvalidHotkeyError,
    PlatformError,
    ProcessNotFoundError,
    UnsupportedFeatureError,
    WindowInfo,
    WindowNotFoundError,
    WindowsPlatform,
    create_windows_platform,
)
from jarvis.os_integration.windows import translate_native_error
from jarvis.utils.errors import NativeUnavailableError
from tests.os_integration import fakes
from tests.os_integration.fakes import FAKE_PNG, FakeNativeSet, FakeWindowInfo


@pytest.fixture
def native() -> FakeNativeSet:
    return FakeNativeSet()


@pytest.fixture
def platform(native: FakeNativeSet) -> WindowsPlatform:
    return WindowsPlatform(native.components())


# --- hotkeys -----------------------------------------------------------------


async def test_hotkeys_are_registered_lazily_and_pressed_events_are_polled(
    platform: WindowsPlatform, native: FakeNativeSet
) -> None:
    assert platform.poll_hotkeys() == []
    assert native.hotkey_managers_made == 0

    hotkey_id = await platform.register_hotkey("Ctrl + Alt + J")
    await platform.register_hotkey("win+k")
    native.hotkeys.press(hotkey_id)

    events = platform.poll_hotkeys()
    assert native.hotkey_managers_made == 1
    assert [(e.hotkey_id, e.hotkey) for e in events] == [(hotkey_id, "alt+ctrl+j")]
    assert platform.poll_hotkeys() == []


async def test_hotkey_events_stream_yields_presses(
    platform: WindowsPlatform, native: FakeNativeSet
) -> None:
    hotkey_id = await platform.register_hotkey("ctrl+j")
    native.hotkeys.press(hotkey_id)

    stream = platform.hotkey_events(poll_interval=0.001)
    event = await asyncio.wait_for(anext(stream), 1.0)

    assert event.hotkey_id == hotkey_id


async def test_malformed_hotkey_raises_invalid_hotkey_error(platform: WindowsPlatform) -> None:
    with pytest.raises(InvalidHotkeyError, match="malformed"):
        await platform.register_hotkey("ctrl++")


async def test_taken_hotkey_surfaces_as_platform_error(platform: WindowsPlatform) -> None:
    await platform.register_hotkey("ctrl+j")

    with pytest.raises(PlatformError, match="already registered"):
        await platform.register_hotkey("ctrl+j")


async def test_unregister_releases_and_tolerates_unknown_ids(
    platform: WindowsPlatform, native: FakeNativeSet
) -> None:
    await platform.unregister_hotkey(99)  # no manager yet: nothing happens
    hotkey_id = await platform.register_hotkey("ctrl+j")

    await platform.unregister_hotkey(hotkey_id)
    await platform.unregister_hotkey(hotkey_id)

    assert native.hotkeys.registered == {}


# --- windows -----------------------------------------------------------------


async def test_windows_are_converted_to_dataclasses(
    platform: WindowsPlatform, native: FakeNativeSet
) -> None:
    native.windows.windows = [FakeWindowInfo(7, "Notes - Code", "Code.exe", r"C:\code.exe", 42)]
    native.windows.active = native.windows.windows[0]

    listed = await platform.list_windows()
    active = await platform.active_window()

    expected = WindowInfo(7, "Notes - Code", "Code.exe", r"C:\code.exe", 42)
    assert listed == [expected]
    assert active == expected


async def test_active_window_can_be_none(platform: WindowsPlatform) -> None:
    assert await platform.active_window() is None


async def test_window_management_forwards_handles(
    platform: WindowsPlatform, native: FakeNativeSet
) -> None:
    native.windows.windows = [FakeWindowInfo(7, "Editor")]

    await platform.focus_window(7)
    await platform.minimize_window(7)
    await platform.maximize_window(7)
    await platform.restore_window(7)
    await platform.close_window(7)

    assert native.windows.calls == [
        ("focus", 7),
        ("minimize", 7),
        ("maximize", 7),
        ("restore", 7),
        ("close", 7),
    ]


async def test_focus_by_title_returns_the_window(
    platform: WindowsPlatform, native: FakeNativeSet
) -> None:
    native.windows.windows = [FakeWindowInfo(7, "Inbox - Mail"), FakeWindowInfo(8, "Editor")]

    window = await platform.focus_window_by_title("editor")

    assert window.handle == 8
    assert native.windows.active is native.windows.windows[1]


async def test_stale_handle_and_unmatched_title_raise_window_not_found(
    platform: WindowsPlatform,
) -> None:
    with pytest.raises(WindowNotFoundError):
        await platform.focus_window(123)
    with pytest.raises(WindowNotFoundError):
        await platform.focus_window_by_title("nothing")
    with pytest.raises(WindowNotFoundError):
        await platform.screenshot_window(0)


# --- processes ---------------------------------------------------------------


async def test_launch_and_close_round_trip(
    platform: WindowsPlatform, native: FakeNativeSet
) -> None:
    pid = await platform.launch("notepad", "notes.txt")
    await platform.close_process(pid)
    await platform.close_process(pid, force=True)
    count = await platform.close_process_by_name("notepad", force=False)

    assert native.processes.launched == [("notepad", "notes.txt")]
    assert native.processes.closed == [(pid, False), (pid, True), (pid, False)]
    assert count == 1


async def test_unknown_process_raises_process_not_found(platform: WindowsPlatform) -> None:
    with pytest.raises(ProcessNotFoundError):
        await platform.close_process(1)
    with pytest.raises(ProcessNotFoundError):
        await platform.close_process_by_name("ghost")


# --- input, notifications, clipboard -------------------------------------------


async def test_text_clipboard_and_notifications(
    platform: WindowsPlatform, native: FakeNativeSet
) -> None:
    await platform.type_text("hello", press_enter=True)
    await platform.set_clipboard_text("copied")
    await platform.notify("Jarvis", "done")
    await platform.notify("Jarvis", "again")

    assert native.text_injector.typed == [("hello", True)]
    assert await platform.get_clipboard_text() == "copied"
    assert native.notifier.shown == [("Jarvis", "done"), ("Jarvis", "again")]
    assert native.notifiers_made == 1  # created once, on first use


# --- system controls -----------------------------------------------------------


async def test_volume_mute_and_brightness(platform: WindowsPlatform, native: FakeNativeSet) -> None:
    await platform.set_volume(30)
    await platform.set_muted(True)
    await platform.set_brightness(80)
    await platform.lock_screen()

    assert await platform.get_volume() == 30
    assert await platform.is_muted() is True
    assert await platform.get_brightness() == 80
    assert native.system.locked == 1


async def test_brightness_unsupported_is_translated(
    platform: WindowsPlatform, native: FakeNativeSet
) -> None:
    native.system.panel = None

    with pytest.raises(UnsupportedFeatureError):
        await platform.get_brightness()
    with pytest.raises(UnsupportedFeatureError):
        await platform.set_brightness(10)


# --- screenshots -----------------------------------------------------------------


async def test_screenshots_return_png_bytes(
    platform: WindowsPlatform, native: FakeNativeSet
) -> None:
    assert await platform.screenshot() == FAKE_PNG
    assert await platform.screenshot(all_monitors=True) == FAKE_PNG
    assert await platform.screenshot_window(5) == FAKE_PNG
    assert native.screenshotter.captures == ["primary", "all", "window:5"]


# --- lifecycle and errors ------------------------------------------------------


async def test_close_releases_lazily_created_components(
    platform: WindowsPlatform, native: FakeNativeSet
) -> None:
    await platform.register_hotkey("ctrl+j")
    await platform.notify("t", "m")

    async with platform:
        pass

    assert platform.poll_hotkeys() == []
    await platform.notify("t", "m")  # a new notifier is created on demand
    assert native.notifiers_made == 2


def test_native_errors_translate_by_class_name() -> None:
    assert type(translate_native_error(fakes.OsError("x"))) is PlatformError
    assert type(translate_native_error(fakes.NativeError("x"))) is PlatformError
    assert isinstance(translate_native_error(fakes.WindowNotFoundError("x")), WindowNotFoundError)
    assert isinstance(translate_native_error(fakes.ProcessNotFoundError("x")), ProcessNotFoundError)
    assert isinstance(translate_native_error(fakes.UnsupportedError("x")), UnsupportedFeatureError)
    assert isinstance(translate_native_error(fakes.ClipboardBusyError("x")), ClipboardBusyError)
    assert translate_native_error(RuntimeError("not ours")) is None


async def test_foreign_runtime_errors_are_not_swallowed(native: FakeNativeSet) -> None:
    def broken() -> int:
        raise RuntimeError("bug in a fake")

    native.system.volume = broken  # type: ignore[method-assign]
    platform = WindowsPlatform(native.components())

    with pytest.raises(RuntimeError, match="bug in a fake"):
        await platform.get_volume()


def test_create_windows_platform_needs_the_native_module(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing(name: str) -> None:
        raise NativeUnavailableError(name, "not built")

    monkeypatch.setattr(_native, "load", missing)

    with pytest.raises(NativeUnavailableError):
        create_windows_platform()
