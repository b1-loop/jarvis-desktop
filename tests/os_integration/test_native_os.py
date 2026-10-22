"""Tests against the compiled ``jarvis_os`` module.

``native``-marked tests need the build and a desktop but change nothing on
the machine; ``hardware``-marked tests alter state (volume, clipboard,
notifications) and are only run by hand.
"""

from types import ModuleType

import pytest

from jarvis import _native
from jarvis.os_integration import UnsupportedFeatureError, create_windows_platform
from jarvis.utils.errors import NativeUnavailableError

pytestmark = pytest.mark.native

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


@pytest.fixture
def jarvis_os() -> ModuleType:
    try:
        return _native.load("jarvis_os")
    except NativeUnavailableError:
        pytest.skip("jarvis_os is not built")


def test_module_reports_implementation_status(jarvis_os: ModuleType) -> None:
    assert "step 4" in jarvis_os.IMPLEMENTATION_STATUS


def test_exceptions_form_a_hierarchy(jarvis_os: ModuleType) -> None:
    assert issubclass(jarvis_os.OsError, jarvis_os.NativeError)
    assert issubclass(jarvis_os.UnsupportedError, jarvis_os.OsError)
    assert issubclass(jarvis_os.WindowNotFoundError, jarvis_os.OsError)
    assert issubclass(jarvis_os.ProcessNotFoundError, jarvis_os.OsError)
    assert issubclass(jarvis_os.NativeError, RuntimeError)


def test_hotkeys_parse_and_format(jarvis_os: ModuleType) -> None:
    hotkey = jarvis_os.parse_hotkey("Ctrl + Alt + J")

    assert hotkey.modifiers == 0x0002 | 0x0001
    assert hotkey.virtual_key == ord("J")
    assert jarvis_os.format_hotkey(hotkey) == "ctrl+alt+j"
    assert hotkey == jarvis_os.parse_hotkey("alt+ctrl+j")
    with pytest.raises(ValueError, match="bananas"):
        jarvis_os.parse_hotkey("ctrl+bananas")


def test_listing_windows_describes_processes(jarvis_os: ModuleType) -> None:
    windows = jarvis_os.WindowManager().list_windows()

    for window in windows:
        assert window.title
        assert window.pid > 0
        assert window.handle > 0


def test_stale_handles_raise_window_not_found(jarvis_os: ModuleType) -> None:
    with pytest.raises(jarvis_os.WindowNotFoundError):
        jarvis_os.WindowManager().focus(0xDEADBEEF)
    with pytest.raises(jarvis_os.WindowNotFoundError):
        jarvis_os.Screenshotter().capture_window(0xDEADBEEF)


def test_unknown_processes_raise_process_not_found(jarvis_os: ModuleType) -> None:
    with pytest.raises(jarvis_os.ProcessNotFoundError):
        jarvis_os.ProcessControl().close(0x7FFFFFFD)
    with pytest.raises(jarvis_os.ProcessNotFoundError):
        jarvis_os.ProcessControl().close_by_name("jarvis-no-such-process-xyz")


def test_screenshot_is_a_png(jarvis_os: ModuleType) -> None:
    png = jarvis_os.Screenshotter().capture_screen(False)

    assert isinstance(png, bytes)
    assert png.startswith(PNG_SIGNATURE)
    assert len(png) > len(PNG_SIGNATURE) + 25  # signature + IHDR + something


def test_volume_reads_as_a_percentage_when_an_endpoint_exists(jarvis_os: ModuleType) -> None:
    controls = jarvis_os.SystemControls()
    try:
        volume = controls.volume()
    except jarvis_os.OsError:
        pytest.skip("no audio endpoint on this machine")

    assert 0 <= volume <= 100
    assert isinstance(controls.muted(), bool)


async def test_facade_reads_brightness_or_reports_unsupported(jarvis_os: ModuleType) -> None:
    platform = create_windows_platform()
    try:
        brightness = await platform.get_brightness()
    except UnsupportedFeatureError:
        return
    assert 0 <= brightness <= 100


async def test_facade_lists_windows_and_captures_the_screen(jarvis_os: ModuleType) -> None:
    async with create_windows_platform() as platform:
        windows = await platform.list_windows()
        png = await platform.screenshot()

    assert all(window.title for window in windows)
    assert png.startswith(PNG_SIGNATURE)


@pytest.mark.hardware
async def test_hotkey_registration_round_trip(jarvis_os: ModuleType) -> None:
    async with create_windows_platform() as platform:
        hotkey_id = await platform.register_hotkey("ctrl+alt+shift+f24")
        assert platform.poll_hotkeys() == []
        await platform.unregister_hotkey(hotkey_id)


@pytest.mark.hardware
async def test_volume_round_trip(jarvis_os: ModuleType) -> None:
    async with create_windows_platform() as platform:
        original = await platform.get_volume()
        await platform.set_volume(33)
        assert await platform.get_volume() == 33
        await platform.set_volume(original)


@pytest.mark.hardware
async def test_clipboard_round_trip_and_notification(jarvis_os: ModuleType) -> None:
    async with create_windows_platform() as platform:
        original = await platform.get_clipboard_text()
        await platform.set_clipboard_text("jarvis ✓")
        assert await platform.get_clipboard_text() == "jarvis ✓"
        await platform.set_clipboard_text(original)
        await platform.notify("Jarvis", "hardware test notification")
