"""Windows implementation of :class:`Platform` backed by ``jarvis_os``.

The native module does the Win32 work on the calling thread; this class
runs each call through ``asyncio.to_thread`` so the event loop never blocks
on COM, WMI or screen capture, and translates the native exception
hierarchy into :mod:`jarvis.os_integration.errors`. Native objects are
injected through Protocols, so the facade is unit-tested with fakes and the
real ``.pyd`` is exercised only by ``native``-marked tests.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from types import ModuleType
from typing import Final, Protocol, cast

from jarvis import _native
from jarvis.os_integration.base import HotkeyEvent, Platform, WindowInfo
from jarvis.os_integration.errors import (
    ClipboardBusyError,
    InvalidHotkeyError,
    PlatformError,
    ProcessNotFoundError,
    UnsupportedFeatureError,
    WindowNotFoundError,
)
from jarvis.utils.logging import get_logger

log = get_logger(__name__)


class NativeHotkey(Protocol):
    """Shape of ``jarvis_os.Hotkey``."""

    @property
    def modifiers(self) -> int:
        """Bitmask of Win32 ``MOD_*`` flags."""

    @property
    def virtual_key(self) -> int:
        """Win32 virtual-key code."""


class NativeHotkeyEvent(Protocol):
    """Shape of ``jarvis_os.HotkeyEvent``."""

    @property
    def id(self) -> int:
        """Registration id."""

    @property
    def hotkey(self) -> NativeHotkey:
        """The combination that was pressed."""


class NativeHotkeyManager(Protocol):
    """Shape of ``jarvis_os.HotkeyManager``."""

    def register_hotkey(self, hotkey: NativeHotkey) -> int:
        """Register system-wide; returns the id."""

    def unregister_hotkey(self, hotkey_id: int) -> None:
        """Release a registration."""

    def poll_events(self) -> list[NativeHotkeyEvent]:
        """Drain presses since the previous poll."""


class NativeWindowInfo(Protocol):
    """Shape of ``jarvis_os.WindowInfo``."""

    @property
    def handle(self) -> int:
        """HWND as an integer."""

    @property
    def title(self) -> str:
        """Window title."""

    @property
    def process_name(self) -> str:
        """Executable file name."""

    @property
    def executable_path(self) -> str:
        """Full executable path, possibly empty."""

    @property
    def pid(self) -> int:
        """Owning process id."""


class NativeWindowManager(Protocol):
    """Shape of ``jarvis_os.WindowManager``."""

    def active_window(self) -> NativeWindowInfo | None:
        """Foreground window, if any."""

    def list_windows(self) -> list[NativeWindowInfo]:
        """Visible top-level windows."""

    def focus(self, handle: int) -> None:
        """Bring to the foreground."""

    def focus_by_title(self, substring: str) -> NativeWindowInfo:
        """Focus by case-insensitive title substring."""

    def minimize(self, handle: int) -> None:
        """Minimise."""

    def maximize(self, handle: int) -> None:
        """Maximise."""

    def restore(self, handle: int) -> None:
        """Restore."""

    def close(self, handle: int) -> None:
        """Post WM_CLOSE."""


class NativeProcessControl(Protocol):
    """Shape of ``jarvis_os.ProcessControl``."""

    def launch(self, target: str, arguments: str) -> int:
        """ShellExecute ``target``; returns the pid or 0."""

    def close(self, pid: int, force: bool) -> None:
        """Close or terminate one process."""

    def close_by_name(self, executable_name: str, force: bool) -> int:
        """Close every process with that image name; returns the count."""


class NativeTextInjector(Protocol):
    """Shape of ``jarvis_os.TextInjector``."""

    def type_text(self, text: str, press_enter: bool) -> None:
        """SendInput the text."""


class NativeNotifier(Protocol):
    """Shape of ``jarvis_os.Notifier``."""

    def notify(self, title: str, message: str) -> None:
        """Show a balloon/toast."""


class NativeClipboard(Protocol):
    """Shape of ``jarvis_os.Clipboard``."""

    def text(self) -> str:
        """Clipboard text or empty."""

    def set_text(self, text: str) -> None:
        """Replace the clipboard text."""


class NativeSystemControls(Protocol):
    """Shape of ``jarvis_os.SystemControls``."""

    def volume(self) -> int:
        """Master volume 0-100."""

    def set_volume(self, percent: int) -> None:
        """Set master volume."""

    def muted(self) -> bool:
        """Mute state."""

    def set_muted(self, muted: bool) -> None:
        """Set mute state."""

    def brightness(self) -> int:
        """Built-in display brightness 0-100."""

    def set_brightness(self, percent: int) -> None:
        """Set built-in display brightness."""

    def lock_screen(self) -> None:
        """LockWorkStation."""


class NativeScreenshotter(Protocol):
    """Shape of ``jarvis_os.Screenshotter``."""

    def capture_screen(self, all_monitors: bool) -> bytes:
        """PNG of the primary monitor or the virtual desktop."""

    def capture_window(self, handle: int) -> bytes:
        """PNG of one window's rectangle."""


@dataclass(frozen=True, slots=True)
class NativeComponents:
    """Everything :class:`WindowsPlatform` needs from the native module.

    Hotkeys and the notifier are factories rather than instances because
    constructing them has side effects (a message-loop thread, a tray icon)
    that should only happen when a tool first needs them.
    """

    windows: NativeWindowManager
    processes: NativeProcessControl
    text_injector: NativeTextInjector
    clipboard: NativeClipboard
    system: NativeSystemControls
    screenshotter: NativeScreenshotter
    make_hotkey_manager: Callable[[], NativeHotkeyManager]
    make_notifier: Callable[[], NativeNotifier]
    parse_hotkey: Callable[[str], NativeHotkey]
    format_hotkey: Callable[[NativeHotkey], str]

    @classmethod
    def from_module(cls, module: ModuleType) -> NativeComponents:
        """Wire the real ``jarvis_os`` classes."""
        return cls(
            windows=cast(NativeWindowManager, module.WindowManager()),
            processes=cast(NativeProcessControl, module.ProcessControl()),
            text_injector=cast(NativeTextInjector, module.TextInjector()),
            clipboard=cast(NativeClipboard, module.Clipboard()),
            system=cast(NativeSystemControls, module.SystemControls()),
            screenshotter=cast(NativeScreenshotter, module.Screenshotter()),
            make_hotkey_manager=cast(Callable[[], NativeHotkeyManager], module.HotkeyManager),
            make_notifier=cast(Callable[[], NativeNotifier], module.Notifier),
            parse_hotkey=cast(Callable[[str], NativeHotkey], module.parse_hotkey),
            format_hotkey=cast(Callable[[NativeHotkey], str], module.format_hotkey),
        )


_NATIVE_ERROR_NAMES: Final[dict[str, type[PlatformError]]] = {
    "NativeError": PlatformError,
    "OsError": PlatformError,
    "UnsupportedError": UnsupportedFeatureError,
    "WindowNotFoundError": WindowNotFoundError,
    "ProcessNotFoundError": ProcessNotFoundError,
    "ClipboardBusyError": ClipboardBusyError,
}
"""Native exception class names (pybind11 registers them as RuntimeError subclasses)."""


def translate_native_error(error: RuntimeError) -> PlatformError | None:
    """Map a ``jarvis_os`` exception onto the Python hierarchy, by class name.

    By name because the native classes only exist when the ``.pyd`` is
    loaded, and the facade must not import it to be testable. Returns
    ``None`` for a ``RuntimeError`` that is not one of ours, so genuine
    programming errors still surface unchanged.
    """
    mapped = _NATIVE_ERROR_NAMES.get(type(error).__name__)
    return None if mapped is None else mapped(str(error))


class WindowsPlatform(Platform):
    """Win32 backend: every call runs off the event loop and errors are translated."""

    def __init__(self, native: NativeComponents) -> None:
        self._native = native
        self._hotkeys: NativeHotkeyManager | None = None
        self._notifier: NativeNotifier | None = None

    async def close(self) -> None:
        """Release the hotkey thread and the tray icon; idempotent."""
        hotkeys, self._hotkeys = self._hotkeys, None
        notifier, self._notifier = self._notifier, None
        # Dropping the last reference joins native threads; keep that off the loop.
        await asyncio.to_thread(_release, hotkeys, notifier)
        log.info("platform.closed")

    async def _call[T](self, function: Callable[..., T], *args: object) -> T:
        try:
            return await asyncio.to_thread(function, *args)
        except RuntimeError as error:
            translated = translate_native_error(error)
            if translated is None:
                raise
            log.warning("platform.native_error", error=str(error), kind=type(translated).__name__)
            raise translated from error

    # --- hotkeys ---------------------------------------------------------

    def _hotkey_manager(self) -> NativeHotkeyManager:
        if self._hotkeys is None:
            self._hotkeys = self._native.make_hotkey_manager()
        return self._hotkeys

    async def register_hotkey(self, spec: str) -> int:
        """Parse the spec, then register it on the native message-loop thread."""
        try:
            hotkey = self._native.parse_hotkey(spec)
        except ValueError as error:
            raise InvalidHotkeyError(str(error)) from error
        manager = await asyncio.to_thread(self._hotkey_manager)
        hotkey_id = await self._call(manager.register_hotkey, hotkey)
        log.info("platform.hotkey.registered", hotkey=self._native.format_hotkey(hotkey))
        return hotkey_id

    async def unregister_hotkey(self, hotkey_id: int) -> None:
        """Release a hotkey; a no-op before any hotkey was registered."""
        if self._hotkeys is not None:
            await self._call(self._hotkeys.unregister_hotkey, hotkey_id)

    def poll_hotkeys(self) -> list[HotkeyEvent]:
        """Drain native presses and spell each combination canonically."""
        if self._hotkeys is None:
            return []
        return [
            HotkeyEvent(event.id, self._native.format_hotkey(event.hotkey))
            for event in self._hotkeys.poll_events()
        ]

    # --- windows ---------------------------------------------------------

    async def active_window(self) -> WindowInfo | None:
        """Foreground window via GetForegroundWindow, converted to a dataclass."""
        window = await self._call(self._native.windows.active_window)
        return None if window is None else _convert_window(window)

    async def list_windows(self) -> list[WindowInfo]:
        """Alt+Tab-style window list via EnumWindows."""
        windows = await self._call(self._native.windows.list_windows)
        return [_convert_window(window) for window in windows]

    async def focus_window(self, handle: int) -> None:
        """SetForegroundWindow with the thread-input dance; raises if refused."""
        await self._call(self._native.windows.focus, handle)

    async def focus_window_by_title(self, substring: str) -> WindowInfo:
        """Focus the first listed window whose title contains the text."""
        return _convert_window(await self._call(self._native.windows.focus_by_title, substring))

    async def minimize_window(self, handle: int) -> None:
        """ShowWindow(SW_MINIMIZE)."""
        await self._call(self._native.windows.minimize, handle)

    async def maximize_window(self, handle: int) -> None:
        """ShowWindow(SW_MAXIMIZE)."""
        await self._call(self._native.windows.maximize, handle)

    async def restore_window(self, handle: int) -> None:
        """ShowWindow(SW_RESTORE)."""
        await self._call(self._native.windows.restore, handle)

    async def close_window(self, handle: int) -> None:
        """Post WM_CLOSE; returns before the application reacts."""
        await self._call(self._native.windows.close, handle)

    # --- processes -------------------------------------------------------

    async def launch(self, target: str, arguments: str = "") -> int:
        """ShellExecuteEx the target so names, documents and URLs all work."""
        pid = await self._call(self._native.processes.launch, target, arguments)
        log.info("platform.launched", target=target, pid=pid)
        return pid

    async def close_process(self, pid: int, *, force: bool = False) -> None:
        """WM_CLOSE to the process's windows, or TerminateProcess with force."""
        await self._call(self._native.processes.close, pid, force)
        log.info("platform.process.closed", pid=pid, force=force)

    async def close_process_by_name(self, name: str, *, force: bool = False) -> int:
        """Close every process whose image name matches."""
        count = await self._call(self._native.processes.close_by_name, name, force)
        log.info("platform.process.closed_by_name", name=name, count=count, force=force)
        return count

    # --- input, notifications, clipboard ---------------------------------

    async def type_text(self, text: str, *, press_enter: bool = False) -> None:
        """SendInput with Unicode key events, layout-independent."""
        await self._call(self._native.text_injector.type_text, text, press_enter)

    async def notify(self, title: str, message: str) -> None:
        """Balloon notification through the tray icon, created on first use."""
        if self._notifier is None:
            self._notifier = await asyncio.to_thread(self._native.make_notifier)
        await self._call(self._notifier.notify, title, message)

    async def get_clipboard_text(self) -> str:
        """CF_UNICODETEXT contents, or empty."""
        return await self._call(self._native.clipboard.text)

    async def set_clipboard_text(self, text: str) -> None:
        """Replace the clipboard with CF_UNICODETEXT."""
        await self._call(self._native.clipboard.set_text, text)

    # --- system controls -------------------------------------------------

    async def get_volume(self) -> int:
        """Core Audio master volume as a percentage."""
        return await self._call(self._native.system.volume)

    async def set_volume(self, percent: int) -> None:
        """Set the Core Audio master volume; clamped natively."""
        await self._call(self._native.system.set_volume, percent)

    async def is_muted(self) -> bool:
        """Core Audio master mute state."""
        return await self._call(self._native.system.muted)

    async def set_muted(self, muted: bool) -> None:
        """Set the Core Audio master mute state."""
        await self._call(self._native.system.set_muted, muted)

    async def get_brightness(self) -> int:
        """WMI panel brightness; unsupported on external monitors."""
        return await self._call(self._native.system.brightness)

    async def set_brightness(self, percent: int) -> None:
        """WMI panel brightness; unsupported on external monitors."""
        await self._call(self._native.system.set_brightness, percent)

    async def lock_screen(self) -> None:
        """LockWorkStation."""
        await self._call(self._native.system.lock_screen)

    # --- screenshots -----------------------------------------------------

    async def screenshot(self, *, all_monitors: bool = False) -> bytes:
        """GDI capture encoded to PNG by WIC, off the event loop."""
        return await self._call(self._native.screenshotter.capture_screen, all_monitors)

    async def screenshot_window(self, handle: int) -> bytes:
        """GDI capture of one window's frame, encoded to PNG."""
        return await self._call(self._native.screenshotter.capture_window, handle)


def _convert_window(window: NativeWindowInfo) -> WindowInfo:
    return WindowInfo(
        handle=int(window.handle),
        title=window.title,
        process_name=window.process_name,
        executable_path=window.executable_path,
        pid=int(window.pid),
    )


def _release(*objects: object) -> None:
    """Drop references on a worker thread; native destructors join their threads."""
    del objects


def create_windows_platform() -> WindowsPlatform:
    """Construct the real backend over the compiled ``jarvis_os`` module.

    Raises:
        NativeUnavailableError: If the extension module is not built.
    """
    module = _native.load("jarvis_os")
    return WindowsPlatform(NativeComponents.from_module(module))


# Design notes:
# - Errors are translated by class name, not by isinstance against the
#   module's classes, so the facade never imports jarvis_os and the fakes in
#   tests can raise plain RuntimeError subclasses with the same names.
# - Hotkey polling is synchronous and cheap (one mutex, no I/O) so the base
#   class's hotkey_events() can call it directly on the loop.
# - The hotkey manager and notifier are created lazily: a text-only session
#   that never registers a hotkey should not spawn a message-loop thread or
#   put an icon in the notification area.
