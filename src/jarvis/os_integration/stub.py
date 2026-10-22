"""``StubPlatform``: a recording, in-memory :class:`Platform` for text-only mode and tests.

It is a stub in the CLAUDE.md sense: nothing reaches the operating system.
Every call is appended to :attr:`StubPlatform.calls`, state such as volume
or the clipboard lives in plain attributes, and tests can inject hotkey
presses with :meth:`StubPlatform.press`. Use it when the native module is
absent (``--text-only``) or when a test needs a platform that never
touches the machine.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
from typing import Final

from jarvis.os_integration.base import HotkeyEvent, Platform, WindowInfo
from jarvis.os_integration.errors import (
    InvalidHotkeyError,
    PlatformError,
    ProcessNotFoundError,
    WindowNotFoundError,
)

STUB_PNG: Final[bytes] = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
)
"""A valid 1x1 PNG, returned by the stub's screenshot methods."""

_STUB_FIRST_PID: Final = 4000


@dataclass
class StubPlatform(Platform):
    """Records calls and simulates state without touching the OS.

    Attributes:
        calls: Every method call as ``(name, args)``, in order.
        windows: What :meth:`list_windows` returns; set it to script a desktop.
        active: What :meth:`active_window` returns.
        volume: Simulated master volume.
        muted: Simulated mute state.
        brightness: Simulated display brightness.
        clipboard: Simulated clipboard text.
        typed: Every text passed to :meth:`type_text`.
        notifications: Every ``(title, message)`` passed to :meth:`notify`.
        hotkeys: Registered specs keyed by id.
        taken: Specs "another application" owns: registering one raises
            ``PlatformError`` like the real backend, so collision handling
            can be tested.
    """

    calls: list[tuple[str, tuple[object, ...]]] = field(default_factory=list)
    windows: list[WindowInfo] = field(default_factory=list)
    active: WindowInfo | None = None
    volume: int = 50
    muted: bool = False
    brightness: int = 50
    clipboard: str = ""
    typed: list[str] = field(default_factory=list)
    notifications: list[tuple[str, str]] = field(default_factory=list)
    hotkeys: dict[int, str] = field(default_factory=dict)
    taken: set[str] = field(default_factory=set)
    _pending: list[HotkeyEvent] = field(default_factory=list)
    _next_hotkey_id: int = 1
    _next_pid: int = _STUB_FIRST_PID

    def _record(self, name: str, *args: object) -> None:
        self.calls.append((name, args))

    def press(self, hotkey_id: int) -> None:
        """Simulate a press of a registered hotkey (test helper)."""
        self._pending.append(HotkeyEvent(hotkey_id, self.hotkeys[hotkey_id]))

    # --- hotkeys ---------------------------------------------------------

    async def register_hotkey(self, spec: str) -> int:
        """Record the spec and hand out the next id; rejects blank specs."""
        self._record("register_hotkey", spec)
        if not spec.strip():
            raise InvalidHotkeyError("hotkey spec is empty")
        if spec.lower().replace(" ", "") in self.taken:
            raise PlatformError(f"hotkey {spec!r} is already registered by another application")
        hotkey_id = self._next_hotkey_id
        self._next_hotkey_id += 1
        self.hotkeys[hotkey_id] = spec.lower().replace(" ", "")
        return hotkey_id

    async def unregister_hotkey(self, hotkey_id: int) -> None:
        """Forget a registration."""
        self._record("unregister_hotkey", hotkey_id)
        self.hotkeys.pop(hotkey_id, None)

    def poll_hotkeys(self) -> list[HotkeyEvent]:
        """Return presses injected with :meth:`press`."""
        pending, self._pending = self._pending, []
        return pending

    # --- windows ---------------------------------------------------------

    async def active_window(self) -> WindowInfo | None:
        """Return :attr:`active`."""
        self._record("active_window")
        return self.active

    async def list_windows(self) -> list[WindowInfo]:
        """Return a copy of :attr:`windows`."""
        self._record("list_windows")
        return list(self.windows)

    def _find(self, handle: int) -> WindowInfo:
        for window in self.windows:
            if window.handle == handle:
                return window
        raise WindowNotFoundError(f"stub has no window with handle {handle}")

    async def focus_window(self, handle: int) -> None:
        """Make the window with ``handle`` the active one."""
        self._record("focus_window", handle)
        self.active = self._find(handle)

    async def focus_window_by_title(self, substring: str) -> WindowInfo:
        """Make the first title match the active window."""
        self._record("focus_window_by_title", substring)
        needle = substring.lower()
        for window in self.windows:
            if needle in window.title.lower():
                self.active = window
                return window
        raise WindowNotFoundError(f"stub has no window with '{substring}' in its title")

    async def minimize_window(self, handle: int) -> None:
        """Record the call; the handle must exist."""
        self._record("minimize_window", handle)
        self._find(handle)

    async def maximize_window(self, handle: int) -> None:
        """Record the call; the handle must exist."""
        self._record("maximize_window", handle)
        self._find(handle)

    async def restore_window(self, handle: int) -> None:
        """Record the call; the handle must exist."""
        self._record("restore_window", handle)
        self._find(handle)

    async def close_window(self, handle: int) -> None:
        """Remove the window from :attr:`windows`."""
        self._record("close_window", handle)
        self.windows.remove(self._find(handle))

    # --- processes -------------------------------------------------------

    async def launch(self, target: str, arguments: str = "") -> int:
        """Hand out a fake pid."""
        self._record("launch", target, arguments)
        pid = self._next_pid
        self._next_pid += 4
        return pid

    async def close_process(self, pid: int, *, force: bool = False) -> None:
        """Accept pids this stub launched, reject everything else."""
        self._record("close_process", pid, force)
        if pid < _STUB_FIRST_PID or pid >= self._next_pid:
            raise ProcessNotFoundError(f"stub never launched pid {pid}")

    async def close_process_by_name(self, name: str, *, force: bool = False) -> int:
        """Count windows whose process name matches."""
        self._record("close_process_by_name", name, force)
        matches = [w for w in self.windows if w.process_name.lower() == name.lower()]
        if not matches:
            raise ProcessNotFoundError(f"stub has no process named '{name}'")
        return len(matches)

    # --- input, notifications, clipboard ---------------------------------

    async def type_text(self, text: str, *, press_enter: bool = False) -> None:
        """Append to :attr:`typed`."""
        self._record("type_text", text, press_enter)
        self.typed.append(text + ("\n" if press_enter else ""))

    async def notify(self, title: str, message: str) -> None:
        """Append to :attr:`notifications`."""
        self._record("notify", title, message)
        self.notifications.append((title, message))

    async def get_clipboard_text(self) -> str:
        """Return :attr:`clipboard`."""
        self._record("get_clipboard_text")
        return self.clipboard

    async def set_clipboard_text(self, text: str) -> None:
        """Set :attr:`clipboard`."""
        self._record("set_clipboard_text", text)
        self.clipboard = text

    # --- system controls -------------------------------------------------

    async def get_volume(self) -> int:
        """Return :attr:`volume`."""
        self._record("get_volume")
        return self.volume

    async def set_volume(self, percent: int) -> None:
        """Set :attr:`volume`, clamped to 0-100."""
        self._record("set_volume", percent)
        self.volume = max(0, min(100, percent))

    async def is_muted(self) -> bool:
        """Return :attr:`muted`."""
        self._record("is_muted")
        return self.muted

    async def set_muted(self, muted: bool) -> None:
        """Set :attr:`muted`."""
        self._record("set_muted", muted)
        self.muted = muted

    async def get_brightness(self) -> int:
        """Return :attr:`brightness`."""
        self._record("get_brightness")
        return self.brightness

    async def set_brightness(self, percent: int) -> None:
        """Set :attr:`brightness`, clamped to 0-100."""
        self._record("set_brightness", percent)
        self.brightness = max(0, min(100, percent))

    async def lock_screen(self) -> None:
        """Record the call only."""
        self._record("lock_screen")

    # --- screenshots -----------------------------------------------------

    async def screenshot(self, *, all_monitors: bool = False) -> bytes:
        """Return :data:`STUB_PNG`."""
        self._record("screenshot", all_monitors)
        return STUB_PNG

    async def screenshot_window(self, handle: int) -> bytes:
        """Return :data:`STUB_PNG`; the handle must exist."""
        self._record("screenshot_window", handle)
        self._find(handle)
        return STUB_PNG


# Design notes:
# - The stub validates the same preconditions as the real backend (stale
#   handles, unknown processes, empty hotkeys) so a tool tested against it
#   also exercises its error handling, not only its happy path.
# - Returning a real PNG keeps downstream code (vision, file saving) honest
#   without a native build.
