"""``Platform`` interface: everything a tool may ask the operating system to do.

Tools depend on this class and nothing else from the OS layer, so a macOS
or Linux backend is a new subclass, not a change to any tool. Every method
that could block (Win32, COM, WMI, screen capture) is a coroutine; backends
run the real work off the event loop with ``asyncio.to_thread``.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class WindowInfo:
    """A top-level window as a tool sees it.

    Attributes:
        handle: Opaque platform handle, valid until the window closes.
        title: Window title at the time of the query.
        process_name: Executable file name, e.g. ``"Code.exe"``.
        executable_path: Full path of the executable; empty when the
            process is more privileged than Jarvis and refuses the query.
        pid: Process id of the owning process.
    """

    handle: int
    title: str
    process_name: str
    executable_path: str
    pid: int


@dataclass(frozen=True, slots=True)
class HotkeyEvent:
    """One press of a registered hotkey.

    Attributes:
        hotkey_id: The id :meth:`Platform.register_hotkey` returned.
        hotkey: Canonical spelling of the combination, e.g. ``"ctrl+alt+j"``.
    """

    hotkey_id: int
    hotkey: str


class Platform(ABC):
    """Operating-system services behind one interface.

    Methods are grouped by concern: hotkeys, windows, processes, input,
    notifications, clipboard, system controls, screenshots. All raise a
    :class:`~jarvis.os_integration.errors.PlatformError` subclass on failure.
    """

    async def __aenter__(self) -> Platform:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.close()

    async def close(self) -> None:  # noqa: B027 - a no-op default is the point
        """Release hotkeys, tray icons and threads. Idempotent; default no-op."""

    # --- hotkeys ---------------------------------------------------------

    @abstractmethod
    async def register_hotkey(self, spec: str) -> int:
        """Register a system-wide hotkey such as ``"ctrl+alt+j"``.

        Returns:
            An id that identifies the hotkey in :class:`HotkeyEvent`.

        Raises:
            InvalidHotkeyError: If ``spec`` cannot be parsed.
            PlatformError: If another application already owns the combination.
        """
        ...

    @abstractmethod
    async def unregister_hotkey(self, hotkey_id: int) -> None:
        """Release a hotkey. Unknown ids are ignored."""
        ...

    @abstractmethod
    def poll_hotkeys(self) -> list[HotkeyEvent]:
        """Return and clear the presses since the previous poll (non-blocking)."""
        ...

    async def hotkey_events(self, *, poll_interval: float = 0.05) -> AsyncIterator[HotkeyEvent]:
        """Every hotkey press, in order, for as long as the caller iterates.

        Polling rather than a callback keeps the native thread free of the
        GIL (see ADR-0004); 50 ms is below what a user perceives as lag.
        """
        while True:
            for event in self.poll_hotkeys():
                yield event
            await asyncio.sleep(poll_interval)

    # --- windows ---------------------------------------------------------

    @abstractmethod
    async def active_window(self) -> WindowInfo | None:
        """The focused window, or ``None`` when nothing has focus."""
        ...

    @abstractmethod
    async def list_windows(self) -> list[WindowInfo]:
        """Visible top-level windows, front-most first."""
        ...

    @abstractmethod
    async def focus_window(self, handle: int) -> None:
        """Bring a window to the foreground, restoring it if minimised."""
        ...

    @abstractmethod
    async def focus_window_by_title(self, substring: str) -> WindowInfo:
        """Focus the front-most window whose title contains ``substring``.

        Raises:
            WindowNotFoundError: If no visible window matches.
        """
        ...

    @abstractmethod
    async def minimize_window(self, handle: int) -> None:
        """Minimise a window."""
        ...

    @abstractmethod
    async def maximize_window(self, handle: int) -> None:
        """Maximise a window."""
        ...

    @abstractmethod
    async def restore_window(self, handle: int) -> None:
        """Restore a minimised or maximised window to its normal size."""
        ...

    @abstractmethod
    async def close_window(self, handle: int) -> None:
        """Ask a window to close; the application may still prompt to save."""
        ...

    # --- processes -------------------------------------------------------

    @abstractmethod
    async def launch(self, target: str, arguments: str = "") -> int:
        """Open an executable, document or URL.

        Returns:
            The new process id, or 0 when the target was handed to an
            already-running process (typical for documents and URLs).
        """
        ...

    @abstractmethod
    async def close_process(self, pid: int, *, force: bool = False) -> None:
        """Close a process gracefully, or terminate it with ``force``.

        Raises:
            ProcessNotFoundError: If no such process is running.
            PlatformError: If a graceful close is impossible because the
                process has no window; the caller decides whether to force.
        """
        ...

    @abstractmethod
    async def close_process_by_name(self, name: str, *, force: bool = False) -> int:
        """Apply :meth:`close_process` to every process called ``name``.

        Returns:
            How many processes were addressed.

        Raises:
            ProcessNotFoundError: If none matched.
        """
        ...

    # --- input, notifications, clipboard ---------------------------------

    @abstractmethod
    async def type_text(self, text: str, *, press_enter: bool = False) -> None:
        """Type ``text`` into the focused control; newlines become Enter."""
        ...

    @abstractmethod
    async def notify(self, title: str, message: str) -> None:
        """Show a system notification."""
        ...

    @abstractmethod
    async def get_clipboard_text(self) -> str:
        """Clipboard text, or an empty string when it holds none."""
        ...

    @abstractmethod
    async def set_clipboard_text(self, text: str) -> None:
        """Replace the clipboard contents with ``text``."""
        ...

    # --- system controls -------------------------------------------------

    @abstractmethod
    async def get_volume(self) -> int:
        """Master volume as a percentage, 0-100."""
        ...

    @abstractmethod
    async def set_volume(self, percent: int) -> None:
        """Set the master volume; values outside 0-100 are clamped."""
        ...

    @abstractmethod
    async def is_muted(self) -> bool:
        """Whether the master output is muted."""
        ...

    @abstractmethod
    async def set_muted(self, muted: bool) -> None:
        """Mute or unmute the master output."""
        ...

    @abstractmethod
    async def get_brightness(self) -> int:
        """Built-in display brightness, 0-100.

        Raises:
            UnsupportedFeatureError: If no display offers software brightness.
        """
        ...

    @abstractmethod
    async def set_brightness(self, percent: int) -> None:
        """Set the built-in display brightness; clamped to 0-100.

        Raises:
            UnsupportedFeatureError: If no display offers software brightness.
        """
        ...

    @abstractmethod
    async def lock_screen(self) -> None:
        """Lock the workstation."""
        ...

    # --- screenshots -----------------------------------------------------

    @abstractmethod
    async def screenshot(self, *, all_monitors: bool = False) -> bytes:
        """PNG of the primary monitor, or of the whole desktop with ``all_monitors``."""
        ...

    @abstractmethod
    async def screenshot_window(self, handle: int) -> bytes:
        """PNG of one window's on-screen rectangle.

        Raises:
            WindowNotFoundError: If the handle is stale.
        """
        ...


# Design notes:
# - An ABC rather than a Protocol: the interface is wide, a backend must
#   implement all of it or fail loudly at construction, and the two concrete
#   helpers (hotkey_events, the context manager) are shared behaviour.
# - Percentages and plain ints everywhere, because tools translate spoken
#   requests ("volume to forty percent") and should not know API scales.
# - Graceful versus forced close is an explicit flag so the agent's
#   confirmation policy can treat "kill" as a separate, confirmable action.
