"""Fakes for the native OS objects, matching the Protocols in jarvis.os_integration.windows.

The exception classes carry the same names as the ones ``jarvis_os``
registers, because the facade translates native errors by class name.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from jarvis.os_integration.windows import NativeComponents


class NativeError(RuntimeError):
    pass


class OsError(NativeError):
    pass


class UnsupportedError(OsError):
    pass


class WindowNotFoundError(OsError):
    pass


class ProcessNotFoundError(OsError):
    pass


@dataclass(frozen=True)
class FakeHotkey:
    modifiers: int
    virtual_key: int


@dataclass(frozen=True)
class FakeHotkeyEvent:
    id: int
    hotkey: FakeHotkey


_MODIFIER_BITS = {"alt": 1, "ctrl": 2, "shift": 4, "win": 8}


def fake_parse_hotkey(spec: str) -> FakeHotkey:
    """Minimal parser: modifiers by name, a single-letter key by ord."""
    tokens = [token.strip().lower() for token in spec.split("+")]
    if not spec.strip() or any(not token for token in tokens):
        raise ValueError(f"malformed hotkey '{spec}'")
    modifiers = sum(_MODIFIER_BITS[token] for token in tokens[:-1])
    return FakeHotkey(modifiers, ord(tokens[-1].upper()))


def fake_format_hotkey(hotkey: FakeHotkey) -> str:
    names = [name for name, bit in _MODIFIER_BITS.items() if hotkey.modifiers & bit]
    return "+".join([*names, chr(hotkey.virtual_key).lower()])


@dataclass
class FakeHotkeyManager:
    registered: dict[int, FakeHotkey] = field(default_factory=dict)
    pending: list[FakeHotkeyEvent] = field(default_factory=list)
    next_id: int = 1

    def register_hotkey(self, hotkey: FakeHotkey) -> int:
        if hotkey in self.registered.values():
            raise OsError("RegisterHotKey failed: hotkey already registered")
        hotkey_id = self.next_id
        self.next_id += 1
        self.registered[hotkey_id] = hotkey
        return hotkey_id

    def unregister_hotkey(self, hotkey_id: int) -> None:
        self.registered.pop(hotkey_id, None)

    def poll_events(self) -> list[FakeHotkeyEvent]:
        events, self.pending = self.pending, []
        return events

    def press(self, hotkey_id: int) -> None:
        self.pending.append(FakeHotkeyEvent(hotkey_id, self.registered[hotkey_id]))


@dataclass(frozen=True)
class FakeWindowInfo:
    handle: int
    title: str
    process_name: str = "app.exe"
    executable_path: str = r"C:\apps\app.exe"
    pid: int = 1234


@dataclass
class FakeWindowManager:
    windows: list[FakeWindowInfo] = field(default_factory=list)
    active: FakeWindowInfo | None = None
    calls: list[tuple[str, int]] = field(default_factory=list)

    def active_window(self) -> FakeWindowInfo | None:
        return self.active

    def list_windows(self) -> list[FakeWindowInfo]:
        return list(self.windows)

    def _require(self, handle: int) -> FakeWindowInfo:
        for window in self.windows:
            if window.handle == handle:
                return window
        raise WindowNotFoundError(f"window {handle} no longer exists")

    def focus(self, handle: int) -> None:
        self.active = self._require(handle)
        self.calls.append(("focus", handle))

    def focus_by_title(self, substring: str) -> FakeWindowInfo:
        for window in self.windows:
            if substring.lower() in window.title.lower():
                self.active = window
                return window
        raise WindowNotFoundError(f"no window with '{substring}' in its title")

    def minimize(self, handle: int) -> None:
        self.calls.append(("minimize", self._require(handle).handle))

    def maximize(self, handle: int) -> None:
        self.calls.append(("maximize", self._require(handle).handle))

    def restore(self, handle: int) -> None:
        self.calls.append(("restore", self._require(handle).handle))

    def close(self, handle: int) -> None:
        self.calls.append(("close", self._require(handle).handle))


@dataclass
class FakeProcessControl:
    launched: list[tuple[str, str]] = field(default_factory=list)
    closed: list[tuple[int, bool]] = field(default_factory=list)
    running: dict[int, str] = field(default_factory=dict)

    def launch(self, target: str, arguments: str) -> int:
        self.launched.append((target, arguments))
        pid = 1000 + len(self.launched)
        self.running[pid] = target
        return pid

    def close(self, pid: int, force: bool) -> None:
        if pid not in self.running:
            raise ProcessNotFoundError(f"no process with pid {pid}")
        self.closed.append((pid, force))

    def close_by_name(self, executable_name: str, force: bool) -> int:
        pids = [pid for pid, name in self.running.items() if name == executable_name]
        if not pids:
            raise ProcessNotFoundError(f"no running process named '{executable_name}'")
        for pid in pids:
            self.close(pid, force)
        return len(pids)


@dataclass
class FakeTextInjector:
    typed: list[tuple[str, bool]] = field(default_factory=list)

    def type_text(self, text: str, press_enter: bool) -> None:
        self.typed.append((text, press_enter))


@dataclass
class FakeNotifier:
    shown: list[tuple[str, str]] = field(default_factory=list)

    def notify(self, title: str, message: str) -> None:
        self.shown.append((title, message))


@dataclass
class FakeClipboard:
    content: str = ""

    def text(self) -> str:
        return self.content

    def set_text(self, text: str) -> None:
        self.content = text


@dataclass
class FakeSystemControls:
    level: int = 50
    mute: bool = False
    panel: int | None = 70
    locked: int = 0

    def volume(self) -> int:
        return self.level

    def set_volume(self, percent: int) -> None:
        self.level = max(0, min(100, percent))

    def muted(self) -> bool:
        return self.mute

    def set_muted(self, muted: bool) -> None:
        self.mute = muted

    def brightness(self) -> int:
        if self.panel is None:
            raise UnsupportedError("no display exposes software brightness control")
        return self.panel

    def set_brightness(self, percent: int) -> None:
        if self.panel is None:
            raise UnsupportedError("no display exposes software brightness control")
        self.panel = percent

    def lock_screen(self) -> None:
        self.locked += 1


FAKE_PNG = b"\x89PNG\r\n\x1a\n" + b"fake"


@dataclass
class FakeScreenshotter:
    captures: list[str] = field(default_factory=list)

    def capture_screen(self, all_monitors: bool) -> bytes:
        self.captures.append("all" if all_monitors else "primary")
        return FAKE_PNG

    def capture_window(self, handle: int) -> bytes:
        if handle == 0:
            raise WindowNotFoundError("window 0 no longer exists")
        self.captures.append(f"window:{handle}")
        return FAKE_PNG


@dataclass
class FakeNativeSet:
    """All fakes plus counters for the lazily created components."""

    windows: FakeWindowManager = field(default_factory=FakeWindowManager)
    processes: FakeProcessControl = field(default_factory=FakeProcessControl)
    text_injector: FakeTextInjector = field(default_factory=FakeTextInjector)
    clipboard: FakeClipboard = field(default_factory=FakeClipboard)
    system: FakeSystemControls = field(default_factory=FakeSystemControls)
    screenshotter: FakeScreenshotter = field(default_factory=FakeScreenshotter)
    hotkeys: FakeHotkeyManager = field(default_factory=FakeHotkeyManager)
    notifier: FakeNotifier = field(default_factory=FakeNotifier)
    hotkey_managers_made: int = 0
    notifiers_made: int = 0

    def _make_hotkeys(self) -> FakeHotkeyManager:
        self.hotkey_managers_made += 1
        return self.hotkeys

    def _make_notifier(self) -> FakeNotifier:
        self.notifiers_made += 1
        return self.notifier

    def components(self) -> NativeComponents:
        return NativeComponents(
            windows=self.windows,
            processes=self.processes,
            text_injector=self.text_injector,
            clipboard=self.clipboard,
            system=self.system,
            screenshotter=self.screenshotter,
            make_hotkey_manager=self._make_hotkeys,
            make_notifier=self._make_notifier,
            parse_hotkey=fake_parse_hotkey,
            format_hotkey=fake_format_hotkey,
        )
