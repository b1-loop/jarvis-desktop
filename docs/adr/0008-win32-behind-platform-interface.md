# 0008. Win32 behind a Platform interface

Date: 2026-10-21
Status: Accepted

## Context

Phase 1 needs computer control: a push-to-talk hotkey, "switch to the
editor", "volume to forty", "take a screenshot", "lock my screen", "type
this". Each of those is a handful of Win32, COM or WMI calls with awkward
rules: some APIs are thread-affine, most want UTF-16, several return
`BOOL` and expect `GetLastError`, and a few (brightness, foreground
focus) simply do not work on every machine.

The project brief says Windows 11 first but keeps macOS and Linux
possible, and it forbids tools from importing concrete implementations
across layers. Tools will be the most numerous code in the project; they
must not know what an `HWND` is.

## Decision

1. **One `Platform` interface in Python** (`jarvis.os_integration.base`)
   is the only thing tools depend on. It is an ABC with async methods,
   plain ints and strings, and small dataclasses (`WindowInfo`,
   `HotkeyEvent`). `WindowsPlatform` implements it over the `jarvis_os`
   pybind11 module through injected Protocols; `StubPlatform` implements
   it in memory for text-only mode and tests. A future macOS backend is
   another subclass and zero tool changes.

2. **Win32 lives in C++ behind small interfaces**, one header/source pair
   per capability (`WindowManager`, `ProcessControl`, `SystemControls`,
   `Screenshotter`, ...). Public headers contain no Win32 types; handles
   cross as `std::uint64_t`, strings as UTF-8, images as PNG bytes.
   Every failure is a subclass of `jarvis::NativeError` and is mapped to
   a Python `PlatformError` subclass by name, so the Python facade never
   imports the extension module.

3. **Thread-affine APIs get their own message loop.** `RegisterHotKey`
   posts `WM_HOTKEY` to the registering thread, and a window is destroyed
   when its creating thread exits. Python's executor threads come and go,
   so `HotkeyManager` and `BalloonNotifier` each own a `MessageLoopThread`
   that pumps messages for the object's lifetime. Requests are marshalled
   onto it and awaited; presses go into a queue that Python polls. As in
   the audio engine (ADR-0003), no native thread ever takes the GIL.

4. **Notifications use the `Shell_NotifyIcon` balloon.** Windows 10 and
   11 render `NIF_INFO` balloons as ordinary toasts in the Action Center.
   It needs no app identity, no WinRT projection and no shortcut with an
   AppUserModelID, so it works from a plain `python.exe`. Accepted limits:
   63/255-character title/body, no buttons, a tray icon while the
   notifier exists.

5. **Brightness uses WMI `WmiMonitorBrightnessMethods`.** It is the
   documented route for panels with software brightness (laptops,
   tablets) and needs no driver-specific code. Machines without such a
   panel raise `UnsupportedError`, which the Python side turns into
   `UnsupportedFeatureError` so a tool can say so instead of pretending.

## Alternatives considered

- **`pywin32` / `ctypes` from Python** – quicker to write, but every
  blocking call would run under the GIL on the event loop thread or need
  its own executor plumbing per call; thread-affine hotkeys would still
  need a dedicated Python thread; and it breaks the project's "C++ is the
  muscle" story. The pybind11 route also gives one exception hierarchy
  and RAII for every handle.
- **A Protocol instead of an ABC for `Platform`** – a Protocol would let
  any duck-typed object pass, but the interface is wide and the two
  shared behaviours (the hotkey event stream, the context manager)
  belong in one place. An ABC fails loudly at construction when a backend
  forgets a method.
- **WinRT `ToastNotificationManager`** – richer toasts with buttons and
  images, but requires an AppUserModelID registered through a Start-menu
  shortcut or package identity, plus the C++/WinRT projection in the
  build. Worth revisiting when the app is packaged.
- **DDC/CI (`SetVCPFeature`) for external monitors** – works on many
  monitors and fails silently on others; a separate feature with its own
  error modes. Out of scope until someone needs it.
- **DXGI Desktop Duplication for screenshots** – the right tool for
  video-rate capture, overkill for a few PNGs per minute; GDI + WIC are
  in every Windows install and need no device.
- **Callbacks into Python for hotkey presses** – would require the
  message-loop thread to acquire the GIL; polling every 50 ms costs
  nothing a user notices and keeps the threading rules uniform.

## Consequences

- Tools are written once against `Platform`; the OS backend can be
  swapped or stubbed per test.
- Everything that blocks runs in `asyncio.to_thread`; the facade is
  tested with fakes in milliseconds and the real module by `native`-marked
  tests that change no machine state.
- Two extra threads exist while hotkeys or notifications are in use;
  both are created lazily and released by `Platform.close()`.
- Focus, brightness and graceful close can legitimately fail; the API
  reports that instead of guessing, and the agent's confirmation policy
  can treat "force" as a separate decision.
- The notification and brightness choices are the two most likely to be
  superseded (by WinRT toasts and DDC/CI); each is isolated in one class.
