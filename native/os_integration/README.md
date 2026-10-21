# os_integration

Win32 implementation of the platform services tools need. Exposed to Python
as `jarvis_os` and wrapped by `jarvis.os_integration.WindowsPlatform`.

## Components

| Header | Class / function | Role | Win32 API |
| --- | --- | --- | --- |
| `hotkey.hpp` | `parseHotkey`, `formatHotkey` | pure `"ctrl+alt+j"` <-> `Hotkey` conversion, no OS needed | - |
| `hotkey.hpp` | `HotkeyManager` | system-wide hotkeys on a dedicated message-loop thread, presses drained by `pollEvents()` | `RegisterHotKey`, `WM_HOTKEY` |
| `window_manager.hpp` | `WindowManager`, `Win32WindowManager` | active window, Alt+Tab-style listing, focus/minimise/maximise/restore/close | `GetForegroundWindow`, `EnumWindows`, `SetForegroundWindow` + `AttachThreadInput`, `ShowWindow`, `WM_CLOSE` |
| `process_control.hpp` | `ProcessControl`, `Win32ProcessControl` | launch; graceful close or forced terminate by pid or image name | `ShellExecuteExW`, `CreateToolhelp32Snapshot`, `TerminateProcess` |
| `text_injector.hpp` | `TextInjector`, `SendInputTextInjector` | type arbitrary UTF-8 at the cursor, optional Enter | `SendInput` + `KEYEVENTF_UNICODE` |
| `notifier.hpp` | `Notifier`, `BalloonNotifier` | toast notifications through a tray icon | `Shell_NotifyIconW` + `NIF_INFO` |
| `system_controls.hpp` | `SystemControls`, `Win32SystemControls` | master volume and mute, panel brightness, lock screen | `IAudioEndpointVolume`, WMI `WmiMonitorBrightnessMethods`, `LockWorkStation` |
| `screenshot.hpp` | `Screenshotter`, `GdiScreenshotter` | primary monitor, virtual desktop or one window as PNG bytes | `BitBlt` + `CreateDIBSection`, WIC PNG encoder |
| `clipboard.hpp` | `Clipboard`, `Win32Clipboard` | get/set plain text | `OpenClipboard`, `CF_UNICODETEXT` |
| `text_encoding.hpp` | `utf8ToWide`, `wideToUtf8` | the single UTF-8/UTF-16 boundary; invalid input throws | `MultiByteToWideChar` |
| `errors.hpp` | `OsError`, `UnsupportedError`, `WindowNotFoundError`, `ProcessNotFoundError` | hierarchy under `jarvis::NativeError`, mapped 1:1 onto Python classes | - |

Private helpers under `src/` (never included by a public header):
`win32_util.hpp` (RAII handle, COM apartment scope, DPI-awareness scope,
error formatting) and `message_loop_thread.hpp` (a thread that pumps a
message queue and runs tasks on it).

No public header includes `<windows.h>`; window handles cross the API as
`std::uint64_t` and everything stateful uses pimpl.

## Threads

Two Win32 facilities are thread-affine: `RegisterHotKey` delivers
`WM_HOTKEY` to the registering thread, and a window dies when its creating
thread exits. `HotkeyManager` and `BalloonNotifier` therefore each own a
`MessageLoopThread`; calls from Python are marshalled onto it and waited
for, presses are appended to a queue that Python polls. No native thread
ever calls into Python or takes the GIL (same rule as the audio engine).
Everything else runs on the calling thread with per-call COM initialisation.

## Limits (accepted for now)

- **Notifications** use the `Shell_NotifyIcon` balloon, which Windows 10/11
  render as a normal toast. Title is cut at 63 characters, body at 255,
  there are no buttons, and an icon sits in the notification area while the
  notifier lives. WinRT `ToastNotification` would lift those limits but
  needs an AppUserModelID registration and a packaged or shortcut-backed
  identity; see ADR-0008.
- **Brightness** goes through WMI, which only exists for panels with
  software brightness (laptops, tablets). External monitors need DDC/CI
  (`SetVCPFeature`) and are reported as `UnsupportedError`.
- **Screenshots** capture what is on screen: a window overlapped by another
  shows the overlap. Capture runs per-monitor-DPI-aware, so a scaled 4K
  display yields native pixels. Video-rate capture would need DXGI Desktop
  Duplication; a few captures per minute do not.
- **Focus** can be refused by Windows (foreground lock) even after the
  `AttachThreadInput` dance; `focus()` verifies and raises `OsError` rather
  than reporting success it cannot confirm.
- **Launch** returns pid 0 when the shell hands a document or URL to an
  already running application.
- Text injection into windows at a higher integrity level (elevated apps,
  the secure desktop) is blocked by UIPI and surfaces as `OsError`.

## Tests

`native/tests/os/` covers every class. Pure parts (hotkey parser, encoding)
always run. Tests that need a desktop skip without one (`OpenInputDesktop`
probe); tests that change machine state (volume, clipboard) only run with
`JARVIS_NATIVE_STATE_TESTS=1`, never in CI. Lock screen, text injection and
process termination are deliberately not exercised automatically.
