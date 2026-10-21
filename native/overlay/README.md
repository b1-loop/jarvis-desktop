# overlay

Always-on-top status panel: state indicator, live waveform, the user's
transcript and the spoken reply. Dear ImGui on Direct3D 11 in a frameless
Win32 tool window, on a UI thread of its own. Exposed to Python as
`jarvis_overlay`.

## Components

| Header | Type | Role |
| --- | --- | --- |
| `model.hpp` | `AssistantState`, `LevelRing`, `OverlayModel`, `OverlaySnapshot` | thread-safe state: setters from Python, one `snapshot()` per frame |
| `layout.hpp` | `Theme`, `truncateUtf8Tail`, `waveformBars`, `anchorToCorner`, ... | pure decisions about what to draw; no ImGui, fully unit-tested |
| `window.hpp` | `OverlayConfig`, `OverlayWindow` | the UI thread: window, device, swap chain, render loop (pimpl) |
| `errors.hpp` | `OverlayError` | raised when the window or the device cannot be created |
| `src/panel.hpp` | `drawPanel` | private: ImGui draw calls for one frame of a snapshot |

## Window

- `WS_POPUP` with `WS_EX_TOPMOST | WS_EX_TOOLWINDOW | WS_EX_NOACTIVATE |
  WS_EX_LAYERED`, plus `WS_EX_TRANSPARENT` when `clickThrough` is set, so it
  never takes focus, never shows in the taskbar and never steals a click.
- Anchored to a configurable corner of the primary monitor's work area with
  a margin; sizes are 96-DPI logical pixels scaled per monitor
  (`SetThreadDpiAwarenessContext` on the UI thread only, `WM_DPICHANGED`
  rescales fonts and spacing).
- Direct3D 11 hardware device first, WARP (software) as the fallback so a VM
  or a CI runner still works; blit-model swap chain because flip model is
  not allowed on layered windows; whole-window opacity via `LWA_ALPHA`.
- Segoe UI from `%WINDIR%\Fonts` when present, ImGui's embedded font
  otherwise.

## Threads

| Thread | Does | Shares |
| --- | --- | --- |
| UI thread (`OverlayWindow::Impl`) | message loop, one `snapshot()` per frame, ImGui + D3D rendering, DPI changes | model mutex (read) |
| Caller threads (Python) | `setState`, `pushLevel`, `setTranscript`, `setResponse`, `setVisible`, `flashWakeWord` | model mutex (write, sub-microsecond) |

The UI thread never calls back into the caller, so the Python binding
needs the GIL only for the constructor and `close()`, which wait for the
thread. Frame pacing: `activeFps` while the state is not Idle, the wake
flash is visible or something changed in the last second; `idleFps`
otherwise; nothing is rendered while hidden.

## Tests

`native/tests/overlay/`:

- `model_test.cpp` – level ring wraparound and clamping, state transitions
  (Idle clears the waveform), text bounding, visibility, wake-flash decay.
- `layout_test.cpp` – colours and labels, UTF-8 aware truncation, bar
  resampling, DPI scaling and corner placement.
- `window_test.cpp` – creates and destroys the real window, checks that
  setters are cheap and thread-safe. `SKIP`s when no window or device can
  be created (session 0, some headless runners).

Build with `-DJARVIS_BUILD_OVERLAY=OFF` to leave the whole component (and
its Python module) out.
