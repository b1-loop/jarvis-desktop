# 0008. Dear ImGui on DirectX 11 for the status overlay

Date: 2026-10-21
Status: Accepted

## Context

The assistant needs a small always-on-top panel that shows, at a glance,
whether Jarvis is idle, listening, thinking or speaking, a live waveform of
the microphone or playback level, and the current transcript and reply. It
must never steal focus or a click from the application the user is working
in, must look right on mixed-DPI monitors, and must keep up with level
samples arriving every 10 ms without ever slowing down the Python event
loop that drives the conversation.

The Python side already runs asyncio, the Anthropic client, STT and TTS;
anything that puts a GUI event loop or periodic redraw work on that
interpreter competes with them under the GIL.

## Decision

The overlay is C++20: a frameless Win32 tool window (`WS_EX_TOPMOST |
WS_EX_TOOLWINDOW | WS_EX_NOACTIVATE | WS_EX_LAYERED`, optionally
`WS_EX_TRANSPARENT` for click-through) rendered with **Dear ImGui** through
its **Win32 and DirectX 11 backends**, on a **UI thread of its own** owned
by `OverlayWindow`. It is exposed to Python as `jarvis_overlay`.

Python never receives callbacks. It calls **setters** (`set_state`,
`push_level`, `set_transcript`, `set_response`, `set_visible`,
`flash_wake_word`) that update an `OverlayModel` under a mutex and return
in well under a microsecond; the UI thread takes one **snapshot** of that
model per frame and draws it. What to draw (colours, labels, UTF-8 aware
truncation, level-to-bar resampling, corner placement) lives in pure
functions in `layout.hpp` so it is unit-tested without a window or a GPU.

Frame pacing follows activity: about 60 fps while the state is not Idle or
something changed within the last second, 10 fps when static, nothing while
hidden. The Direct3D device is created on hardware first and on WARP as a
fallback so a VM or CI runner still gets a window; if neither works the
constructor throws `OverlayError` and the Python side falls back to
`StubOverlay`.

## Alternatives considered

- **Qt (PySide6 or C++)** – a complete toolkit for a panel that has four
  widgets; a 100+ MB dependency, its own event loop to integrate with
  asyncio, and licensing questions that a portfolio project does not need.
- **Tkinter / Dear PyGui / pystray-style Python UI** – all put the redraw
  loop on the Python interpreter. A 60 fps waveform redrawn in Python
  competes with STT decoding and the LLM stream for the GIL; the first
  stutter in the voice pipeline would come from the status panel. The tray
  icon, which redraws once per state change, can stay in Python.
- **Win32 + GDI/Direct2D by hand** – no third-party code, but text layout,
  DPI handling and anti-aliased shapes become our code to maintain. Dear
  ImGui gives all of that in a few thousand lines compiled as a static
  library and is the de-facto choice for in-process overlays.
- **OpenGL or Vulkan backend** – portable, but a third dependency on
  Windows where Direct3D 11 is in the box and WARP guarantees a software
  path. A Metal or Vulkan backend is a `window.cpp` rewrite when macOS or
  Linux land, not a model or layout change.
- **Callbacks from the UI thread into Python** (e.g. "give me the current
  level") – would require the GIL on the UI thread at frame cadence and
  couple rendering to interpreter pauses. Setters plus snapshot keep the
  dependency one-directional and make the model trivially testable.
- **Per-pixel transparency (DirectComposition + flip-model swap chain)** –
  prettier rounded corners, markedly more machinery. Whole-window alpha
  through `LWA_ALPHA` on a layered window with a blit-model swap chain is
  enough for a status panel.

## Consequences

- The panel costs nothing on the Python side beyond a mutex-protected
  write per update, and cannot be blocked by a slow GPU or driver.
- Dear ImGui keeps one global context per process, so one `OverlayWindow`
  at a time is supported; the composition root creates exactly one.
- The overlay is Windows-only until a second rendering backend exists;
  `JARVIS_BUILD_OVERLAY=OFF` removes the component and its module.
- Dear ImGui is fetched at configure time from the official repository at a
  pinned tag and compiled with its own warning level; `/WX` applies only to
  our sources.
- The dashboard window and tray icon (rest of step 13) read the same
  `OverlaySnapshot` or the Python facade's last state, so the three views
  can never disagree about what Jarvis is doing.
