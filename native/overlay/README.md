# overlay

Always-on-top status HUD: an animated ring that tells the assistant's state
by motion, the user's transcript and the spoken reply, plus a short boot
sequence at startup. Dear ImGui on Direct3D 11 in a frameless Win32 tool
window with per-pixel transparency (DirectComposition), on a UI thread of
its own. Exposed to Python as `jarvis_overlay`. Decisions: ADR-0011 (ImGui
overlay) and ADR-0022 (animations and transparency).

![Boot sequence, then idle, listening, thinking and speaking](../../docs/media/overlay-boot.gif)

Recorded with `uv run python scripts/demo_overlay.py --record docs/media/overlay-boot.gif`
(no audio, synthetic levels, dark backdrop behind the transparent window).

## Components

| Header | Type | Role |
| --- | --- | --- |
| `model.hpp` | `AssistantState`, `LevelRing`, `OverlayModel`, `OverlaySnapshot` | thread-safe state: setters from Python, one `snapshot()` per frame; boot request and skip timestamps |
| `animation.hpp` | `BootTimeline`, `StateBlend`, `LevelSmoother`, `ParticleField`, `OverlayAnimator`, easing, keyframes | every animated number as a pure function of elapsed time; no ImGui, fully unit-tested |
| `layout.hpp` | `Theme`, `themeFor`, `HudGeometry`, `panelGeometry`, `bootGeometry`, `truncateUtf8Tail`, ... | pure decisions about where and in which colours to draw |
| `window.hpp` | `OverlayConfig`, `OverlayWindow` | the UI thread: window, device, surface, render loop (pimpl) |
| `errors.hpp` | `OverlayError` | raised when the window or the device cannot be created |
| `src/surface.hpp` | `RenderSurface` | private: DirectComposition swap chain, or the layered fallback |
| `src/hud_draw.hpp` | glow strokes, rings, ticks, sweep, `BlendModes` | private: ImDrawList primitives |
| `src/hud.hpp` | `HudRenderer` | private: one frame of boot and panel, from an `AnimationFrame` |

## Animation system

Everything that moves is computed in `animation.hpp` from **elapsed time**,
never from frame count, so a frame at 2.0 s looks the same whether the
renderer ran at 60 fps or stalled. The UI thread calls
`OverlayAnimator::advance(snapshot, now)` once per frame and hands the
resulting `AnimationFrame` to `HudRenderer`, which only issues draw calls.

**Boot sequence** (`BootTimeline`, about 2.55 s, played once):

| Time | Beat |
| --- | --- |
| 0.00-0.75 s | a point of light flashes and expands into five concentric rings (segmented arcs at different speeds, a dashed ring, tick bezel); a radial sweep turns |
| 0.55-1.40 s | 96 seeded particles spiral inward and dissolve into the ring |
| 1.00-1.55 s | a horizontal scan line passes; hex fragments flicker in and settle |
| 1.40-1.90 s | "J.A.R.V.I.S." types on letter by letter with a cursor |
| 1.85-2.10 s | "SYSTEMS ONLINE" fades in under a growing rule |
| 2.10-2.55 s | ease-in-out cubic glide of the ring into the corner panel; boot-only elements fade, the panel plate and text fade in |

The rings spin fast during the boot and wind down to exactly their idle
speed (`BootFrame::spin`), so the handoff has no jump. `skipBoot()` (and any
state other than Idle) jumps to the handoff beat instead of the end.

**State animations** cross-fade over 200 ms (`StateBlend`; the weights
always sum to 1, and a re-target mid-fade starts from what is on screen):

| State | Ring | Caption |
| --- | --- | --- |
| Idle | dim, slow breathing core (5.5 s period); 15 fps | STANDBY |
| Wake word | bright ripple outwards from the ring (`wakeRipple`) | WAKE WORD |
| Listening | 72 radial spokes driven by the microphone level history | LISTENING |
| Thinking | two counter-rotating arcs and three orbiting dots with tails | PROCESSING |
| Speaking | a closed waveform ring driven by the playback level | SPEAKING |

Levels pass through `LevelSmoother` (50 ms attack, 300 ms release) and a
square-root perceptual curve; `radialWaveform` mirrors the history around
the ring with the newest sample at the top.

**Glow**: each light stroke is drawn in up to four passes (wide/faint to
narrow/bright) inside an additive blend section switched on and off by
draw-list callbacks into the DX11 backend. Plate and text use normal
blending. Themes: `cyan` (default) and `amber`.

## Window and transparency

- Preferred: `WS_EX_NOREDIRECTIONBITMAP | WS_EX_TOPMOST | WS_EX_TOOLWINDOW |
  WS_EX_NOACTIVATE` with a flip-model swap chain from
  `CreateSwapChainForComposition` (`DXGI_ALPHA_MODE_PREMULTIPLIED`) on a
  DirectComposition visual, cleared to transparent. Click-through adds
  `WS_EX_LAYERED | WS_EX_TRANSPARENT` at full layered alpha.
- Fallback (composition unavailable, or `transparent = false`): the layered
  window with whole-window `LWA_ALPHA` and a blit-model swap chain from
  ADR-0011. The choice is made at creation and reported by `transparent()`
  and `fallbackReason()`.
- During the boot the transparent window covers the work area (always
  click-through); afterwards it shrinks to the panel plus a 20 px invisible
  bleed so glows are not clipped.
- Direct3D 11 hardware device first, WARP as the fallback; sizes are 96-DPI
  logical pixels scaled per monitor; Segoe UI for text and Bahnschrift for
  captions when Windows has them.

## Threads and pacing

| Thread | Does | Shares |
| --- | --- | --- |
| UI thread (`OverlayWindow::Impl`) | message loop, one `snapshot()` per frame, animation, ImGui + D3D rendering | model mutex (read), stats mutex (write) |
| Caller threads (Python) | `setState`, `pushLevel`, `setTranscript`, `setResponse`, `setVisible`, `flashWakeWord`, `playBoot`, `skipBoot` | model mutex (write, sub-microsecond) |

The UI thread never calls back into the caller. Frames are paced by a
high-resolution waitable timer inside `MsgWaitForMultipleObjectsEx`:
`activeFps` (60) while booting, cross-fading, rippling, not Idle or within
a second of a change; `idleFps` (15) while the ring only breathes; nothing
while hidden. `frameStats()` reports smoothed CPU time per frame and frame
interval (about 1-1.5 ms and 16.7 ms during the boot on the development
machine; 66.7 ms when idle).

## Tests

`native/tests/overlay/`:

- `animation_test.cpp` – easing endpoints and monotonicity, keyframe tracks,
  SplitMix64 reference values, deterministic and history-independent
  particles, exact boot keyframes, skip behaviour, cross-fade sums,
  frame-rate independence of the smoother and the animator, frame stats.
- `model_test.cpp` – level ring, state transitions, text bounding, wake
  flash, idempotent boot and skip, auto-skip when leaving Idle.
- `layout_test.cpp` – colours, labels, themes, UTF-8 truncation, bars,
  DPI scaling, corner placement, boot and panel geometry, interpolation.
- `window_test.cpp` – creates the real window (both surfaces), checks
  setters are cheap and thread-safe, the boot plays once and frames are
  measured. `SKIP`s when no window or device can be created.

Build with `-DJARVIS_BUILD_OVERLAY=OFF` to leave the whole component (and
its Python module) out.
