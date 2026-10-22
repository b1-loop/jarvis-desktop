# 0022. Overlay animations and per-pixel transparency

Date: 2026-10-22
Status: Accepted

## Context

ADR-0011 put the status overlay in a layered Win32 window with whole-window
alpha (`LWA_ALPHA`) and a blit-model swap chain: a dark, uniformly
translucent rectangle. That was enough for a status readout, but the panel
is also the most visible part of the project. The goal for this step: a
HUD-style overlay in which only glowing shapes show over the desktop, a
short boot sequence when Jarvis starts, and state animations that make the
state readable from motion alone (calm when idle, reactive when listening
or speaking).

Constraints carried over from ADR-0011: the window must never take focus or
a click, must not cost the Python side anything beyond a cheap setter, must
keep working on machines without a GPU (WARP) and must stay testable
without a window. New constraints: the idle panel sits on screen all day,
so it must stay dim, slow and cheap; and animations must look the same at
60 fps and at 15 fps, because the frame rate drops when nothing happens.

## Decision

**Per-pixel transparency through DirectComposition, with the old path as a
runtime fallback.** The window is created with `WS_EX_NOREDIRECTIONBITMAP |
WS_EX_TOPMOST | WS_EX_TOOLWINDOW | WS_EX_NOACTIVATE` (plus `WS_EX_LAYERED |
WS_EX_TRANSPARENT` at full layered alpha for click-through). A flip-model
swap chain is created with `CreateSwapChainForComposition` in
`DXGI_ALPHA_MODE_PREMULTIPLIED` and bound to an `IDCompositionVisual`. The
buffer is cleared to transparent; Dear ImGui's standard "over" blending
into a transparent buffer already produces premultiplied colour, so the
ImGui backend is unchanged. If any of that fails, the window is destroyed
and recreated as the ADR-0011 layered window; the choice and the reason are
exposed (`transparent()`, `fallback_reason()`) and logged by the Python
facade. `OverlayConfig.transparent = false` forces the fallback.

**All animation maths in a pure module.** `jarvis/overlay/animation.hpp`
holds easing curves, keyframe tracks, a `BootTimeline` that maps elapsed
time to per-element progress, a state cross-fade, an attack/release level
smoother, a seeded particle field and an `OverlayAnimator` that combines
them per frame. Everything is a function of elapsed time; the only state
carried between frames is the level smoother, whose coefficient
`1 - exp(-dt/tau)` composes exactly across frame intervals. The module has
no ImGui or Win32 dependency and is covered by Catch2 tests (exact values at
keyframes, monotonic easing, weights summing to 1, deterministic particles,
skip behaviour, frame-rate independence). The private `hud.cpp` only
translates its numbers into `ImDrawList` calls.

**Glow from layered strokes in an additive section.** Every "light" stroke
is drawn three or four times, wide and faint to narrow and bright, between
two draw-list callbacks that switch the DX11 blend state to additive and
back. No shader and no blur pass; the whole HUD is a few thousand vertices.

**Boot sequence as geometry interpolation.** The ~2.5 s boot is drawn in a
centred canvas and its last beat interpolates the ring's centre and radius
into the corner panel's (`lerpGeometry`). On the transparent surface the
window spans the work area during the boot, so the glide is pure drawing
and the window is resized exactly twice; on the fallback the window follows
the canvas. The boot is always click-through. `play_boot()` is idempotent
(a launch while running replays it with the separate `replay_boot()`,
ADR-0025);
`skip_boot()` and any state other than Idle jump straight to the handoff so
the panel still glides into place.

**Pacing with a high-resolution waitable timer.** Frames are paced by
`CREATE_WAITABLE_TIMER_HIGH_RESOLUTION` inside `MsgWaitForMultipleObjectsEx`
(60 fps while animating, 15 fps while the idle ring only breathes), not by
millisecond timeouts that the default 15.6 ms timer resolution rounds.

## Alternatives considered

- **Colour-key transparency (`LWA_COLORKEY`)** – trivial to add to the
  existing window, but anti-aliased edges fringe against the key colour and
  glow (partial alpha) is impossible. It would have looked worse than the
  current translucent panel.
- **`UpdateLayeredWindow` with a GDI DIB** – real per-pixel alpha without
  DirectComposition, but every frame is copied from the GPU to a system
  memory bitmap and back to DWM; at 60 fps with a work-area-sized boot
  window that is the opposite of cheap.
- **Animating the window rectangle on the transparent path too** – one code
  path, but a swap-chain resize per frame during the handoff; measurably
  less smooth than drawing inside one large transparent window.
- **A bloom shader (downsample, blur, composite)** – a nicer falloff, but a
  render-to-texture pipeline and custom shaders beside ImGui's backend for
  an effect that layered strokes approximate well at HUD line widths.
- **Animation state inside the renderer** (counters incremented per frame)
  – simpler at first, but frame-rate dependent and untestable without a
  window; exactly what the pure module exists to avoid.
- **Python-driven animation** – rejected for the reasons in ADR-0011: a
  60 fps loop on the interpreter competes with the voice pipeline.

## Consequences

- On Windows 8+ with a working DWM the overlay is truly transparent; in a
  remote session, a VM without composition or when forced, it degrades to
  the ADR-0011 look and says why in the log.
- The renderer is now three private files (`surface.cpp`, `hud_draw.cpp`,
  `hud.cpp`) behind the same pimpl; the public API only grew
  (`playBoot`, `skipBoot`, `transparent`, `fallbackReason`, `frameStats`).
- Measured on the development machine (hardware device, 2560x1600 at 150 %):
  about 1-1.5 ms of UI-thread CPU per frame, 16.7 ms frame interval during
  the boot and while active, 66.7 ms (15 fps) while idle.
- The boot canvas size is a constant in `window.cpp` that
  `scripts/demo_overlay.py` mirrors to compute its capture box; changing
  one without the other only affects the demo's GIF crop.
- Screen capture of the transparent window includes whatever is behind it,
  so the demo's `--record` mode puts a plain backdrop window behind the
  overlay first.
