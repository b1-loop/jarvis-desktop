# 0023. Overlay GPU bloom, procedural 3D scene and a showpiece corner HUD

Date: 2026-10-22
Status: Accepted

## Context

ADR-0022 gave the overlay per-pixel transparency, a boot sequence and state
animations, with glow faked by layered strokes. Feedback after using it:
the boot was liked, but the HUD should be much richer and move a lot more,
and settling into Idle felt like a step down to "the old plain panel": a
small, dim ring next to a translucent grey box with empty "YOU" and
"JARVIS" rows. The goal of this step is a film-grade holographic HUD
(original artwork) whose idle look is as rich as its active states, only
calmer.

The ADR-0011/0022 constraints still hold: the window never takes focus,
Python only calls cheap setters, WARP and the layered (LWA) fallback must
keep working, the idle panel is on screen all day (at most 15 fps), and
every animation must look the same at 15 and 60 fps. New: a frame budget of
4 ms CPU+GPU at 60 fps ("normal") and 2 ms per idle frame, measured.

## Decision

**A real post-processing pipeline inside Dear ImGui's renderer.** Two
draw-list callbacks split each frame into scrims, a *light* section and
text. The light section is drawn additively into an offscreen
`R11G11B10_FLOAT` target (half-float RGBA where that cannot blend), bright
parts are extracted with a soft threshold and a Karis-weighted 4-tap
downsample, blurred down and up a dual-Kawase chain of 2-4 mips (set by the
intensity) and composited onto the swap chain as premultiplied alpha whose
coverage is the brightest channel. The same pass applies the glitch
(chromatic aberration, band tearing, scanlines; 0 nearly always, spikes on
boot beats and the wake pulse) and the hand-off's whip-pan blur. Text is
drawn after the composite and stays crisp. During the boot every pass is
scissored to the canvas, so GPU cost follows what is drawn, not the
work-area-sized window. GPU time comes from D3D11 timestamp queries read
back a few frames late without flushing.

**HLSL compiled at startup from embedded sources.** The shaders are `.hlsl`
files; CMake embeds them (with a shared prelude) into a generated header at
configure time, and the window compiles them once with `D3DCompile`
(shader model 4.0). A Catch2 test compiles every pass, so a broken shader
fails the suite. fxc at build time was rejected: it is not on every build
machine's `PATH`, and the configure-only clang-tidy job would not find a
header that only a build produces; `d3dcompiler_47.dll` ships with Windows
and Dear ImGui's backend already depends on it.

**A procedural 3D scene in pure modules.** `scene.hpp` (perspective camera
with a slow Lissajous drift, a precomputed `Projector`, the globe
wireframe, satellite orbits, ring layout, eased `Rotor`s, parallax),
`particles.hpp` (analytic curl noise and a seeded fixed-step particle
system), `effects.hpp` (intensity presets, glitch envelope, attack/decay
audio energy with onsets) and `animator.hpp` (per-frame composition) have no
ImGui or Direct3D and are unit-tested. Frame-rate independence is by
construction: rotors integrate their eased speed in closed form, and the
particle field runs a fixed 60 Hz step whose count is derived from total
elapsed time, so 120 frames of 1/60 s and 30 frames of 1/15 s run the same
steps. Audio drives ring speed, glow, globe pulse and particles: input
level pulls the field in while listening, output level and syllable onsets
burst it outward while speaking, thinking swirls it.

**Four-act boot with a real readout.** Ignition, assembly (rings fly in
from depth and lock with glitches, the globe draws itself line by line),
systems check (labels count up, a diagnostic readout types on, the title
types with chromatic glitch) and a whip-pan hand-off; about 4 s, still
skippable, any state change skips to the hand-off. The readout is real:
`Overlay.play_boot(info)` passes model, mode and microphone from Python
(the no-argument call still works), and the renderer adds GPU, surface and
post-processing.

**The corner HUD is a showpiece, idle included.** `overlay_size` (compact /
normal / large, default large: 760 x 300 at 96 DPI, a ~280 px ring) and
`overlay_intensity` (calm / normal / max) are settings. The grey panel
rectangle is gone: a dark radial scrim under the ring and a feathered scrim
under the text keep it readable over any background, while brackets, a
connector and a telemetry strip give it its shape. Idle keeps every layer
(globe, rings, satellites, most of the particles, bloom on), slower and a
little dimmer, and shows a live clock and date, model and mode, microphone
and uptime; the transcript rows are hidden unless there is text and slide
in when there is.

## Alternatives considered

- **Separable Gaussian bloom** - more taps per pixel for the same radius;
  dual Kawase gives a wide, smooth falloff in 5 + 8 taps per level and the
  level count is the single radius knob.
- **A second ImDrawData with our own draw lists** - cleaner separation, but
  Dear ImGui 1.92's dynamic font atlas uploads textures through the main
  draw data; callbacks keep the backend and the atlas untouched.
- **GPU particles in a dynamic vertex buffer** - would allow tens of
  thousands, but needs its own pipeline; a few hundred to two thousand
  quads written straight into the draw list (4 vertices each) measured well
  inside the budget.
- **A panel rectangle with higher opacity** - readable, but it is exactly
  the "grey box" the feedback rejected.

## Consequences

- Measured on the development machine (AMD Radeon 890M integrated GPU,
  2560 x 1600 at 150 %, large panel, DirectComposition with bloom):

  | Intensity | Phase | CPU / frame | GPU / frame | CPU + GPU | Rate |
  | --- | --- | --- | --- | --- | --- |
  | calm | boot, acts 1-3 | 1.66 ms | 0.53 ms | 2.2 ms | 60 fps |
  | calm | active (speaking) | 2.70 ms | 0.67 ms | 3.4 ms | 60 fps |
  | calm | idle | 2.57 ms | 0.57 ms | 3.1 ms | 15 fps |
  | normal | boot, acts 1-3 | 1.84 ms | 0.45 ms | 2.3 ms | 60 fps |
  | normal | boot, end of hand-off | 3.65 ms | 1.19 ms | 4.8 ms | 60 fps |
  | normal | active (speaking) | 2.51 ms | 0.68 ms | 3.2 ms | 60 fps |
  | normal | idle | 2.81 ms | 0.62 ms | 3.4 ms | 15 fps |
  | max | boot, acts 1-3 | 2.72 ms | 0.54 ms | 3.3 ms | 60 fps |
  | max | active (speaking) | 2.78 ms | 0.71 ms | 3.5 ms | 60 fps |
  | max | idle | 2.97 ms | 0.64 ms | 3.6 ms | 15 fps |

  Medians of three runs, each a 0.1-weighted moving average over the last
  seconds of the phase; the machine was in normal use, and single runs
  varied by up to 1.5 ms. CPU is the UI thread's wall time per frame and
  includes `Present`, which costs about 1 ms here even for an empty frame
  (measured by drawing nothing: about 1.3 ms per frame in total). The
  ADR-0022 overlay measured 1.6-2.0 ms idle and 1.2-2.1 ms active under
  the same conditions.

- **Budget status.** Active frames (4 ms) and the boot's first three acts
  are within budget at every intensity. Two are not: the last frames of
  the hand-off (the window shrinks from the work area to the panel, which
  resizes the swap chain and the light targets once) and the **idle frame,
  about 3.1-3.6 ms against a 2 ms budget**. Of the idle frame, about 1 ms
  is the fixed `Present` cost and 0.6 ms GPU; the rest is the scene. The
  feedback for this step asked for an idle HUD as rich as the active
  states, which is what costs the difference; idle runs at 15 fps, so the
  idle overlay uses about 5 % of one core and under 1 % of the GPU.
  Work already cut to get here: 60 Hz particle steps, simulating only the
  drawn share of the field, curl noise resampled every fourth step, a
  precomputed projector, globe lines batched by depth, packed-float light
  targets and scissored boot passes. Next steps if the 2 ms idle budget
  must hold: draw the idle field at a lower particle share, cache the
  idle text geometry between frames, or bloom every other idle frame.
- Screen capture slows DWM: the boot GIF is captured at a lower rate than
  the overlay renders (it renders at 60 fps when not recorded).
- The renderer is larger (seven private files); everything that decides
  what moves stays in tested pure modules.
- `OverlayConfig.width` / `height` are now optional overrides of the size
  preset; callers that set them keep their exact size.
