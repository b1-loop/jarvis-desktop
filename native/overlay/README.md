# overlay

Always-on-top holographic status HUD: a wireframe globe inside five ring
layers, orbiting satellites, data streams and a particle field that follow
the assistant's state and the audio, the user's transcript and the spoken
reply, plus a four-act boot sequence at startup. Dear ImGui on Direct3D 11
with a GPU bloom pipeline, in a frameless Win32 tool window with per-pixel
transparency (DirectComposition), on a UI thread of its own. Exposed to
Python as `jarvis_overlay`. Decisions: ADR-0011 (ImGui overlay), ADR-0022
(animations and transparency), ADR-0023 (GPU bloom, 3D scene, corner HUD).

![Boot: ignition, assembly, systems check, hand-off](../../docs/media/overlay-boot.gif)

![Corner HUD: idle, listening, thinking, speaking](../../docs/media/overlay-states.gif)

Recorded with `uv run python scripts/demo_overlay.py --record docs/media/overlay-boot.gif
--record-states docs/media/overlay-states.gif` (no audio, synthetic levels, a
dark backdrop window behind the transparent overlay).

## Components

| Header | Type | Role |
| --- | --- | --- |
| `model.hpp` | `AssistantState`, `LevelRing`, `OverlayModel`, `OverlaySnapshot`, `BootReadout` | thread-safe state: setters from Python, one `snapshot()` per frame; boot request, skip and readout |
| `animation.hpp` | `BootTimeline` (four acts), `StateBlend`, `LevelSmoother`, `ParticleField`, easing, keyframes, `FrameStats` | every boot and state number as a pure function of elapsed time |
| `animator.hpp` | `OverlayAnimator`, `AnimationFrame`, `bloomStrength`, `fieldForces`, `transcriptWanted` | combines everything per frame; the only state carried between frames |
| `scene.hpp` | `Camera`, `Projector`, `project`, `buildGlobe`, `lineReveal`, `Orbit`, `RingLayer`, `Rotor`, `parallax` | 3D geometry: perspective camera, globe wireframe, satellites, ring layout, eased rotors |
| `particles.hpp` | `curlNoise`, `ParticleSystem` | seeded fixed-step particle simulation (breathe, pull, burst, swirl) |
| `effects.hpp` | `Intensity`, `intensityScale`, `GlitchSpike`, `glitchAmount`, `AudioReactor` | richness presets, glitch envelope, attack/decay audio energy and onsets |
| `layout.hpp` | `Theme`, `PanelSize`, `panelMetrics`, `HudGeometry`, `formatClock`, `truncateUtf8Head`, ... | pure decisions about where and in which colours to draw |
| `shaders.hpp` | `ShaderSource`, `shaderSources` | the embedded HLSL of the post-processing passes |
| `system_info.hpp` | `defaultMicrophoneName` | Core Audio query for the boot readout |
| `window.hpp` | `OverlayConfig`, `OverlayWindow` | the UI thread: window, device, surface, post-processing, render loop (pimpl) |
| `errors.hpp` | `OverlayError` | raised when the window or the device cannot be created |
| `src/surface.hpp` | `RenderSurface` | private: DirectComposition swap chain, or the layered fallback |
| `src/postfx.hpp` | `PostProcessor`, `GpuTimer` | private: light layer, dual-Kawase bloom, composite; timestamp queries |
| `src/hud_draw.hpp` | glow strokes, ellipses, radial glows, `BlendModes` | private: ImDrawList primitives and the light-section callbacks |
| `src/hud.hpp` | `HudRenderer` | private: one frame; split over `hud.cpp` (order, state layers), `hud_scene.cpp` (globe, rings, satellites, particles), `hud_boot.cpp` (boot acts), `hud_panel.cpp` (corner HUD) |

Everything except `window.hpp`'s implementation and `src/` is free of
ImGui, Win32 and Direct3D and is unit-tested.

## Render pipeline

One `ImDrawData` per frame, rendered by Dear ImGui's own DX11 backend; two
draw-list callbacks split it into three sections:

```
draw list:  [scrims] --beginLight--> [light: globe, rings, particles, title ...] --endLight--> [text]
GPU:        swap chain   light layer (R11G11B10_FLOAT, cleared)                    swap chain
                              |
                              +-> prefilter (soft threshold, Karis 4-tap, 1/2 res)
                              +-> dual-Kawase down x(levels-1) -> up x(levels-1), additive
                              +-> composite onto the swap chain: light + bloom, glitch
                                  (chromatic aberration, tearing, scanlines), whip blur,
                                  soft shoulder, alpha = brightest channel x 0.9 (premultiplied)
```

- `beginLight` remembers the bound back buffer, binds the light layer (sized
  like the back buffer), clears it and switches to additive blending.
  `endLight` runs the passes with its own shaders and states, rebinds the
  back buffer, and a `ResetRenderState` callback restores ImGui's pipeline.
  Text drawn afterwards is never bloomed or aberrated.
- During the boot the window spans the work area; every pass is scissored
  to the canvas (plus a margin for particles and the whip trail), so the GPU
  cost follows what is drawn, not the window size.
- Shaders: `shaders/*.hlsl`, embedded at configure time into a generated
  header and compiled once per window with `D3DCompile` (vs/ps 4.0, so
  feature level 10_0 and WARP run them). The Catch2 suite compiles every
  pass.
- Degradation: `bloom = false`, a shader or target that fails at creation,
  or a light target that fails later all leave a reason in `bloomReason()`
  and the HUD draws its light additively straight onto the swap chain
  (ADR-0022 look). The layered (LWA) fallback surface runs the same bloom.

## Scene graph of effects

| Layer (back to front) | Driven by |
| --- | --- |
| scrims: dark radial disc under the ring, feathered dark rounded rect under the text | panel alpha (no panel rectangle) |
| backdrop glow | intensity, boot flash |
| boot: flash, shock ring, sweep, inward particle stream, scan line, hex glyph runs | `BootFrame` |
| particle field (900 at normal; calm 315, max 1800) | `FieldForces`: idle breathing, listening pull = input level, speaking burst = output level + onsets, thinking swirl; curl noise |
| wireframe globe (7 latitudes, 12 meridians, 2 orbit rings), depth-shaded | `Projector` (drifting camera, globe rotor, axial tilt), audio pulse; boot: drawn line by line |
| 5 ring layers: segmented arcs, ticks, degree labels, data-stream dashes | rotors (state x audio x intensity speed, eased), parallax by depth; boot: fly in from depth and lock with glitches |
| connector, brackets, telemetry bars | panel; telemetry = level history |
| satellites with trails | rotors on tilted orbits, projected |
| core, state layers (spokes, thinking arcs, speaking waveform), wake ripple | cross-fade weights, `AudioEnergy` |
| boot title (light layer: blooms, takes the chromatic glitch) | `BootFrame` |
| text: boot readout, caption, idle readout (clock, date, model, mic, uptime) or transcript | crisp, after the light layer |

## Boot sequence (about 4 s)

| Act | Time | What happens |
| --- | --- | --- |
| 1 ignition | 0.00-0.80 s | a point of light grows, flashes (glitch spike), a shock ring rolls out |
| 2 assembly | 0.80-2.20 s | the five rings fly in from in front of the screen, tilted, and lock one after another (small glitches); the globe draws itself line by line; particles spiral in; glyph runs orbit |
| 3 systems check | 2.20-3.45 s | tick labels count up, a scan line passes, the readout types on (Python's `MODEL`/`MODE`/`MIC` plus `GPU`, `SURFACE`, `POST FX`), "J.A.R.V.I.S." types with chromatic glitch, "ALL SYSTEMS NOMINAL" |
| 4 hand-off | 3.45-4.00 s | everything compresses into the corner HUD with a whip-pan motion blur |

`skipBoot()` and any state other than Idle jump to act 4, so the panel still
whips into place. `playBoot(readout)` is idempotent; the first readout wins.

## Corner HUD

Sizes (`overlay_size`): compact 420 x 132, normal 600 x 216, large 760 x 300
(default) at 96 DPI, text scaled 1.0 / 1.25 / 1.5. Idle keeps every layer of
the active states (globe, rings, satellites, 70 % of the particles, bloom
on), only slower and a little dimmer, and shows a live clock and date, the
model and mode, the microphone and the uptime. The transcript rows slide in
when there is text and stay up for 8 s after Jarvis returns to Idle.

## Threads and pacing

| Thread | Does | Shares |
| --- | --- | --- |
| UI thread (`OverlayWindow::Impl`) | message loop, one `snapshot()` per frame, animation, ImGui + D3D rendering, GPU timing | model mutex (read), stats mutex (write) |
| Caller threads (Python) | setters, `playBoot`, `skipBoot` | model mutex (write, sub-microsecond) |

Frames are paced by a high-resolution waitable timer inside
`MsgWaitForMultipleObjectsEx`: 60 fps while booting, cross-fading,
sounding, sliding text or not Idle; 15 fps while idle. Everything moves by
elapsed time (rotors integrate eased speeds in closed form, the particle
field runs a fixed 60 Hz step counted from total elapsed time), so the
15 fps idle shows the same motion as 60 fps, just sampled less often.

## Budgets (measured)

Budget: 4 ms CPU + GPU per frame at 60 fps ("normal"), 2 ms per idle frame.
On the development machine (AMD Radeon 890M integrated GPU, 2560 x 1600 at
150 %, large panel, DirectComposition with bloom), medians of three runs:

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

Active frames and the boot are within budget at every intensity; the idle
frame (about 3.1-3.6 ms, of which about 1 ms is the fixed `Present` cost)
is over its 2 ms budget, and the last frames of the hand-off spike once
while the window shrinks. ADR-0023 has the analysis and the next steps.
Measure yourself with `uv run python scripts/demo_overlay.py`, which prints
CPU and GPU time per phase.

## Tests

`native/tests/overlay/`: `animation_test.cpp` (easing, keyframes, boot
keyframes, cross-fade, smoothing, frame-rate independence), `boot_test.cpp`
(act boundaries, ring fly-in, readout typing, glitch spikes, whip, skip,
transcript reveal), `scene_test.cpp` (projection of known points, projector
equivalence, globe line counts, line-by-line reveal, rotors, parallax),
`particles_test.cpp` (divergence-free curl noise, determinism, frame-rate
independence, stall handling, pull and burst, bounds), `effects_test.cpp`
(intensity monotonic, glitch bounded, audio energy, bloom and forces),
`shaders_test.cpp` (every pass compiles with D3DCompile, SKIP without the
DLL), `layout_test.cpp`, `model_test.cpp`, and `window_test.cpp` (real
window on both surfaces, bloom on and off, GPU timing, every intensity;
SKIP without a desktop).

Build with `-DJARVIS_BUILD_OVERLAY=OFF` to leave the whole component (and
its Python module) out.
