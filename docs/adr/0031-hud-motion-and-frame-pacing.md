# 0031. HUD motion: vblank-aligned frame pacing, springs and envelopes

Date: 2026-10-23
Status: Accepted (updates the pacing of 0022 and 0023 and the idle rate of 0022-0024)

## Context

The HUD (ADR-0022 to ADR-0025, ADR-0028) is time-based throughout: every
animation is a function of elapsed time, so a dropped frame cannot change
where anything is. It still did not *look* smooth, and averages hid why.
`scripts/demo_overlay.py` only reported moving averages, so the first step
was to measure what a viewer sees: frame-interval percentiles, hitches
(frames more than 1.5x their scheduled period apart), and how many
refreshes frames missed their even cadence by, from DWM's composition
clock. That showed five separate causes on the development machine (a
240 Hz panel with a dynamic refresh rate):

1. **Timer pacing beats against the refresh.** Frames were paced by a free
   running 16.67 ms timer and presented with `Present(0)`. DWM shows a
   composed window's frame on the next vblank, so on 240 Hz a frame stayed
   up for 4 or 5 refreshes depending on a few hundred microseconds of
   timer jitter (up to 86 % of refreshes off cadence in a listening phase);
   on 60 Hz the timer's phase drifts across the vblank and drops or repeats
   a frame every few seconds.
2. **Messages drew frames.** Every wake-up of `MsgWaitForMultipleObjectsEx`
   drew a frame, so mouse moves (a drag) or stray messages produced bursts
   of frames at random phases (141 fps with 4 hitches in one idle phase).
3. **Changes waited out the idle period.** The idle rate was 15 fps and
   nothing woke the loop when Python changed the state, so a transition
   started up to 67 ms late, the boot up to 130 ms late (p99 66.7 ms, max
   129.6 ms), and the first transition frame was a visible jolt.
4. **Discrete steps in the picture.** The particle field runs a fixed 60 Hz
   step and was drawn without interpolation, so at 72 or 144 fps, or at
   60 fps with jitter, it advanced 0, 1 or 2 steps per frame. The level
   history shifts by one sample per push, so every spoke of the waveform
   and every telemetry bar jumped 20-100 times a second.
5. **Snaps and kinks.** Snapping during a drag moved the panel up to 16 px
   in one frame, a double-click reset teleported it, a state change
   mid-fade restarted the ease from rest (a kink in velocity), a changed
   intensity stepped the bloom, and after a stall (hidden window, locked
   desktop) integrated state leapt the whole gap in one frame.

## Decision

Pace frames on the vblank grid, draw only due frames, wake on change, and
give every motion with a destination a critically damped spring.

- **Vblank-aligned pacing** (`pacing.hpp`, `frame_clock.cpp`). Each frame
  gets a whole number of refreshes (`round(refresh / target fps)`: 4 at
  240 Hz, 2 at 144 Hz, 1 at 60 Hz for 60 fps) and the loop wakes on its
  high-resolution waitable timer 15 % of a refresh after the slot's vblank,
  read from `DwmGetCompositionTimingInfo` with the nominal refresh rate.
  The frame is presented well before the next vblank whatever the jitter;
  `Present(0)` still never blocks. A late loop resynchronises to the next
  slot instead of catching up. The UI thread runs at above-normal priority:
  it sleeps over 90 % of each frame, and a wake-up queued behind background
  work is exactly a missed vblank.
- **Only due frames.** Messages are handled as they arrive but never draw.
  The facade signals an event after every mutation, which brings the next
  frame forward to the next slot.
- **Idle at 30 fps, not 15.** The idle HUD still turns its rings and drifts
  its particles, and at 15 fps that reads as stepping. Measured cost below:
  about +1 % of one core for the whole process.
- **Springs** (`anim::Spring`, `springStep`): closed-form critically damped
  springs (frame-rate independent, re-targetable with velocity kept, exact
  rest). The state cross-fade (`StateBlend`) is now four such springs in
  closed form from the last re-target, so a state that changes mid-fade
  bends towards the new one; settle time 0.35 s (99 %). Snapping while
  dragging, the landing after release and a double-click reset glide on a
  spring (`SnapGlide`): the placement itself stays exact (snapped, clamped,
  reported to Python), only the drawing follows it.
- **Envelopes.** The drawn level history follows the pushed one per sample
  (attack 35 ms, release 160 ms) and the waveform interpolates between
  samples, so speech punches up at once and decays smoothly. A changed
  intensity's bloom gain ramps over 0.3 s.
- **Interpolated particles.** The field is drawn between its last two fixed
  steps (`ParticleSystem::position`), the classic fixed-timestep render.
- **Clamped deltas.** Integrated state (rotors, smoothers, particles,
  springs, the bloom ramp) advances by at most 0.1 s per frame.
- **No allocation per frame** for the model: `snapshotInto` reuses the
  snapshot's buffers.

The boot hand-off, the expand/collapse morph and the HUD/orb morph keep
their eased, choreographed timelines (panel fades keyed to the travel);
they were already eased, and only a reversal mid-morph turns sharply.

## Measurements

`uv run python scripts/demo_overlay.py --stats-json …` (and `--minimize`),
2560 x 1600 at 240 Hz, large panel, normal intensity, DirectComposition
with bloom; medians of two runs (states) and one run (minimize), before is
`main` with only the measurement added. Intervals in ms; off-cadence is
refreshes missed per presented frame.

| Phase | Interval p50 / p99 / max | Hitches | Off-cadence | CPU / GPU per frame |
| --- | --- | --- | --- | --- |
| boot | 16.6 / 66.7 / 129.6 → 16.7 / 33.6 / 33.8 | 2 → 0 | 13 % → 2 % | 1.47 / 0.43 → 1.13 / 0.34 ms |
| idle after the boot | 36.9 / 51.4 / 65.3 → 25.1 / 33.8 / 33.8 | 2 → 0 | 113 % → 5 % | 1.17 / 0.42 → 1.09 / 0.42 ms |
| listening | 16.6 / 17.4 / 42.1 → 16.6 / 19.3 / 19.8 | 0 → 0 | 86 % → 12 % | 1.19 / 0.43 → 0.86 / 0.28 ms |
| thinking | 16.6 / 17.6 / 17.6 → 16.6 / 18.5 / 18.5 | 0 → 0 | 6 % → 7 % | 1.22 / 0.36 → 0.95 / 0.16 ms |
| speaking | 16.7 / 17.2 / 17.3 → 16.6 / 17.3 / 17.3 | 0 → 0 | 5 % → 1 % | 1.37 / 0.33 → 1.02 / 0.20 ms |
| back to idle | 16.7 / 66.3 / 66.3 → 16.7 / 33.6 / 33.6 | 0 → 0 | 3 % → 2 % | 1.07 / 0.40 → 1.03 / 0.21 ms |
| idle, settled | 66.7 / 67.2 / 67.2 → 33.3 / 35.0 / 35.4 | 0 → 0 | 13 % → 3 % | 1.44 / 0.38 → 1.06 / 0.25 ms |
| minimize (shrink) | 16.6 / 29.9 / 29.9 → 16.6 / 17.4 / 17.4 | 1 → 0 | 28 % → 14 % | 0.71 / 0.09 → 0.85 / 0.13 ms |
| orb idle | 66.7 / 67.6 / 67.6 → 33.3 / 36.1 / 36.5 | 0 → 0 | 33 % → 1 % | 1.13 / 0.15 → 1.07 / 0.13 ms |
| orb speaking | 16.6 / 17.3 / 17.7 → 16.6 / 17.1 / 17.1 | 0 → 0 | 6 % → 2 % | 0.99 / 0.12 → 1.01 / 0.13 ms |
| restore | 16.6 / 28.6 / 28.6 → 16.6 / 17.2 / 17.2 | 1 → 0 | 15 % → 7 % | 0.98 / 0.43 → 1.26 / 0.34 ms |

Idle cost (whole process CPU share of one core, the demo script included):
settled idle HUD 1.8 % → 2.9 %, idle orb 2.3 % → 3.1 %. The p99 of 33 ms
in the boot and at the end of a turn is the deliberate switch between the
idle and the active rate, not a hitch. Run-to-run noise is real on a busy
machine: a run with other builds going ends up with 5-15 % off cadence in
the active phases where a quiet one has 1-3 %.

## Alternatives considered

- **Present(1) or a frame-latency waitable swap chain.** Vsync-locked
  presents run at the full refresh (240 fps here) unless the sync interval
  is raised, and `SyncInterval` stops at 4, so 30 fps idle on 240 Hz is out
  of reach; waiting on the waitable object also blocks a thread that has to
  keep pumping messages. A timer that knows the grid gets the same cadence
  for any rate.
- **`IDXGISwapChain::GetFrameStatistics` for the cadence.** It reports
  nothing for a DirectComposition swap chain, so the cadence is computed
  from the present time and DWM's grid instead.
- **Springs for the expand and orb morphs.** A spring's progress is
  front-loaded, which would re-time the choreographed fades (the panel
  text is gone before the ring travels a third). Left eased; only a
  reversal mid-morph is sharp.
- **Adaptive idle rate (15 fps after a minute of idle).** Saves about 1 %
  of a core, but the step back down is itself a visible change of motion
  quality; not worth it at this cost.

## Consequences

- Motion is even at any refresh rate, and transitions start on the next
  vblank. The demo now prints p50/p95/p99/max, hitches and cadence per
  phase, so a regression shows as a number rather than a feeling.
- The HUD costs a little more while idle (30 instead of 15 frames a
  second); still within the ADR-0023 budget per frame.
- A refresh rate DWM does not report (remote sessions) falls back to the
  plain timer, as before.
- The particle field is drawn up to one step (16 ms) behind its
  simulation; nothing else changes position because of it.
