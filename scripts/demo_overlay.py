"""Show the overlay's boot sequence, then cycle its states with synthetic levels.

Nothing is heard and nothing listens: there is no microphone, no speech and
no model involved. A scripted timeline drives the real native overlay
(``jarvis_overlay``) through its four-act boot (with a sample diagnostic
readout), IDLE, the wake-word pulse, LISTENING, THINKING and SPEAKING with
made-up level envelopes, including loud speech bursts, and prints the
frame timings the UI thread measured (CPU per frame and GPU time from
timestamp queries) for each phase.

``--record`` captures the screen region the overlay moves through during
the boot (from the centred canvas to the corner HUD) at about 20 fps and
writes an optimised GIF; ``--record-states`` captures only the corner HUD
while the states cycle. While recording, a plain dark backdrop window covers
the work area behind the overlay so the GIFs never contain whatever happens
to be open on the desktop.

``--interactive`` scripts the interactive HUD (ADR-0024) instead: the
hover controls appear, a request is typed into the text field, the typed
line is injected as a ``SubmitText`` event and read back the way the
overlay bridge reads it, a scripted answer plays (no LLM, no speech), and
the HUD expands to its conversation view and collapses again. It uses the
window's demo hooks, so the real mouse and keyboard focus are never
touched; ``--record-interactive`` captures it.

Usage::

    uv run python scripts/demo_overlay.py
    uv run python scripts/demo_overlay.py --interactive
    uv run python scripts/demo_overlay.py --record-interactive docs/media/overlay-interactive.gif
    uv run python scripts/demo_overlay.py --theme amber --intensity max
    uv run python scripts/demo_overlay.py --record docs/media/overlay-boot.gif
    uv run python scripts/demo_overlay.py --record-states docs/media/overlay-states.gif
"""

from __future__ import annotations

import argparse
import ctypes
import math
import sys
import time
import tkinter as tk
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Final

from PIL import Image, ImageGrab

from jarvis import _native
from jarvis.ui.hud_protocol import HistoryRole, SubmitText
from jarvis.ui.overlay import (
    Overlay,
    OverlayConfig,
    OverlayError,
    OverlayIntensity,
    OverlaySize,
    OverlayState,
    OverlayTheme,
    create_native_overlay,
)
from jarvis.utils.errors import NativeUnavailableError

_PER_MONITOR_AWARE_V2: Final = -4
_SPI_GETWORKAREA: Final = 0x0030
_BASE_DPI: Final = 96.0
# Must match kBootCanvas, kBleed and the PanelMetrics presets in native/overlay.
_BOOT_CANVAS: Final = 520
_BLEED: Final = 44
_PANEL_SIZES: Final = {
    OverlaySize.COMPACT: (420, 132),
    OverlaySize.NORMAL: (600, 216),
    OverlaySize.LARGE: (760, 300),
}
_CAPTURE_MARGIN: Final = 40
_BACKDROP_COLOUR: Final = "#04070b"

LEVEL_INTERVAL_S: Final = 0.05
"""20 Hz, the rate the voice pipeline's bridge pushes levels at."""
RECORD_FPS: Final = 20
STATES_FPS: Final = 6
"""The states GIF is longer, so it trades frame rate for resolution."""
GIF_WIDTH: Final = 720
GIF_MIN_WIDTH: Final = 360
GIF_BUDGET_BYTES: Final = 3_800_000
STATES_BUDGET_BYTES: Final = 1_000_000
"""The pre-commit hook rejects new files over 1 MB, and the states GIF is new."""
GIF_COLOURS: Final = 64
SETTLE_S: Final = 0.4
SAMPLE_READOUT: Final = {
    "MODEL": "qwen2.5:7b",
    "MODE": "CLOUD",
    "MIC": "Microphone (demo, nothing is recorded)",
}
_EXPANDED: Final = (900, 640)
"""Must match kExpandedWidth/kExpandedHeight in native/overlay/hud_layout.hpp."""
TYPING_INTERVAL_S: Final = 0.07
TYPED_REQUEST: Final = "Vad är klockan i Tokyo?"
TYPED_REPLY: Final = "Klockan är 21:14 i Tokyo."
SAMPLE_HISTORY: Final = (
    (HistoryRole.USER, "what's on my calendar today"),
    (HistoryRole.TOOL, "calendar_agenda · 118 ms"),
    (HistoryRole.JARVIS, "Two meetings: stand-up at ten and a design review at three."),
)

LevelFn = Callable[[float], float]
Box = tuple[int, int, int, int]


@dataclass(frozen=True, slots=True)
class Beat:
    """One step of the scripted demo after the boot."""

    state: OverlayState
    seconds: float
    level: LevelFn | None = None
    transcript: str | None = None
    response: str | None = None
    wake: bool = False


def speech_envelope(t: float) -> float:
    """Syllable-like bursts: a fast carrier under a slower phrase envelope."""
    syllables = 0.5 + 0.5 * math.sin(t * 17.0)
    phrase = 0.5 + 0.5 * math.sin(t * 2.3)
    return 0.04 + 0.32 * syllables * phrase


def voice_envelope(t: float) -> float:
    """Synthesised speech with a loud burst every ~1.1 s (the particles' cue)."""
    base = 0.12 + 0.18 * (0.5 + 0.5 * math.sin(t * 9.0)) * (0.7 + 0.3 * math.sin(t * 1.4))
    burst = max(0.0, math.sin(t * 5.7)) ** 8
    return min(1.0, base + 0.75 * burst)


SCRIPT: Final = (
    Beat(OverlayState.IDLE, 2.0),
    Beat(OverlayState.LISTENING, 2.2, speech_envelope, transcript="", response="", wake=True),
    Beat(OverlayState.THINKING, 1.6, transcript="what's on my calendar today"),
    Beat(OverlayState.SPEAKING, 3.0, voice_envelope, response="Two meetings, the first at ten."),
    Beat(OverlayState.IDLE, 1.5),
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--theme", choices=[t.value for t in OverlayTheme], default="cyan")
    parser.add_argument(
        "--intensity", choices=[i.value for i in OverlayIntensity], default="normal"
    )
    parser.add_argument("--size", choices=[s.value for s in OverlaySize], default="large")
    parser.add_argument("--record", type=Path, metavar="OUT.gif", help="capture the boot")
    parser.add_argument(
        "--record-states", type=Path, metavar="OUT.gif", help="capture the states cycle"
    )
    parser.add_argument(
        "--opaque", action="store_true", help="force the whole-window-alpha fallback"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="scripted HUD interaction (hover controls, typing, expanded view) with timings",
    )
    parser.add_argument(
        "--record-interactive", type=Path, metavar="OUT.gif", help="capture the interaction"
    )
    return parser.parse_args(argv)


def work_area() -> Box:
    """Primary monitor work area (left, top, right, bottom) in physical pixels."""
    rect = (ctypes.c_long * 4)()
    ctypes.windll.user32.SystemParametersInfoW(_SPI_GETWORKAREA, 0, ctypes.byref(rect), 0)
    return rect[0], rect[1], rect[2], rect[3]


def dpi_scale() -> float:
    """System DPI scale (1.0 at 96 DPI)."""
    return float(ctypes.windll.user32.GetDpiForSystem()) / _BASE_DPI


def panel_box(config: OverlayConfig) -> Box:
    """The bottom-right corner HUD plus its bleed, where the states play."""
    left, top, right, bottom = work_area()
    scale = dpi_scale()
    width, height = _PANEL_SIZES[config.size]
    margin = round(config.margin * scale)
    bleed = round(_BLEED * scale)
    return (
        max(right - margin - round(width * scale) - bleed, left),
        max(bottom - margin - round(height * scale) - bleed, top),
        min(right - margin + bleed, right),
        min(bottom - margin + bleed, bottom),
    )


def boot_box(config: OverlayConfig) -> Box:
    """Bounding box of the centred boot canvas and the corner HUD."""
    left, top, right, bottom = work_area()
    half = round(_BOOT_CANVAS * dpi_scale() / 2)
    cx, cy = (left + right) // 2, (top + bottom) // 2
    panel = panel_box(config)
    return (
        max(min(cx - half - _CAPTURE_MARGIN, panel[0]), left),
        max(min(cy - half - _CAPTURE_MARGIN, panel[1]), top),
        min(max(cx + half + _CAPTURE_MARGIN, panel[2]), right),
        min(max(cy + half + _CAPTURE_MARGIN, panel[3]), bottom),
    )


class Backdrop:
    """A plain dark window over the work area, behind the topmost overlay."""

    def __init__(self) -> None:
        left, top, right, bottom = work_area()
        self._root = tk.Tk()
        self._root.overrideredirect(True)
        self._root.configure(background=_BACKDROP_COLOUR)
        # Topmost so always-on-top desktop widgets stay out of the GIF; the
        # overlay is created afterwards and therefore sits above it.
        self._root.attributes("-topmost", True)
        self._root.geometry(f"{right - left}x{bottom - top}+{left}+{top}")
        self._root.update()

    def pump(self) -> None:
        """Process Tk events so the window keeps painting."""
        self._root.update()

    def close(self) -> None:
        """Destroy the window."""
        self._root.destroy()


class Recorder:
    """Grabs a screen box at a fixed rate into memory while it is armed."""

    def __init__(self, box: Box, fps: int = RECORD_FPS, budget: int = GIF_BUDGET_BYTES) -> None:
        self._box = box
        self._fps = fps
        self._budget = budget
        self._frames: list[Image.Image] = []
        self._next = 0.0
        self.armed = False

    def tick(self, now: float) -> None:
        """Grab a frame if one is due and the recorder is armed."""
        if self.armed and now >= self._next:
            self._frames.append(ImageGrab.grab(bbox=self._box))
            self._next = now + 1.0 / self._fps

    def save(self, output: Path) -> int:
        """Write an optimised GIF, shrinking until it fits the budget; returns bytes."""
        output.parent.mkdir(parents=True, exist_ok=True)
        width = min(GIF_WIDTH, self._frames[0].width)
        while True:
            palette = shrink(self._frames[0], width)
            frames = [palette, *(shrink(frame, width, palette) for frame in self._frames[1:])]
            frames[0].save(
                output,
                save_all=True,
                append_images=frames[1:],
                duration=round(1000 / self._fps),
                loop=0,
                optimize=True,
                disposal=1,
            )
            size = output.stat().st_size
            if size <= self._budget or width <= GIF_MIN_WIDTH:
                return size
            width = int(width * 0.85)


def shrink(frame: Image.Image, width: int, palette: Image.Image | None = None) -> Image.Image:
    """Downscale to ``width`` and quantise, to ``palette``'s colours when given.

    One palette for the whole GIF keeps unchanged pixels identical between
    frames, which is what lets the GIF encoder store only what moved.
    """
    height = round(frame.height * width / frame.width)
    small = frame.resize((width, height), Image.Resampling.LANCZOS)
    if palette is not None:
        return small.quantize(palette=palette, dither=Image.Dither.NONE)
    return small.quantize(
        colors=GIF_COLOURS, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE
    )


def run_for(seconds: float, step: Callable[[float], None]) -> None:
    """Call ``step(elapsed)`` at the level rate for ``seconds``."""
    start = time.perf_counter()
    while (elapsed := time.perf_counter() - start) < seconds:
        step(elapsed)
        time.sleep(LEVEL_INTERVAL_S / 4)


def report(label: str, overlay: Overlay) -> None:
    """Print the UI thread's frame timings and restart the measurement."""
    stats = overlay._native.frame_stats()  # type: ignore[attr-defined]
    fps = 1000 / max(stats.interval_ms, 1e-6)
    print(
        f"{label:<9} frames={stats.frames:<4} cpu/frame={stats.render_ms:.2f} ms "
        f"gpu/frame={stats.gpu_ms:.2f} ms interval={stats.interval_ms:.1f} ms ({fps:.0f} fps)"
    )
    overlay._native.reset_frame_stats()  # type: ignore[attr-defined]


def play_beat(overlay: Overlay, beat: Beat, tick: Callable[[], None]) -> None:
    """Apply one scripted step and feed its level envelope."""
    if beat.transcript is not None:
        overlay.set_transcript(beat.transcript)
    if beat.response is not None:
        overlay.set_response(beat.response)
    if beat.wake:
        overlay.wake_pulse()
    overlay.set_state(beat.state)
    last = -LEVEL_INTERVAL_S

    def step(elapsed: float) -> None:
        nonlocal last
        if beat.level is not None and elapsed - last >= LEVEL_INTERVAL_S:
            overlay.push_level(beat.level(elapsed))
            last = elapsed
        tick()

    run_for(beat.seconds, step)


def demo(
    overlay: Overlay,
    boot_seconds: float,
    tick: Callable[[], None],
    recorders: tuple[Recorder | None, Recorder | None],
) -> None:
    """Boot, measure, then walk through the script, measuring every state."""
    boot_recorder, states_recorder = recorders
    overlay._native.reset_frame_stats()  # type: ignore[attr-defined]
    if boot_recorder is not None:
        boot_recorder.armed = True
    overlay.play_boot(SAMPLE_READOUT)
    run_for(boot_seconds + 0.6, lambda _elapsed: tick())
    if boot_recorder is not None:
        boot_recorder.armed = False
    report("boot", overlay)
    if states_recorder is not None:
        states_recorder.armed = True
    for beat in SCRIPT:
        play_beat(overlay, beat, tick)
        report(beat.state.value.lower(), overlay)


def interactive_box(config: OverlayConfig) -> Box:
    """The corner HUD and the default expanded panel that grows out of it."""
    left, top, _right, _bottom = work_area()
    scale = dpi_scale()
    panel = panel_box(config)
    width = round((_EXPANDED[0] + 2 * _BLEED) * scale)
    height = round((_EXPANDED[1] + 2 * _BLEED) * scale)
    return (max(panel[2] - width, left), max(panel[3] - height, top), panel[2], panel[3])


def type_into_field(overlay: Overlay, text: str, tick: Callable[[], None]) -> None:
    """Show ``text`` appearing in the field one character at a time (demo hook)."""
    for count in range(1, len(text) + 1):
        overlay._native.set_demo(hover=True, input=text[:count])  # type: ignore[attr-defined]
        run_for(TYPING_INTERVAL_S, lambda _elapsed: tick())


def answer(overlay: Overlay, request: str, reply: str, tick: Callable[[], None]) -> None:
    """What the session does with a typed request, scripted: no LLM, no speech."""
    overlay.inject_ui_event(SubmitText(request))
    submitted = overlay.poll_ui_events()  # exactly what the bridge would receive
    print(f"bridge received: {submitted}")
    overlay._native.set_demo(hover=True, input="")  # type: ignore[attr-defined]
    overlay.set_transcript(request)
    overlay.add_history(HistoryRole.USER, request)
    overlay.set_state(OverlayState.LISTENING)
    overlay.set_state(OverlayState.THINKING)
    run_for(1.0, lambda _elapsed: tick())
    overlay.add_history(HistoryRole.TOOL, "current_time · 4 ms")
    overlay.set_response(reply)
    overlay.set_state(OverlayState.SPEAKING)

    def speak(elapsed: float) -> None:
        overlay.push_level(voice_envelope(elapsed))
        tick()

    run_for(1.6, speak)
    overlay.add_history(HistoryRole.JARVIS, reply)
    overlay.set_state(OverlayState.IDLE)


def interactive_demo(overlay: Overlay, tick: Callable[[], None]) -> None:
    """Hover, type a request, answer it, expand, collapse; timings per phase."""
    native = overlay._native  # demo hooks and frame stats live on the native window
    native.reset_frame_stats()  # type: ignore[attr-defined]
    run_for(2.0, lambda _elapsed: tick())
    report("idle", overlay)
    native.set_demo(hover=True)  # type: ignore[attr-defined]
    run_for(2.0, lambda _elapsed: tick())
    report("hover", overlay)
    type_into_field(overlay, TYPED_REQUEST, tick)
    answer(overlay, TYPED_REQUEST, TYPED_REPLY, tick)
    report("typing", overlay)
    overlay.set_expanded(True)
    run_for(0.6, lambda _elapsed: tick())
    native.reset_frame_stats()  # type: ignore[attr-defined]
    run_for(2.5, lambda _elapsed: tick())
    report("expanded", overlay)
    overlay.set_expanded(False)
    native.set_demo(hover=False)  # type: ignore[attr-defined]
    run_for(1.5, lambda _elapsed: tick())
    report("collapse", overlay)


def run_interactive(args: argparse.Namespace, config: OverlayConfig) -> int:
    """The ``--interactive`` / ``--record-interactive`` mode."""
    backdrop = Backdrop() if args.record_interactive is not None else None
    try:
        overlay = create_native_overlay(replace(config, boot_animation=False))
    except (NativeUnavailableError, OverlayError) as error:
        print(f"Overlay unavailable: {error}", file=sys.stderr)
        return 1
    recorder = Recorder(interactive_box(config), STATES_FPS, STATES_BUDGET_BYTES)

    def tick() -> None:
        if backdrop is not None:
            backdrop.pump()
        recorder.tick(time.perf_counter())

    with overlay:
        for line in SAMPLE_HISTORY:
            overlay.add_history(*line)
        run_for(SETTLE_S, lambda _elapsed: tick())
        # Armed only once the HUD is on screen: the GIF's palette comes from
        # the first frame, and a frame of bare backdrop has no HUD colours.
        recorder.armed = args.record_interactive is not None
        interactive_demo(overlay, tick)
    if backdrop is not None:
        backdrop.close()
    if args.record_interactive is not None:
        size = recorder.save(args.record_interactive)
        print(f"wrote {args.record_interactive} ({size / 1e6:.2f} MB)")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Run the demo; returns a process exit code."""
    args = parse_args(argv)
    if sys.platform != "win32":
        print("The overlay is Windows-only.", file=sys.stderr)
        return 1
    ctypes.windll.user32.SetProcessDpiAwarenessContext(ctypes.c_ssize_t(_PER_MONITOR_AWARE_V2))
    config = OverlayConfig(
        theme=OverlayTheme(args.theme),
        intensity=OverlayIntensity(args.intensity),
        size=OverlaySize(args.size),
        transparent=not args.opaque,
    )
    if args.interactive or args.record_interactive is not None:
        return run_interactive(args, config)
    return run_states(args, config)


def run_states(args: argparse.Namespace, config: OverlayConfig) -> int:
    """The default mode: boot, then the four states, optionally recorded."""
    recording = args.record is not None or args.record_states is not None
    backdrop = Backdrop() if recording else None
    try:
        overlay = create_native_overlay(config)
    except (NativeUnavailableError, OverlayError) as error:
        print(f"Overlay unavailable: {error}", file=sys.stderr)
        return 1

    boot_seconds = float(_native.load("jarvis_overlay").BOOT_DURATION_SECONDS)
    print(f"per-pixel transparency: {overlay.transparent}, bloom: {overlay.bloom}")
    boot_recorder = Recorder(boot_box(config)) if args.record else None
    states_recorder = (
        Recorder(panel_box(config), STATES_FPS, STATES_BUDGET_BYTES) if args.record_states else None
    )

    def tick() -> None:
        if backdrop is not None:
            backdrop.pump()
        now = time.perf_counter()
        for recorder in (boot_recorder, states_recorder):
            if recorder is not None:
                recorder.tick(now)

    with overlay:
        run_for(SETTLE_S, lambda _elapsed: tick())
        demo(overlay, boot_seconds, tick, (boot_recorder, states_recorder))
    if backdrop is not None:
        backdrop.close()
    for recorder, output in ((boot_recorder, args.record), (states_recorder, args.record_states)):
        if recorder is not None and output is not None:
            size = recorder.save(output)
            print(f"wrote {output} ({size / 1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# Design notes:
# - The demo drives the public facade (Overlay), not the raw native window,
#   except for frame_stats(), which only the demo and tests need.
# - Levels are pushed at 20 Hz like OverlayBridge does, so what the GIF shows
#   is what the voice pipeline produces, smoothing and all; the speaking
#   envelope adds loud bursts so the particle field's outward kick shows.
# - Recording happens on the same thread as the script; ImageGrab is the
#   slow part, which is why the GIFs run at 20 fps while the overlay itself
#   keeps rendering at 60 fps on its own thread. Two recorders are armed in
#   turn so one run produces both the boot and the states GIF.
# - The capture boxes mirror native constants (boot canvas, bleed, panel
#   presets); changing those without these only affects the GIF crop.
