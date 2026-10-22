"""Show the overlay's boot sequence, then cycle its states with synthetic levels.

Nothing is heard and nothing listens: there is no microphone, no speech and
no model involved. A scripted timeline drives the real native overlay
(``jarvis_overlay``) through IDLE, the wake-word pulse, LISTENING, THINKING
and SPEAKING with made-up level envelopes, and prints the frame timings the
UI thread measured during the boot and while idle.

``--record`` captures the screen region the overlay moves through (from the
centred boot canvas to the corner panel) at about 20 fps and writes an
optimised GIF. While recording, a plain dark backdrop window covers the
work area behind the overlay so the GIF never contains whatever happens to
be open on the desktop.

Usage::

    uv run python scripts/demo_overlay.py
    uv run python scripts/demo_overlay.py --theme amber
    uv run python scripts/demo_overlay.py --record docs/media/overlay-boot.gif
"""

from __future__ import annotations

import argparse
import ctypes
import math
import sys
import time
import tkinter as tk
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from PIL import Image, ImageGrab

from jarvis import _native
from jarvis.ui.overlay import (
    Overlay,
    OverlayConfig,
    OverlayError,
    OverlayState,
    OverlayTheme,
    create_native_overlay,
)
from jarvis.utils.errors import NativeUnavailableError

_PER_MONITOR_AWARE_V2: Final = -4
_SPI_GETWORKAREA: Final = 0x0030
_BASE_DPI: Final = 96.0
# Must match kBootCanvas in native/overlay/src/window.cpp.
_BOOT_CANVAS: Final = 440
_CAPTURE_MARGIN: Final = 40
_BACKDROP_COLOUR: Final = "#04070b"

LEVEL_INTERVAL_S: Final = 0.05
"""20 Hz, the rate the voice pipeline's bridge pushes levels at."""
RECORD_FPS: Final = 20
GIF_WIDTH: Final = 720
GIF_BUDGET_BYTES: Final = 3_000_000
SETTLE_S: Final = 0.4

LevelFn = Callable[[float], float]


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
    """Steadier, slightly louder output level for synthesised speech."""
    return 0.12 + 0.18 * (0.5 + 0.5 * math.sin(t * 9.0)) * (0.7 + 0.3 * math.sin(t * 1.4))


SCRIPT: Final = (
    Beat(OverlayState.IDLE, 1.2),
    Beat(OverlayState.LISTENING, 2.0, speech_envelope, transcript="", response="", wake=True),
    Beat(OverlayState.THINKING, 1.4, transcript="what's on my calendar today"),
    Beat(OverlayState.SPEAKING, 2.2, voice_envelope, response="Two meetings, the first at ten."),
    Beat(OverlayState.IDLE, 1.5),
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--theme", choices=[t.value for t in OverlayTheme], default="cyan")
    parser.add_argument("--record", type=Path, metavar="OUT.gif", help="capture a GIF")
    parser.add_argument(
        "--opaque", action="store_true", help="force the whole-window-alpha fallback"
    )
    return parser.parse_args(argv)


def work_area() -> tuple[int, int, int, int]:
    """Primary monitor work area (left, top, right, bottom) in physical pixels."""
    rect = (ctypes.c_long * 4)()
    ctypes.windll.user32.SystemParametersInfoW(_SPI_GETWORKAREA, 0, ctypes.byref(rect), 0)
    return rect[0], rect[1], rect[2], rect[3]


def capture_box(config: OverlayConfig) -> tuple[int, int, int, int]:
    """Bounding box of the boot canvas and the bottom-right corner panel."""
    left, top, right, bottom = work_area()
    scale = ctypes.windll.user32.GetDpiForSystem() / _BASE_DPI
    half = round(_BOOT_CANVAS * scale / 2)
    cx, cy = (left + right) // 2, (top + bottom) // 2
    margin = round(config.margin * scale)
    return (
        max(cx - half - _CAPTURE_MARGIN, left),
        max(cy - half - _CAPTURE_MARGIN, top),
        min(right - margin + _CAPTURE_MARGIN, right),
        min(bottom - margin + _CAPTURE_MARGIN, bottom),
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
    """Grabs a screen box at a fixed rate into memory."""

    def __init__(self, box: tuple[int, int, int, int]) -> None:
        self._box = box
        self._frames: list[Image.Image] = []
        self._next = 0.0

    def tick(self, now: float) -> None:
        """Grab a frame if one is due."""
        if now >= self._next:
            self._frames.append(ImageGrab.grab(bbox=self._box))
            self._next = now + 1.0 / RECORD_FPS

    def save(self, output: Path) -> int:
        """Write an optimised GIF, shrinking until it fits the budget; returns bytes."""
        output.parent.mkdir(parents=True, exist_ok=True)
        width = GIF_WIDTH
        while True:
            frames = [shrink(frame, width) for frame in self._frames]
            frames[0].save(
                output,
                save_all=True,
                append_images=frames[1:],
                duration=round(1000 / RECORD_FPS),
                loop=0,
                optimize=True,
                disposal=1,
            )
            size = output.stat().st_size
            if size <= GIF_BUDGET_BYTES or width <= GIF_WIDTH // 2:
                return size
            width = int(width * 0.8)


def shrink(frame: Image.Image, width: int) -> Image.Image:
    """Downscale to ``width`` and quantise to an adaptive 64-colour palette."""
    height = round(frame.height * width / frame.width)
    small = frame.resize((width, height), Image.Resampling.LANCZOS)
    return small.quantize(colors=64, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)


def run_for(seconds: float, step: Callable[[float], None]) -> None:
    """Call ``step(elapsed)`` at the level rate for ``seconds``."""
    start = time.perf_counter()
    while (elapsed := time.perf_counter() - start) < seconds:
        step(elapsed)
        time.sleep(LEVEL_INTERVAL_S / 4)


def report(label: str, overlay: Overlay) -> None:
    """Print the UI thread's frame timings and restart the measurement."""
    stats = overlay._native.frame_stats()  # type: ignore[attr-defined]
    print(
        f"{label:<6} frames={stats.frames:<4} cpu/frame={stats.render_ms:.2f} ms "
        f"interval={stats.interval_ms:.1f} ms ({1000 / max(stats.interval_ms, 1e-6):.0f} fps)"
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


def demo(overlay: Overlay, boot_seconds: float, tick: Callable[[], None]) -> None:
    """Boot, measure, then walk through the script."""
    overlay._native.reset_frame_stats()  # type: ignore[attr-defined]
    overlay.play_boot()
    run_for(boot_seconds, lambda _elapsed: tick())
    report("boot", overlay)
    for beat in SCRIPT:
        play_beat(overlay, beat, tick)
        if beat is SCRIPT[0]:
            report("idle", overlay)


def main(argv: list[str] | None = None) -> int:
    """Run the demo; returns a process exit code."""
    args = parse_args(argv)
    if sys.platform != "win32":
        print("The overlay is Windows-only.", file=sys.stderr)
        return 1
    ctypes.windll.user32.SetProcessDpiAwarenessContext(ctypes.c_ssize_t(_PER_MONITOR_AWARE_V2))
    config = OverlayConfig(theme=OverlayTheme(args.theme), transparent=not args.opaque)
    backdrop = Backdrop() if args.record else None
    try:
        overlay = create_native_overlay(config)
    except (NativeUnavailableError, OverlayError) as error:
        print(f"Overlay unavailable: {error}", file=sys.stderr)
        return 1

    boot_seconds = float(_native.load("jarvis_overlay").BOOT_DURATION_SECONDS)
    print(f"per-pixel transparency: {overlay.transparent}")
    recorder = Recorder(capture_box(config)) if args.record else None

    def tick() -> None:
        if backdrop is not None:
            backdrop.pump()
        if recorder is not None:
            recorder.tick(time.perf_counter())

    with overlay:
        run_for(SETTLE_S, lambda _elapsed: tick())
        demo(overlay, boot_seconds, tick)
    if backdrop is not None:
        backdrop.close()
    if recorder is not None and args.record is not None:
        size = recorder.save(args.record)
        print(f"wrote {args.record} ({size / 1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# Design notes:
# - The demo drives the public facade (Overlay), not the raw native window,
#   except for frame_stats(), which only the demo and tests need.
# - Levels are pushed at 20 Hz like OverlayBridge does, so what the GIF shows
#   is what the voice pipeline produces, smoothing and all.
# - Recording happens on the same thread as the script; ImageGrab is the
#   slow part, which is why the GIF runs at 20 fps while the overlay itself
#   keeps rendering at 60 fps on its own thread.
