"""Record a few seconds from the default microphone through the native engine.

Produces a 16 kHz mono 16-bit WAV file, which doubles as the smoke test that
"Python can read mic frames": if this script writes a file with a non-zero
peak level, capture, the ring buffer, the worker thread and the pybind11
boundary all work.

Usage::

    uv run python scripts/record_test_audio.py --seconds 3 --output recording.wav
"""

from __future__ import annotations

import argparse
import sys
import time
import wave
from pathlib import Path

import numpy as np

from jarvis.audio.engine import create_native_engine
from jarvis.audio.models import AudioModelPaths
from jarvis.config import Settings

SAMPLE_RATE = 16_000
INT16_FULL_SCALE = 32_767


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--seconds", type=float, default=3.0, help="recording length")
    parser.add_argument("--output", type=Path, default=Path("recording.wav"), help="WAV path")
    parser.add_argument("--models-dir", type=Path, default=Settings().models_dir)
    return parser.parse_args(argv)


def record(models_dir: Path, seconds: float) -> np.ndarray:
    """Capture ``seconds`` of audio and return float32 samples."""
    engine = create_native_engine(AudioModelPaths.in_directory(models_dir), wake_word=False)
    engine.start()
    try:
        engine.set_listening(True)
        time.sleep(seconds)
        engine.set_listening(False)
        return engine.take_captured()
    finally:
        engine.stop()


def write_wav(path: Path, samples: np.ndarray) -> None:
    """Write float32 samples as 16-bit PCM."""
    pcm = (np.clip(samples, -1.0, 1.0) * INT16_FULL_SCALE).astype(np.int16)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes(pcm.tobytes())


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    args = parse_args(argv)
    print(f"recording {args.seconds:.1f} s ... speak now")
    samples = record(args.models_dir, args.seconds)
    write_wav(args.output, samples)
    peak = float(np.max(np.abs(samples))) if samples.size else 0.0
    print(f"wrote {args.output} ({samples.size / SAMPLE_RATE:.2f} s, peak {peak:.3f})")
    return 0 if samples.size else 1


if __name__ == "__main__":
    sys.exit(main())
