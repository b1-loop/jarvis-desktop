"""Tell the user's voice from Jarvis's own voice coming back through the speakers.

Without acoustic echo cancellation the VAD cannot know who is talking: on
laptop speakers Jarvis's answer reaches the microphone and looks like
speech (the user's log showed "speaking -> listening reason=barge-in" with
nobody talking). The engine reports both levels, though: the RMS of every
captured frame and of every rendered playback block. While Jarvis speaks
and nobody interrupts, the microphone level is the playback level times a
fixed acoustic *coupling* (speaker volume, distance, room). A speech onset
whose input level is no louder than that prediction is echo.

:class:`EchoDetector` keeps a short history of both levels, estimates the
coupling from the playback so far, and answers :meth:`EchoDetector.is_echo`
at a speech onset: the mean input level over the last ``window`` seconds is
compared with ``ratio * coupling * mean output level`` over the same span.
"""

from __future__ import annotations

import statistics
from collections import deque
from dataclasses import dataclass
from typing import Final

DEFAULT_ECHO_RATIO: Final = 2.0
"""Mirrors ``Settings.barge_in_echo_ratio``: the user must be twice as loud as the echo."""

DEFAULT_WINDOW_SECONDS: Final = 0.3
"""Mirrors ``Settings.barge_in_echo_window_ms``."""

SILENT_OUTPUT_RMS: Final = 1e-3
"""Below this the speaker is effectively silent, so nothing can echo."""

MIN_CALIBRATION_SAMPLES: Final = 5
"""Coupling estimates from fewer level pairs than this are not trusted."""

COUPLING_QUANTILE_INDEX: Final = 0
"""Lower quartile of the input/output ratios (index 0 of ``statistics.quantiles``)."""

HISTORY_SECONDS: Final = 10.0
"""Older level pairs are forgotten; the coupling only changes with volume or position."""


@dataclass(frozen=True, slots=True)
class LevelSample:
    """Microphone and speaker RMS at one moment (monotonic seconds)."""

    at: float
    input_rms: float
    output_rms: float


class EchoDetector:
    """Decides whether a speech onset during playback is only Jarvis's echo.

    Args:
        ratio: How many times louder than the predicted echo the microphone
            must be for an onset to count as the user; 0 disables the check.
        window_seconds: History compared at an onset.
    """

    def __init__(
        self, *, ratio: float = DEFAULT_ECHO_RATIO, window_seconds: float = DEFAULT_WINDOW_SECONDS
    ) -> None:
        self._ratio = ratio
        self._window = window_seconds
        self._samples: deque[LevelSample] = deque()

    @property
    def enabled(self) -> bool:
        """False when ``ratio`` is 0: every onset counts as the user."""
        return self._ratio > 0.0

    def reset(self) -> None:
        """Forget the history; called when a new answer starts playing."""
        self._samples.clear()

    def record(self, sample: LevelSample) -> None:
        """Add one level pair (the voice front end samples every few milliseconds)."""
        self._samples.append(sample)
        while self._samples and sample.at - self._samples[0].at > HISTORY_SECONDS:
            self._samples.popleft()

    def coupling(self, before: float) -> float | None:
        """Estimated input/output level ratio of the echo path, from samples before ``before``.

        The lower quartile, not the mean: frames where the user already
        talks only raise the ratio, frames of pure echo sit at the bottom.
        None until enough audible playback has been observed.
        """
        ratios = [
            s.input_rms / s.output_rms
            for s in self._samples
            if s.at < before and s.output_rms >= SILENT_OUTPUT_RMS
        ]
        if len(ratios) < MIN_CALIBRATION_SAMPLES:
            return None
        return statistics.quantiles(ratios, n=4)[COUPLING_QUANTILE_INDEX]

    def is_echo(self, now: float) -> bool:
        """Whether the speech reported at ``now`` is explained by the playback alone.

        Not echo when the check is disabled, when nothing audible was
        playing in the window, or before the coupling could be estimated
        (the echo guard covers that first moment).
        """
        if not self.enabled:
            return False
        start = now - self._window
        window = [s for s in self._samples if start <= s.at <= now]
        output = statistics.fmean(s.output_rms for s in window) if window else 0.0
        coupling = self.coupling(start)
        if output < SILENT_OUTPUT_RMS or coupling is None:
            return False
        heard = statistics.fmean(s.input_rms for s in window)
        return heard < self._ratio * coupling * output


# Design notes:
# - Calibrated rather than the bare "input < k * output": the playback level
#   is a digital level before the volume knob and the microphone level is
#   acoustic, so their absolute ratio says nothing on its own. With
#   headphones the coupling is near zero and every real onset passes; with
#   speakers it is large and echo-only onsets fall under the prediction.
# - The calibration comes for free: the echo guard after the first audio
#   of an answer (barge_in_echo_guard_ms) is exactly a stretch of playback
#   in which the user is not expected to talk yet.
# - Levels are polled, not streamed: the engine exposes the latest RMS
#   only, and a 10 ms poll gives about 30 pairs per 300 ms window, plenty
#   for a mean. Unit tests drive the detector with synthetic traces.
