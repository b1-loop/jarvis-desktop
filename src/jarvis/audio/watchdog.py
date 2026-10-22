"""Capture watchdog: notices a microphone that went quiet and reopens it.

A WASAPI capture stream can stop delivering audio without an error: after a
long system stall or a resume from sleep its five-second read timeout ends
miniaudio's capture loop, a device can be invalidated, or an endpoint can be
re-enumerated. The process stays alive and nothing reacts to "Hey Jarvis"
any more. This watchdog is the cure (see docs/adr/0027).

It watches two signals:

* the native sample counter, which grows every device period (about 10 ms)
  even in silence, so a counter that stays flat for ``stall_seconds`` means
  the stream is dead, never that the room is quiet;
* device notifications from miniaudio (stopped, interrupted, rerouted).

Recovery reopens the current default endpoint, with exponential backoff up
to a cap, and tells the user once per outage when it keeps failing.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Final, Protocol

from jarvis.utils.logging import get_logger

log = get_logger(__name__)

MICROPHONE_LOST: Final = "Microphone stopped; trying again."
"""Told to the user once an outage has outlasted a few restart attempts."""

MICROPHONE_BACK: Final = "Microphone is working again."
"""Told to the user when an outage they were told about has ended."""

NO_SAMPLES: Final = "no_samples"
"""Stall reason: the sample counter stopped growing without any notification."""

Clock = Callable[[], float]
"""Monotonic seconds; injected so tests drive time by hand."""

Notifier = Callable[[str], Awaitable[None]]
"""Shows ``text`` to the user (a system notification in voice mode)."""

DeviceName = Callable[[], str | None]
"""Name of the current default microphone, for log context only."""


class CaptureControl(Protocol):
    """What the watchdog needs from the capture engine."""

    def samples_received(self) -> int:
        """Samples the device has delivered so far; monotonic."""

    async def restart_capture(self) -> bool:
        """Reopen the default capture device; False when the engine is not running."""


@dataclass(frozen=True, slots=True)
class WatchdogConfig:
    """Timing of stall detection and recovery.

    Attributes:
        stall_seconds: A flat sample counter for this long is a stall. Frames
            arrive every ~10 ms, so 2.5 s is 250 missed periods: no false
            alarm, and still well before a user gives up on the wake word.
        lost_grace_seconds: After a "stopped" notification, how long frames
            may take to resume before a restart. miniaudio stops the device
            itself while rerouting to a new default, then restarts it.
        check_interval_seconds: How often the counter is read.
        initial_backoff_seconds: Wait after the first failed attempt.
        max_backoff_seconds: The cap; the watchdog keeps trying at this
            pace for as long as the outage lasts (a USB microphone may be
            plugged back in an hour later).
        notify_after_attempts: Failed attempts before the user is told.
    """

    stall_seconds: float = 2.5
    lost_grace_seconds: float = 0.5
    check_interval_seconds: float = 0.25
    initial_backoff_seconds: float = 1.0
    max_backoff_seconds: float = 30.0
    notify_after_attempts: int = 3

    def backoff(self, attempts: int) -> float:
        """Seconds to wait after the ``attempts``-th attempt of an outage."""
        delay: float = self.initial_backoff_seconds * 2 ** max(attempts - 1, 0)
        return min(delay, self.max_backoff_seconds)


@dataclass(slots=True)
class _Outage:
    """One stall, from detection to recovery."""

    reason: str
    silent_since: float
    next_attempt: float
    attempts: int = 0


@dataclass(slots=True)
class _DeviceLoss:
    """A notification that the device stopped, waiting for frames to resume."""

    reason: str
    at: float
    samples: int


class CaptureWatchdog:
    """Detects a stalled capture and restarts it.

    Args:
        capture: The engine to watch and restart.
        config: Timing; the defaults suit a real microphone.
        notify: Tells the user about a lasting outage; None logs only.
        clock: Monotonic time source.
        device_name: Current default microphone, for the logs.
        restart_errors: Exceptions a failed restart raises (the native
            ``AudioDeviceError``); they count as a failed attempt, anything
            else is a bug and propagates.
    """

    def __init__(
        self,
        capture: CaptureControl,
        *,
        config: WatchdogConfig | None = None,
        notify: Notifier | None = None,
        clock: Clock = time.monotonic,
        device_name: DeviceName | None = None,
        restart_errors: tuple[type[Exception], ...] = (RuntimeError,),
    ) -> None:
        self._capture = capture
        self._config = config or WatchdogConfig()
        self._notify = notify
        self._clock = clock
        self._device_name = device_name
        self._restart_errors = restart_errors
        self._last_count: int | None = None
        self._last_progress = 0.0
        self._last_check: float | None = None
        self._loss: _DeviceLoss | None = None
        self._outage: _Outage | None = None
        self._notified = False

    @property
    def stalled(self) -> bool:
        """True from detecting a stall until frames arrive again."""
        return self._outage is not None

    def reset(self) -> None:
        """Forget everything; called when the engine (re)starts on purpose."""
        self._last_count = None
        self._last_check = None
        self._loss = None
        self._outage = None
        self._notified = False

    def device_lost(self, reason: str) -> None:
        """The device reported that it stopped; restart unless frames resume soon."""
        if self._loss is None:
            now = self._clock()
            self._loss = _DeviceLoss(reason, now, self._capture.samples_received())
            log.warning("audio.capture.device_lost", reason=reason, device=self._device())

    def device_rerouted(self) -> None:
        """The stream moved to a new default endpoint; the counter confirms it works."""
        log.info("audio.capture.rerouted", device=self._device())

    async def run(self) -> None:
        """Check forever at the configured interval; cancel to stop."""
        try:
            while True:
                await self.check()
                await asyncio.sleep(self._config.check_interval_seconds)
        except Exception:
            # A dead watchdog is the silent failure it exists to prevent;
            # make the cause visible before propagating it into the task.
            log.exception("audio.watchdog.crashed")
            raise

    async def check(self) -> None:
        """One observation: note progress, detect a stall, or attempt recovery."""
        now = self._clock()
        self._rebase_after_gap(now)
        self._last_check = now
        count = self._capture.samples_received()
        if self._last_count is None or count > self._last_count:
            await self._progress(now, count)
            return
        if self._outage is None:
            self._outage = self._detect(now, count)
        if self._outage is not None and now >= self._outage.next_attempt:
            await self._attempt(self._outage)

    def _rebase_after_gap(self, now: float) -> None:
        """Ignore time the watchdog itself was not running.

        A suspended machine or a stalled process also stops this loop; the
        microphone cannot be judged for time nobody was looking, so the
        silence is measured again from now. A dead stream still shows up
        ``stall_seconds`` later.
        """
        if self._last_check is None:
            return
        gap = now - self._last_check
        if gap <= self._config.stall_seconds:
            return
        log.info("audio.watchdog.resumed_after_gap", gap_s=round(gap, 1))
        self._last_progress = now
        if self._loss is not None:
            self._loss.at = now

    async def _progress(self, now: float, count: int) -> None:
        self._last_count = count
        self._last_progress = now
        if self._loss is not None and count > self._loss.samples:
            log.info("audio.capture.device_back", reason=self._loss.reason)
            self._loss = None
        if self._outage is not None:
            await self._recovered(now, self._outage)

    def _detect(self, now: float, count: int) -> _Outage | None:
        """A new outage when the device was lost or the counter stayed flat too long."""
        loss = self._loss
        if loss is not None and now - loss.at >= self._config.lost_grace_seconds:
            reason = loss.reason
        elif now - self._last_progress >= self._config.stall_seconds:
            reason = NO_SAMPLES
        else:
            return None
        log.warning(
            "audio.capture.stalled",
            reason=reason,
            silent_s=round(now - self._last_progress, 1),
            samples_received=count,
            device=self._device(),
        )
        return _Outage(reason=reason, silent_since=self._last_progress, next_attempt=now)

    async def _attempt(self, outage: _Outage) -> None:
        if outage.attempts >= self._config.notify_after_attempts and not self._notified:
            self._notified = True
            await self._tell(MICROPHONE_LOST)
        outage.attempts += 1
        self._loss = None  # the restart answers the notification
        try:
            reopened = await self._capture.restart_capture()
        except self._restart_errors as error:
            log.warning("audio.capture.restart_failed", attempt=outage.attempts, error=str(error))
        else:
            log.info("audio.capture.restarted", attempt=outage.attempts, reopened=reopened)
        outage.next_attempt = self._clock() + self._config.backoff(outage.attempts)

    async def _recovered(self, now: float, outage: _Outage) -> None:
        self._outage = None
        log.info(
            "audio.capture.recovered",
            reason=outage.reason,
            attempts=outage.attempts,
            outage_s=round(now - outage.silent_since, 1),
            device=self._device(),
        )
        if self._notified:
            self._notified = False
            await self._tell(MICROPHONE_BACK)

    async def _tell(self, text: str) -> None:
        if self._notify is None:
            return
        try:
            await self._notify(text)
        except Exception:
            # The messenger must not take recovery down with it; the log
            # still carries the outage.
            log.exception("audio.watchdog.notify_failed", text=text)

    def _device(self) -> str | None:
        return self._device_name() if self._device_name is not None else None


# Design notes:
# - Policy in Python, mechanism in C++: the native engine only counts samples,
#   forwards miniaudio's notifications and offers restartCapture(); when to
#   restart, how long to back off and what to tell the user live here, where
#   they are tested with a fake clock and no device (ADR-0027).
# - The sample counter, not the processed-sample clock, is watched: it is
#   incremented in the device callback, so a busy worker (inference on a
#   loaded machine) never looks like a dead microphone.
# - A restart that "succeeds" but delivers nothing is just another failed
#   attempt; only frames end an outage.
# - device_name is called only when logging an outage, never per check; on
#   Windows it is a COM call into the audio endpoint API.
