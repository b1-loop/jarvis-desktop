"""Facade over the native status overlay (``jarvis_overlay``).

The overlay is a small always-on-top panel showing what Jarvis is doing:
a state indicator, a live waveform, the user's transcript and the spoken
reply. All rendering happens on a native UI thread; Python only pushes
state through cheap, thread-safe setters and never receives callbacks.

The native window is injected through the :class:`NativeOverlay` Protocol
so the facade is unit-tested with fakes, and :class:`StubOverlay` stands in
for it in text-only mode or when the native build is missing. The
overlay bridge (``core/overlay_bridge.py``) is the only caller of the
setters; the tray icon and dashboard read the same state from the event bus.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, cast

from jarvis import _native
from jarvis.utils.errors import JarvisError, NativeUnavailableError
from jarvis.utils.logging import get_logger

log = get_logger(__name__)


class OverlayError(JarvisError):
    """The native overlay window could not be created.

    Raised by :func:`create_native_overlay` when there is no desktop or no
    Direct3D 11 device (not even the WARP software rasteriser). Callers that
    can live without the panel should catch it and use :class:`StubOverlay`.
    """


class OverlayState(StrEnum):
    """Mirror of the native ``AssistantState`` enum, by member name."""

    IDLE = "IDLE"
    LISTENING = "LISTENING"
    THINKING = "THINKING"
    SPEAKING = "SPEAKING"


class OverlayCorner(StrEnum):
    """Mirror of the native ``Corner`` enum, by member name."""

    TOP_LEFT = "TOP_LEFT"
    TOP_RIGHT = "TOP_RIGHT"
    BOTTOM_LEFT = "BOTTOM_LEFT"
    BOTTOM_RIGHT = "BOTTOM_RIGHT"


@dataclass(frozen=True, slots=True)
class OverlayConfig:
    """Geometry and behaviour of the panel. Sizes are logical pixels at 96 DPI.

    Attributes:
        corner: Which corner of the primary monitor's work area to anchor to.
        width: Panel width.
        height: Panel height.
        margin: Distance from the two edges of the chosen corner.
        click_through: Let clicks fall through to the window underneath.
        opacity: Whole-panel opacity, 0 (invisible) to 255 (opaque).
        active_fps: Frame rate while something is happening.
        idle_fps: Frame rate once the panel has been static for a second.
        level_history: Number of level samples kept for the waveform.
        max_text_chars: Longest transcript or reply kept, in characters.
    """

    corner: OverlayCorner = OverlayCorner.BOTTOM_RIGHT
    width: int = 380
    height: int = 132
    margin: int = 24
    click_through: bool = True
    opacity: int = 235
    active_fps: int = 60
    idle_fps: int = 10
    level_history: int = 64
    max_text_chars: int = 240


class NativeOverlay(Protocol):
    """Shape of ``jarvis_overlay.OverlayWindow`` as the facade uses it.

    ``set_state`` takes the state's member name rather than the native enum
    so fakes and the stub need no access to the compiled module.
    """

    def set_state(self, name: str) -> None:
        """Switch the indicator; ``name`` is an ``OverlayState`` value."""

    def push_level(self, level: float) -> None:
        """Append one waveform sample in ``[0, 1]``."""

    def set_transcript(self, text: str) -> None:
        """Show what the user said."""

    def set_response(self, text: str) -> None:
        """Show what Jarvis is saying."""

    def set_visible(self, visible: bool) -> None:
        """Show or hide the panel without destroying it."""

    def flash_wake_word(self) -> None:
        """Pulse the indicator to acknowledge the wake word."""

    def visible(self) -> bool:
        """Whether the panel is currently shown."""

    def running(self) -> bool:
        """True while the native UI thread is alive."""

    def close(self) -> None:
        """Destroy the window and join the UI thread. Idempotent."""


class StubOverlay:
    """Stand-in for the native window in text-only mode.

    Records the latest values so the rest of the application (and tests)
    can read them back, and logs state changes at debug level so a
    terminal session still shows what the panel would have displayed.
    Nothing is drawn anywhere; this is a stub, not a fallback renderer.
    """

    def __init__(self) -> None:
        self.state = OverlayState.IDLE.value
        self.transcript = ""
        self.response = ""
        self.levels: list[float] = []
        self.wake_flashes = 0
        self._visible = True
        self._running = True

    def set_state(self, name: str) -> None:
        """Record the state and log the transition."""
        self.state = name
        log.debug("overlay.stub.state", state=name)

    def push_level(self, level: float) -> None:
        """Record the level; the list is unbounded only because it is a stub."""
        self.levels.append(level)

    def set_transcript(self, text: str) -> None:
        """Record the transcript."""
        self.transcript = text

    def set_response(self, text: str) -> None:
        """Record the reply."""
        self.response = text

    def set_visible(self, visible: bool) -> None:
        """Record visibility."""
        self._visible = visible

    def flash_wake_word(self) -> None:
        """Count the flash."""
        self.wake_flashes += 1

    def visible(self) -> bool:
        """Whether ``set_visible(True)`` was the last visibility call."""
        return self._visible

    def running(self) -> bool:
        """True until :meth:`close`."""
        return self._running

    def close(self) -> None:
        """Mark the stub as closed."""
        self._running = False


class Overlay:
    """What the orchestrator talks to; one method per thing the panel shows.

    Keeps the last state it was told so the tray icon can read it without a
    round trip into the native module, and refuses to forward anything once
    closed so a late TTS chunk after shutdown cannot touch a dead window.
    """

    def __init__(self, native: NativeOverlay) -> None:
        self._native = native
        self._state = OverlayState.IDLE
        self._closed = False

    def __enter__(self) -> Overlay:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    @property
    def state(self) -> OverlayState:
        """The last state passed to :meth:`set_state`."""
        return self._state

    @property
    def running(self) -> bool:
        """True while the panel is open and its UI thread alive."""
        return not self._closed and self._native.running()

    @property
    def visible(self) -> bool:
        """Whether the panel is shown."""
        return self._native.visible()

    def set_state(self, state: OverlayState) -> None:
        """Switch the indicator colour and label."""
        self._state = state
        self._forward(lambda: self._native.set_state(state.value))

    def push_level(self, level: float) -> None:
        """Append one waveform sample; values outside ``[0, 1]`` are clamped natively."""
        self._forward(lambda: self._native.push_level(float(level)))

    def set_transcript(self, text: str) -> None:
        """Show what the user said (the newest words win if it is long)."""
        self._forward(lambda: self._native.set_transcript(text))

    def set_response(self, text: str) -> None:
        """Show what Jarvis is saying."""
        self._forward(lambda: self._native.set_response(text))

    def flash_wake_word(self) -> None:
        """Acknowledge the wake word with a short pulse on the indicator."""
        self._forward(self._native.flash_wake_word)

    def show(self) -> None:
        """Show the panel."""
        self._forward(lambda: self._native.set_visible(True))

    def hide(self) -> None:
        """Hide the panel; state keeps updating underneath."""
        self._forward(lambda: self._native.set_visible(False))

    def close(self) -> None:
        """Destroy the window. Idempotent; later setters are ignored."""
        if self._closed:
            return
        self._closed = True
        self._native.close()
        log.info("overlay.closed")

    def _forward(self, call: _Setter) -> None:
        if self._closed:
            log.debug("overlay.ignored_after_close")
            return
        call()


class _Setter(Protocol):
    def __call__(self) -> None: ...


def create_native_overlay(config: OverlayConfig | None = None) -> Overlay:
    """Construct the real ``jarvis_overlay.OverlayWindow`` and wrap it.

    Raises:
        NativeUnavailableError: If the extension module is not built.
        OverlayError: If the window or its Direct3D device cannot be created.
    """
    config = config or OverlayConfig()
    module = _native.load("jarvis_overlay")
    native_config = module.OverlayConfig()
    native_config.corner = module.Corner.__members__[config.corner.value]
    native_config.width = config.width
    native_config.height = config.height
    native_config.margin = config.margin
    native_config.click_through = config.click_through
    native_config.opacity = config.opacity
    native_config.active_fps = config.active_fps
    native_config.idle_fps = config.idle_fps
    native_config.limits.level_history = config.level_history
    native_config.limits.max_text_chars = config.max_text_chars
    try:
        window = module.OverlayWindow(native_config)
    except module.OverlayError as exc:
        raise OverlayError(str(exc)) from exc
    log.info("overlay.created", corner=config.corner.value)
    return Overlay(cast(NativeOverlay, window))


def create_overlay(config: OverlayConfig | None = None, *, enabled: bool = True) -> Overlay:
    """Return the native overlay when possible, otherwise a :class:`StubOverlay`.

    This is the graceful-degradation entry point for the composition root:
    a missing build or a headless session logs a warning and the assistant
    carries on without a panel.
    """
    if not enabled:
        return Overlay(StubOverlay())
    try:
        return create_native_overlay(config)
    except (NativeUnavailableError, OverlayError) as exc:
        log.warning("overlay.unavailable", reason=str(exc))
        return Overlay(StubOverlay())


# Design notes:
# - set_state crosses the boundary as a string on purpose: the native enum
#   lives in a module that text-only mode never imports, and a name lookup
#   costs nothing against the per-frame work on the other side.
# - The facade swallows nothing but calls after close(); every other error is
#   a programming error and should surface.
# - No asyncio here: every native setter returns in microseconds, so wrapping
#   them in executors would only add latency to the waveform.
