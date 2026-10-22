"""Facade over the native status overlay (``jarvis_overlay``).

The overlay is a small always-on-top HUD showing what Jarvis is doing: an
animated ring whose motion tells the state (breathing when idle, a radial
waveform while listening and speaking, orbiting arcs while thinking, a
ripple when the wake word is heard), the user's transcript and the spoken
reply, with a wireframe globe, orbiting satellites and a particle field
around it, all bloomed on the GPU (ADR-0023). At startup it plays a ~4 s
boot sequence in the middle of the screen, with a diagnostic readout of
real values passed in by the caller, before whipping into its corner.
All rendering and animation happen on a native UI thread; Python only
pushes state through cheap, thread-safe setters and never receives
callbacks.

The native window is injected through the :class:`NativeOverlay` Protocol
so the facade is unit-tested with fakes, and :class:`StubOverlay` stands in
for it in text-only mode or when the native build is missing. The
overlay bridge (``core/overlay_bridge.py``) is the only caller of the
setters and the launcher calls :meth:`Overlay.play_boot` when Jarvis
starts; the tray icon and dashboard read the same state from the event bus.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
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


class OverlayTheme(StrEnum):
    """Palette of the HUD; values are the settings spelling, names the native enum."""

    CYAN = "cyan"
    AMBER = "amber"


class OverlayIntensity(StrEnum):
    """How rich the HUD is (particles, bloom, motion); values are the settings spelling."""

    CALM = "calm"
    NORMAL = "normal"
    MAX = "max"


class OverlaySize(StrEnum):
    """Preset size of the corner HUD; values are the settings spelling."""

    COMPACT = "compact"
    NORMAL = "normal"
    LARGE = "large"


BootInfo = Mapping[str, str]
"""Diagnostic readout for the boot sequence, label to value, in display order."""


@dataclass(frozen=True, slots=True)
class OverlayConfig:
    """Geometry and behaviour of the panel. Sizes are logical pixels at 96 DPI.

    Attributes:
        corner: Which corner of the primary monitor's work area to anchor to.
        size: Preset size of the HUD (compact 420x132, normal 600x216, large
            760x300 at 96 DPI); its text scales with it.
        width: Explicit panel width overriding the preset, or None.
        height: Explicit panel height overriding the preset, or None.
        margin: Distance from the two edges of the chosen corner.
        click_through: Let clicks fall through to the window underneath.
        opacity: Overall opacity, 0 (invisible) to 255 (opaque).
        active_fps: Frame rate while something animates.
        idle_fps: Frame rate while the idle ring only breathes.
        level_history: Number of level samples kept for the waveform.
        max_text_chars: Longest transcript or reply kept, in characters.
        transparent: Prefer per-pixel transparency (DirectComposition); the
            whole-window-alpha fallback is used when it is off or unavailable.
        theme: Palette of the HUD.
        boot_animation: Whether :meth:`Overlay.play_boot` plays anything.
        intensity: Particle count, bloom strength and motion of the HUD.
        bloom: Run the GPU bloom pipeline (it falls back to plain additive
            light by itself if the GPU cannot run it).
    """

    corner: OverlayCorner = OverlayCorner.BOTTOM_RIGHT
    size: OverlaySize = OverlaySize.LARGE
    width: int | None = None
    height: int | None = None
    margin: int = 24
    click_through: bool = True
    opacity: int = 235
    active_fps: int = 60
    idle_fps: int = 15
    level_history: int = 64
    max_text_chars: int = 240
    transparent: bool = True
    theme: OverlayTheme = OverlayTheme.CYAN
    boot_animation: bool = True
    intensity: OverlayIntensity = OverlayIntensity.NORMAL
    bloom: bool = True


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

    def wake_pulse(self) -> None:
        """Ripple the ring outwards to acknowledge the wake word."""

    def play_boot(self, info: Sequence[tuple[str, str]] = ()) -> bool:
        """Start the boot sequence with ``info`` as its readout; True only once."""

    def skip_boot(self) -> bool:
        """Cut a running boot sequence short; True if one was skipped."""

    def transparent(self) -> bool:
        """Whether the window has per-pixel transparency."""

    def bloom(self) -> bool:
        """Whether the GPU bloom pipeline renders the light."""

    def bloom_reason(self) -> str:
        """Why there is no bloom; empty when there is."""

    def fallback_reason(self) -> str:
        """Why the fallback surface is used; empty when transparent."""

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
        self.boot_started = False
        self.boot_skipped = False
        self.boot_info: list[tuple[str, str]] = []
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

    def wake_pulse(self) -> None:
        """Count the pulse."""
        self.wake_flashes += 1

    def play_boot(self, info: Sequence[tuple[str, str]] = ()) -> bool:
        """Record the first request (and its readout) only, like the native model."""
        if self.boot_started:
            return False
        self.boot_started = True
        self.boot_info = list(info)
        return True

    def skip_boot(self) -> bool:
        """Record a skip of a started boot, once."""
        if not self.boot_started or self.boot_skipped:
            return False
        self.boot_skipped = True
        return True

    def transparent(self) -> bool:
        """A stub has no window, so nothing is transparent."""
        return False

    def fallback_reason(self) -> str:
        """Explain that nothing is drawn."""
        return "stub overlay: nothing is drawn"

    def bloom(self) -> bool:
        """A stub draws nothing, so nothing is bloomed."""
        return False

    def bloom_reason(self) -> str:
        """Explain that nothing is drawn."""
        return "stub overlay: nothing is drawn"

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

    Args:
        native: The window (or a stub/fake with the same shape).
        boot_animation: When False, :meth:`play_boot` is a logged no-op.
    """

    def __init__(self, native: NativeOverlay, *, boot_animation: bool = True) -> None:
        self._native = native
        self._boot_animation = boot_animation
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

    @property
    def transparent(self) -> bool:
        """Whether the panel is drawn with per-pixel transparency."""
        return self._native.transparent()

    @property
    def bloom(self) -> bool:
        """Whether the HUD's light is bloomed on the GPU."""
        return self._native.bloom()

    def set_state(self, state: OverlayState) -> None:
        """Switch the HUD to ``state``; the native side cross-fades over ~200 ms."""
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

    def wake_pulse(self) -> None:
        """Acknowledge the wake word with a bright ripple from the ring."""
        self._forward(self._native.wake_pulse)

    def flash_wake_word(self) -> None:
        """Same as :meth:`wake_pulse`; the name the voice pipeline's bridge uses."""
        self.wake_pulse()

    def play_boot(self, info: BootInfo | None = None) -> bool:
        """Play the boot sequence once. Non-blocking and idempotent.

        Returns immediately; the animation (about 4 s) runs on the native
        UI thread and ends with the panel in its corner. Later calls, calls
        after :meth:`close` and calls with ``boot_animation`` disabled do
        nothing.

        Args:
            info: The systems-check readout, e.g. ``{"MODEL": "qwen2.5:7b",
                "MODE": "CLOUD", "MIC": "Microphone (Realtek Audio)"}``, shown
                in insertion order. The idle panel keeps showing ``MODEL``,
                ``MODE`` and ``MIC`` afterwards. None shows only what the
                renderer knows itself (GPU, surface, bloom).

        Returns:
            True only for the call that started the sequence.
        """
        if self._closed or not self._boot_animation:
            log.debug("overlay.boot.skipped", closed=self._closed, enabled=self._boot_animation)
            return False
        lines = [(str(label), str(value)) for label, value in (info or {}).items()]
        started = self._native.play_boot(lines)
        if started:
            log.info("overlay.boot.started", readout=[label for label, _ in lines])
        return started

    def skip_boot(self) -> bool:
        """Cut a running boot sequence short; it still glides into the corner."""
        if self._closed:
            return False
        return self._native.skip_boot()

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

    Logs which surface the window ended up with, so "why is my overlay a
    dark box?" has an answer in the log (per-pixel transparency needs
    DirectComposition; the fallback uses whole-window alpha).

    Raises:
        NativeUnavailableError: If the extension module is not built.
        OverlayError: If the window or its Direct3D device cannot be created.
    """
    config = config or OverlayConfig()
    module = _native.load("jarvis_overlay")
    native_config = module.OverlayConfig()
    native_config.corner = module.Corner.__members__[config.corner.value]
    native_config.size = module.PanelSize.__members__[config.size.name]
    native_config.width = config.width or 0
    native_config.height = config.height or 0
    native_config.margin = config.margin
    native_config.click_through = config.click_through
    native_config.opacity = config.opacity
    native_config.active_fps = config.active_fps
    native_config.idle_fps = config.idle_fps
    native_config.transparent = config.transparent
    native_config.theme = module.Theme.__members__[config.theme.name]
    native_config.intensity = module.Intensity.__members__[config.intensity.name]
    native_config.bloom = config.bloom
    native_config.limits.level_history = config.level_history
    native_config.limits.max_text_chars = config.max_text_chars
    try:
        window = module.OverlayWindow(native_config)
    except module.OverlayError as exc:
        raise OverlayError(str(exc)) from exc
    log.info(
        "overlay.created",
        corner=config.corner.value,
        theme=config.theme.value,
        size=config.size.value,
        intensity=config.intensity.value,
        transparent=window.transparent(),
        fallback_reason=window.fallback_reason() or None,
        bloom=window.bloom(),
        bloom_reason=window.bloom_reason() or None,
    )
    return Overlay(cast(NativeOverlay, window), boot_animation=config.boot_animation)


def default_microphone_name() -> str | None:
    """Friendly name of the default capture device, for the boot readout.

    The audio engine records from the default device, so this is the
    microphone Jarvis listens on. None when the native overlay is not built
    or there is no capture device.
    """
    try:
        module = _native.load("jarvis_overlay")
    except NativeUnavailableError:
        return None
    name: str | None = module.default_microphone_name()
    return name


def create_overlay(config: OverlayConfig | None = None, *, enabled: bool = True) -> Overlay:
    """Return the native overlay when possible, otherwise a :class:`StubOverlay`.

    This is the graceful-degradation entry point for the composition root:
    a missing build or a headless session logs a warning and the assistant
    carries on without a panel.
    """
    boot_animation = (config or OverlayConfig()).boot_animation
    if not enabled:
        return Overlay(StubOverlay(), boot_animation=boot_animation)
    try:
        return create_native_overlay(config)
    except (NativeUnavailableError, OverlayError) as exc:
        log.warning("overlay.unavailable", reason=str(exc))
        return Overlay(StubOverlay(), boot_animation=boot_animation)


# Design notes:
# - set_state crosses the boundary as a string on purpose: the native enum
#   lives in a module that text-only mode never imports, and a name lookup
#   costs nothing against the per-frame work on the other side.
# - The boot readout crosses the boundary as a list of pairs, not a dict:
#   the order is the display order, and pybind11 maps a list of tuples onto
#   the native vector without a custom caster.
# - play_boot returns a bool instead of raising on repeats: the launcher and
#   the voice front end may both ask for it at startup, and "already
#   playing" is not an error, just a no-op worth knowing about in tests.
# - All animation timing lives natively (native/overlay/animation.hpp) and
#   depends on elapsed time only, so Python never needs to tick anything.
# - The facade swallows nothing but calls after close(); every other error is
#   a programming error and should surface.
# - No asyncio here: every native setter returns in microseconds, so wrapping
#   them in executors would only add latency to the waveform.
