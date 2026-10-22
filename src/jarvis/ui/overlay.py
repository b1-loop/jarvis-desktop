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
callbacks. The HUD is also interactive (ADR-0024): hover controls, a text
field, dragging, an expanded conversation view and a settings page. What
the user does there is queued natively and drained by
:meth:`Overlay.poll_ui_events` (the overlay bridge polls every ~30 ms);
the types that cross the boundary are in ``ui/hud_protocol.py``. The HUD
can be minimized to a small glowing orb at the nearest screen edge
(:meth:`Overlay.minimize`, :meth:`Overlay.restore`), by itself after a
quiet spell (auto-minimize), and hides while a full-screen application is
in front (ADR-0024, update).

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
from datetime import datetime
from enum import StrEnum
from typing import Any, Final, Protocol, cast

from jarvis import _native
from jarvis.ui.hud_protocol import (
    HistoryRole,
    SettingsPage,
    UiEvent,
    event_to_native,
    parse_ui_event,
)
from jarvis.utils.errors import JarvisError, NativeUnavailableError
from jarvis.utils.logging import get_logger

log = get_logger(__name__)

HISTORY_TIME_FORMAT: Final = "%H:%M"


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
class SavedPosition:
    """Where the user left the HUD: logical pixels from a monitor's work-area corner.

    Attributes:
        monitor: Device name (``MONITORINFOEX.szDevice``); unknown names fall
            back to the primary monitor, and the HUD is always clamped fully on
            screen.
        x: Offset from the work area's left edge.
        y: Offset from its top edge.
        width: Logical width (the expanded panel's; 0 for the corner HUD).
        height: Logical height.
    """

    monitor: str
    x: float
    y: float
    width: float = 0.0
    height: float = 0.0


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
        interactive: Hover controls, typing, dragging and the expanded view;
            transparent pixels stay click-through (ADR-0024).
        position: Where the corner HUD was dragged to; None anchors it.
        expanded_bounds: Position and size of the expanded panel, or None.
        start_minimized: Open as the orb (the user left it minimized).
        orb_position: Where the orb was dragged to; None docks it next to
            the HUD.
        auto_minimize_seconds: Shrink to the orb after this many quiet
            seconds; 0 is off.
        hide_in_fullscreen: Hide the HUD and the orb while a full-screen
            application is in front on their monitor.
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
    interactive: bool = True
    position: SavedPosition | None = None
    expanded_bounds: SavedPosition | None = None
    start_minimized: bool = False
    orb_position: SavedPosition | None = None
    auto_minimize_seconds: float = 0.0
    hide_in_fullscreen: bool = True


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

    def poll_ui_events(self) -> list[dict[str, object]]:
        """Everything the user did since the last call, as dicts with a ``kind``."""

    def inject_ui_event(self, event: dict[str, object]) -> None:
        """Test hook: queue ``event`` as if the user had caused it."""

    def set_muted(self, muted: bool) -> None:
        """Show the microphone control as muted."""

    def add_history(self, role: str, text: str, time: str) -> None:
        """Append one conversation line to the expanded view."""

    def set_settings(self, page: dict[str, object]) -> None:
        """Replace the settings page (``SettingsPage.to_native()``)."""

    def set_appearance(self, theme: str, intensity: str, size: str) -> None:
        """Switch theme, intensity and size live (settings spellings)."""

    def set_expanded(self, expanded: bool) -> None:
        """Expand to the panel or collapse to the corner HUD."""

    def expanded(self) -> bool:
        """Whether the HUD is (or is animating towards) the expanded panel."""

    def set_minimized(self, minimized: bool) -> None:
        """Shrink the HUD to the orb or grow it back (animated)."""

    def minimized(self) -> bool:
        """Whether the HUD is (or is animating towards) the orb."""

    def set_auto_minimize(self, seconds: float) -> None:
        """Auto-minimize after ``seconds`` of quiet; 0 switches it off."""

    def set_hide_in_fullscreen(self, hide: bool) -> None:
        """Hide while a full-screen application is in front."""


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
        self.muted = False
        self.history: list[tuple[str, str, str]] = []
        self.settings: dict[str, object] = {}
        self.appearance: tuple[str, str, str] | None = None
        self.auto_minimize_seconds = 0.0
        self.hide_in_fullscreen = True
        self._expanded = False
        self._minimized = False
        self._events: list[dict[str, object]] = []
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

    def poll_ui_events(self) -> list[dict[str, object]]:
        """Injected events only: nobody can click a stub."""
        events, self._events = self._events, []
        return events

    def inject_ui_event(self, event: dict[str, object]) -> None:
        """Queue ``event`` for the next poll."""
        self._events.append(dict(event))

    def set_muted(self, muted: bool) -> None:
        """Record the microphone state."""
        self.muted = muted

    def add_history(self, role: str, text: str, time: str) -> None:
        """Record the line."""
        self.history.append((role, text, time))

    def set_settings(self, page: dict[str, object]) -> None:
        """Record the page."""
        self.settings = page

    def set_appearance(self, theme: str, intensity: str, size: str) -> None:
        """Record the appearance."""
        self.appearance = (theme, intensity, size)

    def set_expanded(self, expanded: bool) -> None:
        """Record the request."""
        self._expanded = expanded

    def expanded(self) -> bool:
        """The last request."""
        return self._expanded

    def set_minimized(self, minimized: bool) -> None:
        """Record the request; minimizing collapses, like the native window."""
        self._minimized = minimized
        if minimized:
            self._expanded = False

    def minimized(self) -> bool:
        """The last request."""
        return self._minimized

    def set_auto_minimize(self, seconds: float) -> None:
        """Record the delay; nothing ever shrinks a stub by itself."""
        self.auto_minimize_seconds = max(seconds, 0.0)

    def set_hide_in_fullscreen(self, hide: bool) -> None:
        """Record the switch."""
        self.hide_in_fullscreen = hide


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

    # --- interactive HUD (ADR-0024) -----------------------------------------

    @property
    def expanded(self) -> bool:
        """Whether the HUD shows (or is opening) the expanded panel."""
        return not self._closed and self._native.expanded()

    def poll_ui_events(self) -> list[UiEvent]:
        """Everything the user did in the HUD since the last poll, typed.

        Raises:
            HudProtocolError: If the native side reports an unknown event,
                which means the build and this code are out of step.
        """
        if self._closed:
            return []
        return [parse_ui_event(raw) for raw in self._native.poll_ui_events()]

    def inject_ui_event(self, event: UiEvent) -> None:
        """Test and demo hook: queue ``event`` as if the user had caused it."""
        self._forward(lambda: self._native.inject_ui_event(event_to_native(event)))

    def set_muted(self, muted: bool) -> None:
        """Show the microphone control as muted (or not)."""
        self._forward(lambda: self._native.set_muted(muted))

    def add_history(self, role: HistoryRole, text: str, at: datetime | None = None) -> None:
        """Append one line to the expanded view's conversation, stamped ``HH:MM``."""
        stamp = (at or datetime.now()).strftime(HISTORY_TIME_FORMAT)
        self._forward(lambda: self._native.add_history(role.value, text, stamp))

    def set_settings(self, page: SettingsPage) -> None:
        """Replace the settings page the HUD shows."""
        self._forward(lambda: self._native.set_settings(page.to_native()))

    def set_appearance(
        self, theme: OverlayTheme, intensity: OverlayIntensity, size: OverlaySize
    ) -> None:
        """Switch palette, richness and size without restarting."""
        self._forward(lambda: self._native.set_appearance(theme.value, intensity.value, size.value))

    def set_expanded(self, expanded: bool) -> None:
        """Expand to the panel or collapse to the corner HUD (animated)."""
        self._forward(lambda: self._native.set_expanded(expanded))

    # --- minimize to the orb (ADR-0024, update) -----------------------------

    @property
    def minimized(self) -> bool:
        """Whether the HUD is (or is shrinking into) the orb, whoever asked for it."""
        return not self._closed and self._native.minimized()

    def minimize(self) -> None:
        """Shrink the HUD into the orb at the nearest screen edge (animated).

        The orb keeps showing the state, clicks on it restore the HUD and
        everything around it is click-through. Asked for here (tray, hotkey),
        no ``Minimized`` event comes back; the caller remembers the choice.
        """
        self._forward(lambda: self._native.set_minimized(True))

    def restore(self) -> None:
        """Grow the orb back into the HUD (animated)."""
        self._forward(lambda: self._native.set_minimized(False))

    def set_auto_minimize(self, seconds: float) -> None:
        """Shrink to the orb after ``seconds`` of quiet; 0 switches it off."""
        self._forward(lambda: self._native.set_auto_minimize(float(seconds)))

    def set_hide_in_fullscreen(self, hide: bool) -> None:
        """Hide while a full-screen application is in front on the HUD's monitor."""
        self._forward(lambda: self._native.set_hide_in_fullscreen(hide))

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
    try:
        window = module.OverlayWindow(_native_config(module, config))
    except module.OverlayError as exc:
        raise OverlayError(str(exc)) from exc
    log.info(
        "overlay.created",
        corner=config.corner.value,
        theme=config.theme.value,
        size=config.size.value,
        intensity=config.intensity.value,
        interactive=config.interactive,
        transparent=window.transparent(),
        fallback_reason=window.fallback_reason() or None,
        bloom=window.bloom(),
        bloom_reason=window.bloom_reason() or None,
    )
    return Overlay(cast(NativeOverlay, window), boot_animation=config.boot_animation)


def _native_config(module: Any, config: OverlayConfig) -> Any:
    """``jarvis_overlay.OverlayConfig`` mirroring ``config``."""
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
    native_config.interactive = config.interactive
    native_config.position = _native_position(module, config.position)
    native_config.expanded_bounds = _native_position(module, config.expanded_bounds)
    native_config.start_minimized = config.start_minimized
    native_config.orb_position = _native_position(module, config.orb_position)
    native_config.auto_minimize_seconds = config.auto_minimize_seconds
    native_config.hide_in_fullscreen = config.hide_in_fullscreen
    native_config.limits.level_history = config.level_history
    native_config.limits.max_text_chars = config.max_text_chars
    return native_config


def _native_position(module: Any, position: SavedPosition | None) -> object | None:
    """``jarvis_overlay.SavedPlacement`` for ``position``, or None."""
    if position is None:
        return None
    placement: object = module.SavedPlacement(
        position.monitor, position.x, position.y, position.width, position.height
    )
    return placement


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
# - minimize()/restore() mirror set_expanded: the request is applied on the
#   UI thread's next frame and not echoed back as an event, while the HUD's
#   own minimize control, a click on the orb and auto-minimize do report
#   Minimized/Restored, flagged automatic or not, so Python remembers only
#   the user's choices.
# - UI events are polled, not called back: a callback from the native UI
#   thread would need the GIL there and could stall a frame behind the
#   event loop; a queue drained by the bridge keeps both sides independent.
