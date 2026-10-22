"""Fakes for the native overlay, matching the Protocol in jarvis.ui.overlay."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field


@dataclass
class FakeNativeOverlay:
    """Recording stand-in for ``jarvis_overlay.OverlayWindow``.

    Every call is appended to ``calls`` as ``(method, argument)`` so tests
    can assert on order as well as on the final values.
    """

    calls: list[tuple[str, object]] = field(default_factory=list)
    _visible: bool = True
    _running: bool = True
    _transparent: bool = True
    _boot_started: bool = False

    def set_state(self, name: str) -> None:
        self.calls.append(("set_state", name))

    def push_level(self, level: float) -> None:
        self.calls.append(("push_level", level))

    def set_transcript(self, text: str) -> None:
        self.calls.append(("set_transcript", text))

    def set_response(self, text: str) -> None:
        self.calls.append(("set_response", text))

    def set_visible(self, visible: bool) -> None:
        self._visible = visible
        self.calls.append(("set_visible", visible))

    def wake_pulse(self) -> None:
        self.calls.append(("wake_pulse", None))

    def play_boot(self, info: Sequence[tuple[str, str]] = ()) -> bool:
        started = not self._boot_started
        self._boot_started = True
        self.calls.append(("play_boot", started))
        if info:
            self.calls.append(("boot_info", list(info)))
        return started

    def skip_boot(self) -> bool:
        self.calls.append(("skip_boot", None))
        return self._boot_started

    def transparent(self) -> bool:
        return self._transparent

    def fallback_reason(self) -> str:
        return "" if self._transparent else "fake: no composition"

    def bloom(self) -> bool:
        return self._transparent

    def bloom_reason(self) -> str:
        return "" if self._transparent else "fake: no bloom"

    def visible(self) -> bool:
        return self._visible

    def running(self) -> bool:
        return self._running

    def close(self) -> None:
        self._running = False
        self.calls.append(("close", None))

    def names(self) -> list[str]:
        """Method names in call order, for terse assertions."""
        return [name for name, _ in self.calls]
