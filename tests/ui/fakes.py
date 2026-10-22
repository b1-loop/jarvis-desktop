"""Fakes for the native overlay, matching the Protocol in jarvis.ui.overlay."""

from __future__ import annotations

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

    def flash_wake_word(self) -> None:
        self.calls.append(("flash_wake_word", None))

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
