"""What crosses the boundary between Python and the interactive HUD (ADR-0024).

Two directions, both plain data:

* **Events out.** The native UI thread never calls into Python. It queues
  what the user did and :meth:`jarvis.ui.overlay.Overlay.poll_ui_events`
  drains the queue as dicts, which :func:`parse_ui_event` turns into the
  typed events below (``SubmitText``, ``CloseRequested``, ...).
* **Data in.** The settings page is described by :class:`SettingsPage`
  (choices and current values); the native side renders and edits it and
  reports edits as :class:`SettingChanged`. The conversation history is
  pushed line by line as :class:`HistoryRole` plus text.

Only data lives here, no behaviour: what an event means is decided by the
handlers in ``core/hud_wiring.py``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum

from jarvis.utils.errors import JarvisError


class HudProtocolError(JarvisError):
    """The native HUD reported something this module does not understand."""


@dataclass(frozen=True, slots=True)
class SubmitText:
    """The user typed a request and pressed Enter."""

    text: str


@dataclass(frozen=True, slots=True)
class CloseRequested:
    """Hide the HUD; the tray's "Show overlay" brings it back."""


@dataclass(frozen=True, slots=True)
class QuitRequested:
    """Quit Jarvis (already confirmed in the HUD)."""


@dataclass(frozen=True, slots=True)
class ToggleMute:
    """Mute or unmute the microphone."""


@dataclass(frozen=True, slots=True)
class SetExpanded:
    """The HUD expanded to the panel (True) or collapsed to the corner."""

    expanded: bool


@dataclass(frozen=True, slots=True)
class SettingChanged:
    """A setting was edited; the value is text, validated by Python."""

    key: str
    value: str


@dataclass(frozen=True, slots=True)
class OpenLog:
    """Open the log file."""


@dataclass(frozen=True, slots=True)
class RestartRequested:
    """Restart Jarvis so restart-only settings take effect."""


@dataclass(frozen=True, slots=True)
class SettingsOpened:
    """The settings page opened; lists that may have changed should be refreshed."""


@dataclass(frozen=True, slots=True)
class PlacementChanged:
    """The user moved the HUD, the expanded panel or the orb (or reset one).

    Attributes:
        expanded: The expanded panel's placement (else the corner HUD's or,
            with ``orb``, the orb's).
        custom: False after a reset: a double-click on the ring sends the HUD
            back to its corner; moving the HUD sends the orb back next to it.
        monitor: Device name of the monitor it is on.
        x: Logical pixels from the monitor work area's left edge.
        y: Logical pixels from its top edge.
        width: Logical width (the expanded panel's size).
        height: Logical height.
        orb: The minimized orb's placement.
    """

    expanded: bool
    custom: bool
    monitor: str = ""
    x: float = 0.0
    y: float = 0.0
    width: float = 0.0
    height: float = 0.0
    orb: bool = False


@dataclass(frozen=True, slots=True)
class Minimized:
    """The HUD shrank to the orb: its minimize control, or auto-minimize.

    Attributes:
        automatic: True when the auto-minimize policy did it; such an orb is
            not the user's choice and is not remembered for the next start.
    """

    automatic: bool = False


@dataclass(frozen=True, slots=True)
class Restored:
    """The orb grew back into the HUD: a click on it, or auto-minimize for a turn.

    Attributes:
        automatic: True when the auto-minimize policy did it.
    """

    automatic: bool = False


UiEvent = (
    SubmitText
    | CloseRequested
    | QuitRequested
    | ToggleMute
    | SetExpanded
    | SettingChanged
    | OpenLog
    | RestartRequested
    | SettingsOpened
    | PlacementChanged
    | Minimized
    | Restored
)
"""One thing the user did in the HUD."""

_SIMPLE: dict[str, UiEvent] = {
    "close": CloseRequested(),
    "quit": QuitRequested(),
    "toggle_mute": ToggleMute(),
    "open_log": OpenLog(),
    "restart": RestartRequested(),
    "settings_opened": SettingsOpened(),
}


def parse_ui_event(raw: Mapping[str, object]) -> UiEvent:
    """The typed event for one dict from the native queue.

    Raises:
        HudProtocolError: For an unknown kind or a malformed payload, which
            means the native module and this file disagree (a build mismatch).
    """
    kind = raw.get("kind")
    simple = _SIMPLE.get(str(kind))
    if simple is not None:
        return simple
    try:
        return _parse_payload(str(kind), raw)
    except (KeyError, TypeError, ValueError) as error:
        raise HudProtocolError(f"malformed HUD event {dict(raw)!r}: {error}") from error


def _parse_payload(kind: str, raw: Mapping[str, object]) -> UiEvent:
    if kind == "submit_text":
        return SubmitText(text=str(raw["text"]))
    if kind == "set_expanded":
        return SetExpanded(expanded=bool(raw["expanded"]))
    if kind == "setting_changed":
        return SettingChanged(key=str(raw["key"]), value=str(raw["value"]))
    if kind == "placement_changed":
        return PlacementChanged(
            expanded=bool(raw["expanded"]),
            custom=bool(raw["custom"]),
            monitor=str(raw.get("monitor", "")),
            x=_number(raw, "x"),
            y=_number(raw, "y"),
            width=_number(raw, "width"),
            height=_number(raw, "height"),
            orb=bool(raw.get("orb", False)),
        )
    if kind == "minimize":
        return Minimized(automatic=bool(raw["automatic"]))
    if kind == "restore":
        return Restored(automatic=bool(raw["automatic"]))
    raise HudProtocolError(f"unknown HUD event kind {kind!r}")


def _number(raw: Mapping[str, object], key: str) -> float:
    value = raw.get(key, 0.0)
    if not isinstance(value, int | float):
        raise TypeError(f"{key} must be a number, not {type(value).__name__}")
    return float(value)


def event_to_native(event: UiEvent) -> dict[str, object]:
    """Inverse of :func:`parse_ui_event`, for the native test hook."""
    for kind, simple in _SIMPLE.items():
        if event == simple:
            return {"kind": kind}
    if isinstance(event, SubmitText):
        return {"kind": "submit_text", "text": event.text}
    if isinstance(event, SetExpanded):
        return {"kind": "set_expanded", "expanded": event.expanded}
    if isinstance(event, SettingChanged):
        return {"kind": "setting_changed", "key": event.key, "value": event.value}
    if isinstance(event, Minimized):
        return {"kind": "minimize", "automatic": event.automatic}
    if isinstance(event, Restored):
        return {"kind": "restore", "automatic": event.automatic}
    raise HudProtocolError(f"{type(event).__name__} cannot be injected")


class HistoryRole(StrEnum):
    """Who a line of the expanded view's conversation belongs to."""

    USER = "user"
    JARVIS = "jarvis"
    TOOL = "tool"
    NOTICE = "notice"


class SettingKind(StrEnum):
    """How a setting is edited in the HUD."""

    CHOICE = "choice"
    TOGGLE = "toggle"
    SLIDER = "slider"
    ACTION = "action"


@dataclass(frozen=True, slots=True)
class SettingItem:
    """One row of the settings page, as the native side renders it.

    Attributes:
        key: Reported back in :class:`SettingChanged`; a ``Settings`` field
            name, or an action key (``open_log``, ``start_with_windows``).
        label: What the user reads.
        section: Heading the row is grouped under.
        kind: How it is edited.
        value: Current value as text (``"true"``, ``"0.80"``, a choice value).
        choices: ``(value, label)`` pairs for a choice.
        minimum: Slider minimum.
        maximum: Slider maximum.
        step: Slider step (0 for continuous).
        restart_required: Show "applies after restart".
        note: An extra line under the label.
    """

    key: str
    label: str
    section: str
    kind: SettingKind
    value: str = ""
    choices: Sequence[tuple[str, str]] = ()
    minimum: float = 0.0
    maximum: float = 1.0
    step: float = 0.0
    restart_required: bool = False
    note: str = ""

    def to_native(self) -> dict[str, object]:
        """The dict ``jarvis_overlay.OverlayWindow.set_settings`` expects per item."""
        return {
            "key": self.key,
            "label": self.label,
            "section": self.section,
            "kind": self.kind.value,
            "value": self.value,
            "choices": [(value, label) for value, label in self.choices],
            "minimum": self.minimum,
            "maximum": self.maximum,
            "step": self.step,
            "restart_required": self.restart_required,
            "note": self.note,
        }


@dataclass(frozen=True, slots=True)
class SettingsPage:
    """The whole settings page.

    Attributes:
        items: Rows in display order (grouped by their section).
        restart_pending: A restart-only setting changed: show "Restart".
        notice: A status line ("Saved.", or why a value was rejected).
    """

    items: Sequence[SettingItem] = field(default_factory=tuple)
    restart_pending: bool = False
    notice: str = ""

    def to_native(self) -> dict[str, object]:
        """The dict ``jarvis_overlay.OverlayWindow.set_settings`` expects."""
        return {
            "items": [item.to_native() for item in self.items],
            "restart_pending": self.restart_pending,
            "notice": self.notice,
        }


# Design notes:
# - Dicts at the boundary, dataclasses everywhere else: pybind11 turns a
#   dict into a native struct without a custom caster, fakes in tests need
#   no compiled module, and the dataclasses give mypy the real shapes.
# - parse_ui_event raises on anything unknown instead of skipping it: an
#   unknown kind means the native build and the Python code are from
#   different versions, which should be loud.
