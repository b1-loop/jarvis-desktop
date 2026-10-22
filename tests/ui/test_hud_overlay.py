"""The interactive HUD through the overlay facade: events out, data in (ADR-0024)."""

from __future__ import annotations

import time
from datetime import datetime
from types import ModuleType

import pytest

from jarvis import _native
from jarvis.ui.hud_protocol import (
    CloseRequested,
    HistoryRole,
    HudProtocolError,
    PlacementChanged,
    SetExpanded,
    SettingChanged,
    SettingItem,
    SettingKind,
    SettingsPage,
    SubmitText,
    event_to_native,
    parse_ui_event,
)
from jarvis.ui.overlay import (
    Overlay,
    OverlayConfig,
    OverlayCorner,
    OverlayError,
    OverlayIntensity,
    OverlaySize,
    OverlayTheme,
    SavedPosition,
    StubOverlay,
    create_native_overlay,
)
from jarvis.utils.errors import NativeUnavailableError

PAGE = SettingsPage(
    items=(
        SettingItem(
            key="overlay_theme",
            label="Theme",
            section="OVERLAY",
            kind=SettingKind.CHOICE,
            value="cyan",
            choices=(("cyan", "Cyan"), ("amber", "Amber")),
        ),
        SettingItem(
            key="wake_word_threshold",
            label="Wake word threshold",
            section="VOICE",
            kind=SettingKind.SLIDER,
            value="0.80",
            minimum=0.5,
            maximum=0.95,
            step=0.05,
        ),
    ),
    restart_pending=True,
    notice="Saved.",
)


def test_every_event_kind_parses_into_its_type() -> None:
    assert parse_ui_event({"kind": "submit_text", "text": "hej"}) == SubmitText("hej")
    assert parse_ui_event({"kind": "close"}) == CloseRequested()
    assert parse_ui_event({"kind": "set_expanded", "expanded": True}) == SetExpanded(True)
    assert parse_ui_event({"kind": "setting_changed", "key": "k", "value": "v"}) == (
        SettingChanged("k", "v")
    )
    placement = parse_ui_event(
        {
            "kind": "placement_changed",
            "expanded": True,
            "custom": True,
            "monitor": "D",
            "x": 1,
            "y": 2.5,
            "width": 900,
            "height": 640,
        }
    )
    assert placement == PlacementChanged(True, True, "D", 1.0, 2.5, 900.0, 640.0)


@pytest.mark.parametrize(
    "raw",
    [
        {"kind": "warp"},
        {"kind": "submit_text"},
        {"kind": "placement_changed", "expanded": True, "custom": True, "x": "far"},
    ],
)
def test_malformed_events_are_loud(raw: dict[str, object]) -> None:
    with pytest.raises(HudProtocolError):
        parse_ui_event(raw)


def test_events_round_trip_through_the_injection_format() -> None:
    for event in (
        SubmitText("åäö"),
        CloseRequested(),
        SetExpanded(False),
        SettingChanged("a", "b"),
    ):
        assert parse_ui_event(event_to_native(event)) == event


def test_the_settings_page_crosses_as_plain_data() -> None:
    native = PAGE.to_native()
    items = native["items"]
    assert isinstance(items, list)
    assert items[0]["choices"] == [("cyan", "Cyan"), ("amber", "Amber")]
    assert items[1]["kind"] == "slider"
    assert native["restart_pending"] is True


def test_the_facade_forwards_hud_data_and_polls_typed_events() -> None:
    native = StubOverlay()
    overlay = Overlay(native)

    overlay.set_muted(True)
    overlay.add_history(HistoryRole.JARVIS, "Two meetings.", at=datetime(2026, 10, 22, 14, 5))
    overlay.set_settings(PAGE)
    overlay.set_appearance(OverlayTheme.AMBER, OverlayIntensity.MAX, OverlaySize.COMPACT)
    overlay.set_expanded(True)
    overlay.inject_ui_event(SubmitText("hej"))

    assert native.muted
    assert native.history == [("jarvis", "Two meetings.", "14:05")]
    assert native.settings == PAGE.to_native()
    assert native.appearance == ("amber", "max", "compact")
    assert overlay.expanded
    assert overlay.poll_ui_events() == [SubmitText("hej")]
    assert overlay.poll_ui_events() == []


def test_a_closed_facade_polls_nothing() -> None:
    native = StubOverlay()
    overlay = Overlay(native)
    native.inject_ui_event({"kind": "close"})
    overlay.close()

    assert overlay.poll_ui_events() == []
    assert not overlay.expanded


# --- against the compiled module -------------------------------------------


@pytest.fixture
def jarvis_overlay() -> ModuleType:
    try:
        return _native.load("jarvis_overlay")
    except NativeUnavailableError:
        pytest.skip("jarvis_overlay is not built")


def open_native(config: OverlayConfig) -> Overlay:
    try:
        return create_native_overlay(config)
    except OverlayError as exc:
        pytest.skip(f"overlay window unavailable here: {exc}")


@pytest.mark.native
def test_native_smoke_a_typed_request_reaches_python(jarvis_overlay: ModuleType) -> None:
    del jarvis_overlay
    overlay = open_native(OverlayConfig(corner=OverlayCorner.TOP_LEFT, boot_animation=False))
    with overlay:
        overlay.inject_ui_event(SubmitText("vad är klockan i Tokyo?"))
        overlay.inject_ui_event(SettingChanged("overlay_theme", "amber"))

        assert overlay.poll_ui_events() == [
            SubmitText("vad är klockan i Tokyo?"),
            SettingChanged("overlay_theme", "amber"),
        ]
        assert overlay.poll_ui_events() == []


@pytest.mark.native
def test_native_hud_data_is_accepted_and_rendered(jarvis_overlay: ModuleType) -> None:
    del jarvis_overlay
    config = OverlayConfig(
        corner=OverlayCorner.TOP_LEFT,
        boot_animation=False,
        position=SavedPosition(monitor="NO-SUCH-DISPLAY", x=30.0, y=30.0),
    )
    overlay = open_native(config)
    with overlay:
        overlay.set_muted(True)
        overlay.add_history(HistoryRole.USER, "hej")
        overlay.add_history(HistoryRole.TOOL, "calendar_agenda · 120 ms")
        overlay.set_settings(PAGE)
        overlay.set_appearance(OverlayTheme.AMBER, OverlayIntensity.CALM, OverlaySize.NORMAL)
        overlay.set_expanded(True)
        time.sleep(0.6)  # the morph runs about 0.4 s

        hud = overlay._native.hud_snapshot()  # type: ignore[attr-defined]
        assert hud["muted"] is True
        assert [text for _role, text, _time in hud["history"]] == [
            "hej",
            "calendar_agenda · 120 ms",
        ]
        assert hud["setting_keys"] == ["overlay_theme", "wake_word_threshold"]
        assert (hud["theme"], hud["intensity"], hud["size"]) == ("amber", "calm", "normal")
        assert overlay.expanded
        assert overlay.running
