"""Minimizing the HUD to its orb through the overlay facade (ADR-0024, update)."""

from __future__ import annotations

import time
from types import ModuleType

import pytest

from jarvis import _native
from jarvis.ui.hud_protocol import (
    Minimized,
    PlacementChanged,
    Restored,
    event_to_native,
    parse_ui_event,
)
from jarvis.ui.overlay import (
    Overlay,
    OverlayConfig,
    OverlayCorner,
    OverlayError,
    OverlayState,
    SavedPosition,
    StubOverlay,
    create_native_overlay,
)
from jarvis.utils.errors import NativeUnavailableError

MORPH_SECONDS = 0.9
"""The 0.5 s morph plus a few idle-rate frames of slack."""


def test_minimize_and_restore_events_parse_with_their_flag() -> None:
    assert parse_ui_event({"kind": "minimize", "automatic": True}) == Minimized(automatic=True)
    assert parse_ui_event({"kind": "restore", "automatic": False}) == Restored()
    for event in (Minimized(), Minimized(automatic=True), Restored(), Restored(automatic=True)):
        assert parse_ui_event(event_to_native(event)) == event


def test_an_orb_placement_is_told_apart_from_the_huds() -> None:
    raw: dict[str, object] = {
        "kind": "placement_changed",
        "expanded": False,
        "custom": True,
        "monitor": "D",
        "x": 12,
        "y": 900,
    }
    assert parse_ui_event(raw) == PlacementChanged(False, True, "D", 12.0, 900.0)
    assert parse_ui_event({**raw, "orb": True}) == PlacementChanged(
        False, True, "D", 12.0, 900.0, orb=True
    )


def test_the_facade_minimizes_restores_and_forwards_the_options() -> None:
    native = StubOverlay()
    overlay = Overlay(native)
    overlay.set_expanded(True)

    overlay.minimize()
    assert overlay.minimized
    assert not overlay.expanded  # minimizing collapses first, like the window
    overlay.restore()
    assert not overlay.minimized

    overlay.set_auto_minimize(45)
    overlay.set_hide_in_fullscreen(False)
    assert native.auto_minimize_seconds == 45.0
    assert native.hide_in_fullscreen is False


def test_the_stub_never_shrinks_by_itself_and_reports_nothing() -> None:
    native = StubOverlay()
    overlay = Overlay(native)
    overlay.set_auto_minimize(-3)
    overlay.minimize()

    assert native.auto_minimize_seconds == 0.0
    assert overlay.poll_ui_events() == []


def test_a_closed_facade_is_never_minimized() -> None:
    native = StubOverlay()
    overlay = Overlay(native)
    overlay.minimize()
    overlay.close()

    assert not overlay.minimized
    overlay.restore()  # ignored after close
    assert native.minimized()


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
def test_native_smoke_minimize_and_restore_the_real_overlay(jarvis_overlay: ModuleType) -> None:
    del jarvis_overlay
    overlay = open_native(OverlayConfig(corner=OverlayCorner.TOP_LEFT, boot_animation=False))
    with overlay:
        overlay.minimize()
        assert overlay.minimized
        time.sleep(MORPH_SECONDS)
        hud = overlay._native.hud_snapshot()  # type: ignore[attr-defined]
        assert hud["minimized"] is True
        assert overlay.running

        overlay.restore()
        time.sleep(MORPH_SECONDS)
        assert not overlay.minimized
        assert overlay.running
        # Asked for from Python: nothing is echoed back as an event.
        assert overlay.poll_ui_events() == []


@pytest.mark.native
def test_native_auto_minimize_around_a_turn(jarvis_overlay: ModuleType) -> None:
    del jarvis_overlay
    config = OverlayConfig(
        corner=OverlayCorner.TOP_LEFT, boot_animation=False, auto_minimize_seconds=0.3
    )
    overlay = open_native(config)
    with overlay:
        time.sleep(MORPH_SECONDS)
        assert overlay.minimized
        assert Minimized(automatic=True) in overlay.poll_ui_events()

        # The wake word starts a turn: the orb pops back to the full HUD ...
        overlay.set_state(OverlayState.LISTENING)
        overlay.wake_pulse()
        time.sleep(0.2)
        assert not overlay.minimized
        assert Restored(automatic=True) in overlay.poll_ui_events()

        # ... and returns after the turn plus the delay.
        overlay.set_state(OverlayState.IDLE)
        time.sleep(MORPH_SECONDS)
        assert overlay.minimized


@pytest.mark.native
def test_native_start_minimized_with_a_remembered_orb(jarvis_overlay: ModuleType) -> None:
    del jarvis_overlay
    config = OverlayConfig(
        corner=OverlayCorner.TOP_LEFT,
        boot_animation=False,
        start_minimized=True,
        orb_position=SavedPosition(monitor="NO-SUCH-DISPLAY", x=40.0, y=40.0),
        hide_in_fullscreen=False,
    )
    overlay = open_native(config)
    with overlay:
        assert overlay.minimized
        time.sleep(MORPH_SECONDS)
        hud = overlay._native.hud_snapshot()  # type: ignore[attr-defined]
        assert hud["hide_in_fullscreen"] is False
        assert overlay.running
