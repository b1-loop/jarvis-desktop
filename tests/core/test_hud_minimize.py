"""Minimizing the HUD to its orb: what is remembered, what the tray shows (ADR-0024, update)."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from jarvis.config import HudPosition, Settings
from jarvis.core.events import EventBus
from jarvis.core.hud_wiring import MINIMIZED_KEY, ORB_POSITION_KEY, POSITION_KEY, HudActions
from jarvis.core.ui_wiring import AppControls
from jarvis.ui.hud_protocol import PlacementChanged
from jarvis.ui.tray import StubTray
from tests.core.test_hud_wiring import actions

MakeSettings = Callable[..., Settings]


def reloaded(make_settings: MakeSettings, tmp_path: Path) -> Settings:
    """The settings the next start would see (user_settings.json layered in)."""
    return make_settings(data_dir=tmp_path, user_settings_enabled=True)


def bound_tray(hud: HudActions) -> StubTray:
    """Bind ``hud`` to controls with a stub tray, as the composition root does."""
    tray = StubTray(EventBus())
    controls = AppControls(stop=lambda: None)
    controls.attach_tray(tray)
    hud.bind_controls(controls)
    return tray


async def test_a_launch_while_running_restores_the_orb_and_remembers_it(
    make_settings: MakeSettings, tmp_path: Path
) -> None:
    hud, overlay, _typed, _store = actions(make_settings, tmp_path)
    overlay.is_minimized = True
    overlay.is_expanded = True
    tray = bound_tray(hud)
    assert tray.hud_minimized

    await hud.bring_forward()

    assert not overlay.minimized
    assert not overlay.expanded  # the boot hands off into the corner HUD
    assert not tray.hud_minimized
    assert reloaded(make_settings, tmp_path).hud_minimized is False
    assert overlay.replays == [None]  # no readout source: the boot plays without one


async def test_the_users_minimize_and_restore_are_remembered(
    make_settings: MakeSettings, tmp_path: Path
) -> None:
    hud, _overlay, _typed, store = actions(make_settings, tmp_path)

    await hud.minimized(automatic=False)
    assert store.load()[MINIMIZED_KEY] is True
    assert reloaded(make_settings, tmp_path).hud_minimized is True

    await hud.restored(automatic=False)
    assert reloaded(make_settings, tmp_path).hud_minimized is False


async def test_an_automatic_orb_is_never_remembered(
    make_settings: MakeSettings, tmp_path: Path
) -> None:
    hud, _overlay, _typed, store = actions(make_settings, tmp_path)

    await hud.minimized(automatic=True)
    await hud.restored(automatic=True)
    await hud.minimized(automatic=True)

    assert MINIMIZED_KEY not in store.load()
    assert reloaded(make_settings, tmp_path).hud_minimized is False


async def test_toggling_flips_the_hud_saves_it_and_relabels_the_tray(
    make_settings: MakeSettings, tmp_path: Path
) -> None:
    hud, overlay, _typed, _store = actions(make_settings, tmp_path)
    tray = bound_tray(hud)
    assert not tray.hud_minimized

    await hud.toggle_minimized()
    assert overlay.minimized and tray.hud_minimized
    assert reloaded(make_settings, tmp_path).hud_minimized is True

    await hud.toggle_minimized()
    assert not overlay.minimized and not tray.hud_minimized
    assert reloaded(make_settings, tmp_path).hud_minimized is False


async def test_events_from_the_hud_relabel_the_tray(
    make_settings: MakeSettings, tmp_path: Path
) -> None:
    hud, _overlay, _typed, _store = actions(make_settings, tmp_path)
    tray = bound_tray(hud)

    await hud.minimized(automatic=True)
    assert tray.hud_minimized
    await hud.restored(automatic=True)
    assert not tray.hud_minimized


async def test_a_hud_that_started_minimized_shows_restore_in_the_tray(
    make_settings: MakeSettings, tmp_path: Path
) -> None:
    hud, overlay, _typed, _store = actions(make_settings, tmp_path)
    overlay.is_minimized = True

    tray = bound_tray(hud)

    assert tray.hud_minimized


async def test_the_orbs_position_is_remembered_and_forgotten(
    make_settings: MakeSettings, tmp_path: Path
) -> None:
    hud, _overlay, _typed, store = actions(make_settings, tmp_path)

    await hud.placement_changed(
        PlacementChanged(expanded=False, custom=True, monitor="DISPLAY1", x=12, y=900, orb=True)
    )
    assert reloaded(make_settings, tmp_path).hud_orb_position == HudPosition(
        monitor="DISPLAY1", x=12, y=900
    )
    assert POSITION_KEY not in store.load()  # the HUD's own place is untouched

    await hud.placement_changed(PlacementChanged(expanded=False, custom=False, orb=True))
    assert store.load()[ORB_POSITION_KEY] is None
    assert reloaded(make_settings, tmp_path).hud_orb_position is None
