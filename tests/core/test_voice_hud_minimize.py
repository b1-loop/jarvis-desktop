"""The voice front end's HUD minimize wiring: the hotkey, the tray entry, the start state.

Everything runs on StubPlatform (no real global hotkey is ever registered)
and StubOverlay; user settings go to a temporary directory.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from contextlib import AsyncExitStack
from dataclasses import replace
from pathlib import Path

import pytest

from jarvis.config import HudPosition, Settings
from jarvis.core import voice_bootstrap
from jarvis.core.voice_bootstrap import VoiceFrontEndBuilder, overlay_config
from jarvis.os_integration.stub import StubPlatform
from jarvis.ui.hud_protocol import SettingChanged
from jarvis.ui.overlay import Overlay, SavedPosition, StubOverlay
from tests.core.test_voice_bootstrap import factories, request

MakeSettings = Callable[..., Settings]


def specs(platform: StubPlatform) -> set[str]:
    return set(platform.hotkeys.values())


def builder_with(overlay: Overlay) -> VoiceFrontEndBuilder:
    return VoiceFrontEndBuilder(
        factories=replace(factories(), overlay=lambda _settings: overlay), write=lambda _t: None
    )


async def test_the_hud_hotkey_minimizes_and_restores(
    make_settings: MakeSettings, tmp_path: Path
) -> None:
    platform = StubPlatform()
    overlay = Overlay(StubOverlay())
    async with AsyncExitStack() as stack:
        front = await builder_with(overlay)(
            request(make_settings(data_dir=tmp_path), stack, platform)
        )
        assert specs(platform) == {"ctrl+alt+j", "ctrl+alt+h"}
        hud_id = next(i for i, spec in platform.hotkeys.items() if spec == "ctrl+alt+h")

        platform.press(hud_id)
        await asyncio.sleep(0.15)
        assert overlay.minimized
        platform.press(hud_id)
        await asyncio.sleep(0.15)
        assert not overlay.minimized
        assert front.toggle_hud_minimized is not None
    assert platform.hotkeys == {}  # both released with the session


async def test_the_tray_entry_toggles_the_hud(make_settings: MakeSettings, tmp_path: Path) -> None:
    overlay = Overlay(StubOverlay())
    async with AsyncExitStack() as stack:
        front = await builder_with(overlay)(request(make_settings(data_dir=tmp_path), stack))
        assert front.toggle_hud_minimized is not None

        await front.toggle_hud_minimized()
        assert overlay.minimized
    assert make_settings(data_dir=tmp_path, user_settings_enabled=True).hud_minimized is True


async def test_a_launch_while_running_shows_the_hud_and_replays_the_boot(
    make_settings: MakeSettings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(voice_bootstrap, "default_microphone_name", lambda: "Test Mic")
    native = StubOverlay()
    overlay = Overlay(native)
    async with AsyncExitStack() as stack:
        front = await builder_with(overlay)(request(make_settings(data_dir=tmp_path), stack))
        assert front.bring_hud_forward is not None
        overlay.minimize()
        overlay.hide()

        await front.bring_hud_forward()

        assert overlay.visible
        assert not overlay.minimized
        assert native.boot_replays == 1
        assert dict(native.boot_info)["MIC"] == "Test Mic"
    assert make_settings(data_dir=tmp_path, user_settings_enabled=True).hud_minimized is False


def test_a_first_launch_boots_even_when_the_hud_was_left_minimized(
    make_settings: MakeSettings,
) -> None:
    settings = make_settings(overlay_enabled=False, hud_minimized=True)

    overlay = voice_bootstrap.build_overlay(settings)

    assert isinstance(overlay._native, StubOverlay)
    assert overlay._native.boot_started  # the native window boots in full, then shrinks
    assert overlay_config(settings).start_minimized


async def test_a_hud_hotkey_equal_to_the_voice_hotkey_is_skipped(
    make_settings: MakeSettings, tmp_path: Path
) -> None:
    platform = StubPlatform()
    settings = make_settings(data_dir=tmp_path, hud_toggle_hotkey="Alt+Ctrl+J")
    async with AsyncExitStack() as stack:
        await builder_with(Overlay(StubOverlay()))(request(settings, stack, platform))
        assert specs(platform) == {"ctrl+alt+j"}


async def test_a_taken_hud_hotkey_leaves_the_voice_hotkey_working(
    make_settings: MakeSettings, tmp_path: Path
) -> None:
    platform = StubPlatform(taken={"ctrl+alt+h"})
    async with AsyncExitStack() as stack:
        front = await builder_with(Overlay(StubOverlay()))(
            request(make_settings(data_dir=tmp_path), stack, platform)
        )
        assert specs(platform) == {"ctrl+alt+j"}
        assert front.toggle_hud_minimized is not None  # the tray entry still works


async def test_an_empty_hud_hotkey_registers_nothing_extra(
    make_settings: MakeSettings, tmp_path: Path
) -> None:
    platform = StubPlatform()
    settings = make_settings(data_dir=tmp_path, hud_toggle_hotkey="")
    async with AsyncExitStack() as stack:
        await builder_with(Overlay(StubOverlay()))(request(settings, stack, platform))
        assert specs(platform) == {"ctrl+alt+j"}


def test_the_overlay_starts_as_it_was_left(make_settings: MakeSettings) -> None:
    settings = make_settings(
        hud_minimized=True,
        hud_orb_position=HudPosition(monitor="DISPLAY2", x=12, y=900),
        hud_auto_minimize_seconds=60,
        hud_hide_in_fullscreen=False,
    )

    config = overlay_config(settings)

    assert config.start_minimized is True
    assert config.orb_position == SavedPosition(monitor="DISPLAY2", x=12, y=900)
    assert config.auto_minimize_seconds == 60.0
    assert config.hide_in_fullscreen is False


def test_the_defaults_are_a_full_hud_with_full_screen_hiding(make_settings: MakeSettings) -> None:
    settings = make_settings()
    config = overlay_config(settings)

    assert settings.hud_toggle_hotkey == "ctrl+alt+h"
    assert (config.start_minimized, config.orb_position) == (False, None)
    assert (config.auto_minimize_seconds, config.hide_in_fullscreen) == (0.0, True)


async def test_the_settings_page_applies_the_minimize_options_live(
    make_settings: MakeSettings, tmp_path: Path
) -> None:
    native = StubOverlay()
    overlay = Overlay(native)
    async with AsyncExitStack() as stack:
        await builder_with(overlay)(request(make_settings(data_dir=tmp_path), stack))
        # What the HUD's settings page sends when the user picks a value.
        overlay.inject_ui_event(SettingChanged("hud_auto_minimize_seconds", "60.00"))
        overlay.inject_ui_event(SettingChanged("hud_hide_in_fullscreen", "false"))
        await asyncio.sleep(0.2)  # the bridge drains every 30 ms

        assert native.auto_minimize_seconds == 60.0
        assert native.hide_in_fullscreen is False
    saved = make_settings(data_dir=tmp_path, user_settings_enabled=True)
    assert (saved.hud_auto_minimize_seconds, saved.hud_hide_in_fullscreen) == (60.0, False)
