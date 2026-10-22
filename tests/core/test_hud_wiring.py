"""What the HUD's actions do to the session."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path

from jarvis.config import Settings
from jarvis.core.hud_wiring import EXPANDED_BOUNDS_KEY, POSITION_KEY, HudActions
from jarvis.core.ui_wiring import AppControls
from jarvis.ui.hud_protocol import HistoryRole, PlacementChanged, SettingsPage
from jarvis.ui.hud_settings import SettingsController
from jarvis.user_settings import UserSettingsStore, user_settings_path

MakeSettings = Callable[..., Settings]


class FakeOverlay:
    def __init__(self) -> None:
        self.hidden = 0
        self.transcripts: list[str] = []
        self.history: list[tuple[HistoryRole, str]] = []
        self.pages: list[SettingsPage] = []
        self.is_minimized = False

    @property
    def minimized(self) -> bool:
        return self.is_minimized

    def minimize(self) -> None:
        self.is_minimized = True

    def restore(self) -> None:
        self.is_minimized = False

    def hide(self) -> None:
        self.hidden += 1

    def set_transcript(self, text: str) -> None:
        self.transcripts.append(text)

    def add_history(self, role: HistoryRole, text: str) -> None:
        self.history.append((role, text))

    def set_settings(self, page: SettingsPage) -> None:
        self.pages.append(page)


class FakeTyped:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def submit(self, text: str) -> None:
        self.lines.append(text)


class FakeMicrophone:
    def __init__(self) -> None:
        self.muted = False

    async def toggle(self) -> bool:
        self.muted = not self.muted
        return self.muted


def actions(
    make_settings: MakeSettings, tmp_path: Path, **kwargs: object
) -> tuple[HudActions, FakeOverlay, FakeTyped, UserSettingsStore]:
    overlay, typed = FakeOverlay(), FakeTyped()
    store = UserSettingsStore(user_settings_path(tmp_path))
    settings = SettingsController(settings=make_settings(), store=store, view=overlay, appliers={})
    built = HudActions(
        overlay=overlay,
        typed=typed,
        settings=settings,
        store=store,
        **kwargs,  # type: ignore[arg-type]
    )
    return built, overlay, typed, store


def test_a_typed_request_is_shown_recorded_and_handed_on(
    make_settings: MakeSettings, tmp_path: Path
) -> None:
    hud, overlay, typed, _store = actions(make_settings, tmp_path)

    hud.submit("vad är klockan?")

    assert overlay.transcripts == ["vad är klockan?"]
    assert overlay.history == [(HistoryRole.USER, "vad är klockan?")]
    assert typed.lines == ["vad är klockan?"]


def test_close_hides_and_quit_goes_through_the_app_controls(
    make_settings: MakeSettings, tmp_path: Path
) -> None:
    hud, overlay, _typed, _store = actions(make_settings, tmp_path)
    stopped: list[bool] = []

    hud.quit()  # not bound yet: logged, nothing happens
    hud.bind_controls(AppControls(stop=lambda: stopped.append(True)))
    hud.close()
    hud.quit()

    assert overlay.hidden == 1
    assert stopped == [True]


async def test_mute_uses_the_microphone_until_the_controls_are_bound(
    make_settings: MakeSettings, tmp_path: Path
) -> None:
    microphone = FakeMicrophone()
    hud, _overlay, _typed, _store = actions(make_settings, tmp_path, microphone=microphone)

    await hud.toggle_mute()
    assert microphone.muted

    hud.bind_controls(AppControls(stop=lambda: None, microphone=microphone))
    await hud.toggle_mute()
    await asyncio.sleep(0.01)  # the controls flip the microphone in a task
    assert not microphone.muted


async def test_placements_are_remembered_and_reset(
    make_settings: MakeSettings, tmp_path: Path
) -> None:
    hud, _overlay, _typed, store = actions(make_settings, tmp_path)

    await hud.placement_changed(
        PlacementChanged(expanded=False, custom=True, monitor="DISPLAY2", x=40.0, y=60.0)
    )
    await hud.placement_changed(
        PlacementChanged(
            expanded=True, custom=True, monitor="DISPLAY1", x=0, y=0, width=900, height=640
        )
    )
    assert store.load()[POSITION_KEY] == {"monitor": "DISPLAY2", "x": 40.0, "y": 60.0}
    assert store.load()[EXPANDED_BOUNDS_KEY]["width"] == 900

    await hud.placement_changed(PlacementChanged(expanded=False, custom=False))
    assert store.load()[POSITION_KEY] is None
    assert load_position(make_settings, tmp_path) is None


def load_position(make_settings: MakeSettings, tmp_path: Path) -> object:
    return make_settings(data_dir=tmp_path, user_settings_enabled=True).overlay_position


def test_restart_and_log_say_why_when_unavailable(
    make_settings: MakeSettings, tmp_path: Path
) -> None:
    hud, overlay, _typed, _store = actions(make_settings, tmp_path)

    hud.restart()
    assert "start it again" in overlay.pages[-1].notice
    hud.open_log()
    assert "terminal" in overlay.pages[-1].notice


def test_restart_arranges_a_relaunch_and_quits(make_settings: MakeSettings, tmp_path: Path) -> None:
    requested: list[bool] = []
    stopped: list[bool] = []
    opened: list[bool] = []
    hud, _overlay, _typed, _store = actions(
        make_settings,
        tmp_path,
        restart=lambda: requested.append(True),
        open_log=lambda: opened.append(True),
    )
    hud.bind_controls(AppControls(stop=lambda: stopped.append(True)))

    hud.restart()
    hud.open_log()

    assert requested == [True] and stopped == [True] and opened == [True]
