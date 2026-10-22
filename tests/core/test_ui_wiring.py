import asyncio
from collections.abc import Callable
from contextlib import AsyncExitStack
from dataclasses import dataclass, field

from jarvis.config import Settings
from jarvis.core.events import EventBus
from jarvis.core.ui_wiring import AppControls, build_dashboard, build_tray
from jarvis.ui.dashboard import StubDashboard
from jarvis.ui.tray import StubTray
from tests.core.voice_fakes import eventually

MakeSettings = Callable[..., Settings]


@dataclass
class Microphone:
    muted: bool = False

    async def toggle(self) -> bool:
        self.muted = not self.muted
        return self.muted


@dataclass
class Dashboard:
    shown: int = 0

    def show(self) -> None:
        self.shown += 1


@dataclass
class Tray:
    muted: list[bool] = field(default_factory=list)

    def set_microphone_muted(self, muted: bool) -> None:
        self.muted.append(muted)


async def test_menu_actions_reach_the_session() -> None:
    stops: list[int] = []
    overlay_toggles: list[int] = []
    dashboard, tray, microphone = Dashboard(), Tray(), Microphone()
    controls = AppControls(
        stop=lambda: stops.append(1),
        dashboard=dashboard,  # type: ignore[arg-type]
        toggle_overlay=lambda: overlay_toggles.append(1),
        microphone=microphone,
    )
    controls.attach_tray(tray)  # type: ignore[arg-type]

    controls.show_dashboard()
    controls.toggle_overlay()
    controls.toggle_microphone()
    await eventually(lambda: tray.muted)
    controls.quit()

    assert dashboard.shown == 1
    assert overlay_toggles == [1]
    assert tray.muted == [True]
    assert stops == [1]


async def test_missing_parts_are_logged_no_ops() -> None:
    controls = AppControls(stop=lambda: None)

    controls.show_dashboard()
    controls.toggle_overlay()
    controls.toggle_microphone()
    await asyncio.sleep(0)


async def test_disabled_dashboard_and_tray_are_stubs(make_settings: MakeSettings) -> None:
    settings = make_settings(dashboard_enabled=False, tray_enabled=False)
    bus = EventBus()
    async with AsyncExitStack() as stack:
        dashboard = await build_dashboard(settings, bus, stack, memory=None)
        tray = await build_tray(settings, bus, stack, AppControls(stop=lambda: None))

        assert isinstance(dashboard, StubDashboard)
        assert isinstance(tray, StubTray)
