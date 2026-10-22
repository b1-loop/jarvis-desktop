"""The HUD hotkey and the hotkey router, on StubPlatform (no real global hotkey is registered)."""

from __future__ import annotations

import asyncio

import pytest
from structlog.testing import capture_logs

from jarvis.core.hotkey_router import HotkeyRouter
from jarvis.core.hud_hotkey import HudHotkey, canonical_hotkey, same_hotkey
from jarvis.core.voice_triggers import HotkeyTrigger
from jarvis.os_integration.stub import StubPlatform
from jarvis.utils.errors import JarvisError

POLL = 0.001


class Toggles:
    """Counts toggles; can fail once to prove the listener survives."""

    def __init__(self, *, fail_first: bool = False) -> None:
        self.count = 0
        self._fail = fail_first

    async def __call__(self) -> None:
        if self._fail:
            self._fail = False
            raise JarvisError("could not save")
        self.count += 1


async def settle() -> None:
    """Give the listener a few polls."""
    await asyncio.sleep(POLL * 20)


def hud_id(platform: StubPlatform, spec: str = "ctrl+alt+h") -> int:
    return next(hotkey_id for hotkey_id, name in platform.hotkeys.items() if name == spec)


@pytest.mark.parametrize(
    ("first", "second", "same"),
    [
        ("ctrl+alt+j", "Alt + Ctrl + J", True),
        ("control+alt+j", "ctrl+alt+j", True),
        ("win+escape", "super+esc", True),
        ("ctrl+alt+h", "ctrl+alt+j", False),
        ("ctrl+alt+h", "ctrl+shift+h", False),
        ("", "", False),
    ],
)
def test_hotkeys_collide_by_their_keys_not_their_spelling(
    first: str, second: str, same: bool
) -> None:
    assert same_hotkey(first, second) is same


def test_the_canonical_form_ignores_order_case_and_spaces() -> None:
    assert canonical_hotkey(" Ctrl+ALT + h ") == frozenset({"ctrl", "alt", "h"})


async def test_a_press_toggles_the_hud_and_closing_releases_the_hotkey() -> None:
    platform = StubPlatform()
    toggles = Toggles()
    hotkey = HudHotkey(HotkeyRouter(platform).view(), "ctrl+alt+h", toggles, poll_interval=POLL)

    assert await hotkey.start()
    platform.press(hud_id(platform))
    platform.press(hud_id(platform))
    await settle()
    assert toggles.count == 2

    await hotkey.aclose()
    await hotkey.aclose()  # idempotent
    assert platform.hotkeys == {}
    assert not hotkey.active


async def test_the_voice_hotkey_is_never_taken() -> None:
    platform = StubPlatform()
    hotkey = HudHotkey(
        HotkeyRouter(platform).view(), "Alt+Ctrl+J", Toggles(), reserved=("ctrl+alt+j",)
    )

    with capture_logs() as logs:
        assert not await hotkey.start()

    assert platform.hotkeys == {}
    assert ("register_hotkey", ("Alt+Ctrl+J",)) not in platform.calls
    assert any(entry["event"] == "hud.hotkey.collides" for entry in logs)


async def test_a_combination_owned_by_another_program_is_skipped_cleanly() -> None:
    platform = StubPlatform(taken={"ctrl+alt+h"})
    hotkey = HudHotkey(HotkeyRouter(platform).view(), "ctrl+alt+h", Toggles())

    with capture_logs() as logs:
        assert not await hotkey.start()

    assert not hotkey.active
    assert any(entry["event"] == "hud.hotkey.unavailable" for entry in logs)
    await hotkey.aclose()  # nothing to release, no error


async def test_an_empty_setting_disables_the_hotkey() -> None:
    platform = StubPlatform()
    hotkey = HudHotkey(HotkeyRouter(platform).view(), "  ", Toggles())

    assert not await hotkey.start()
    assert platform.calls == []


async def test_a_failing_toggle_does_not_stop_the_listener() -> None:
    platform = StubPlatform()
    toggles = Toggles(fail_first=True)
    hotkey = HudHotkey(HotkeyRouter(platform).view(), "ctrl+alt+h", toggles, poll_interval=POLL)
    await hotkey.start()

    platform.press(hud_id(platform))
    await settle()
    platform.press(hud_id(platform))
    await settle()

    assert toggles.count == 1
    await hotkey.aclose()


async def test_the_voice_trigger_and_the_hud_hotkey_never_steal_each_others_presses() -> None:
    platform = StubPlatform()
    router = HotkeyRouter(platform)
    voice = HotkeyTrigger(router.view(), "ctrl+alt+j", poll_interval=POLL)
    await voice.register()
    toggles = Toggles()
    hud = HudHotkey(router.view(), "ctrl+alt+h", toggles, poll_interval=POLL)
    await hud.start()
    voice_id = hud_id(platform, "ctrl+alt+j")

    waiting = asyncio.create_task(voice.wait())
    await settle()  # the trigger drops presses from before it started waiting
    platform.press(hud_id(platform))
    platform.press(voice_id)
    hit = await asyncio.wait_for(waiting, 1.0)
    await settle()

    assert hit.source == "hotkey"
    assert toggles.count == 1
    await hud.aclose()
    await voice.aclose()


async def test_presses_of_unrouted_hotkeys_are_dropped() -> None:
    platform = StubPlatform()
    router = HotkeyRouter(platform)
    stray = await platform.register_hotkey("ctrl+alt+x")  # registered past the router
    view = router.view()
    mine = await view.register_hotkey("ctrl+alt+h")

    platform.press(stray)
    platform.press(mine)

    assert [press.hotkey_id for press in view.poll_hotkeys()] == [mine]
    assert view.poll_hotkeys() == []
    await view.unregister_hotkey(stray)  # not this view's: ignored
    assert stray in platform.hotkeys
