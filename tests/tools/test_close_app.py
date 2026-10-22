from collections.abc import Callable
from dataclasses import dataclass

import pytest

from jarvis.config import Settings
from jarvis.os_integration import PlatformError, StubPlatform
from jarvis.tools.close_app import CloseAppInput, CloseAppTool
from tests.tools.local_fakes import NOTEPAD, local_context


@dataclass
class RefusingPlatform(StubPlatform):
    """Graceful close is impossible (no window), as for a tray-only process."""

    async def close_process_by_name(self, name: str, *, force: bool = False) -> int:
        raise PlatformError(f"{name} has no window to close")


@pytest.fixture
def settings(make_settings: Callable[..., Settings]) -> Settings:
    return make_settings()


def test_always_requires_confirmation_and_warns_harder_on_force() -> None:
    tool = CloseAppTool()

    gentle = tool.confirmation_prompt(CloseAppInput(name="notepad.exe"))
    forced = tool.confirmation_prompt(CloseAppInput(name="notepad.exe", force=True))

    assert tool.requires_confirmation
    assert gentle == "Do you want me to close notepad.exe?"
    assert "lose any unsaved work" in forced
    assert "kill" in forced


async def test_closes_gracefully_by_default(settings: Settings) -> None:
    platform = StubPlatform(windows=[NOTEPAD])

    result = await CloseAppTool().call(
        {"name": "notepad.exe"}, local_context(settings, platform=platform)
    )

    assert result.ok
    assert result.spoken == "Closed notepad.exe."
    assert platform.calls == [("close_process_by_name", ("notepad.exe", False))]
    assert result.data == {"name": "notepad.exe", "force": False, "count": 1}


async def test_force_is_passed_through(settings: Settings) -> None:
    platform = StubPlatform(windows=[NOTEPAD])

    result = await CloseAppTool().call(
        {"name": "notepad.exe", "force": True}, local_context(settings, platform=platform)
    )

    assert result.spoken == "Killed notepad.exe."
    assert platform.calls == [("close_process_by_name", ("notepad.exe", True))]


async def test_reports_an_app_that_is_not_running(settings: Settings) -> None:
    result = await CloseAppTool().call(
        {"name": "ghost.exe"}, local_context(settings, platform=StubPlatform())
    )

    assert not result.ok
    assert result.spoken == "ghost.exe isn't running."


async def test_offers_to_force_when_a_graceful_close_is_refused(settings: Settings) -> None:
    result = await CloseAppTool().call(
        {"name": "tray.exe"}, local_context(settings, platform=RefusingPlatform())
    )

    assert not result.ok
    assert "I can force it" in result.spoken


async def test_a_failed_forced_close_reports_the_platform_error(settings: Settings) -> None:
    result = await CloseAppTool().call(
        {"name": "tray.exe", "force": True}, local_context(settings, platform=RefusingPlatform())
    )

    assert not result.ok
    assert "has no window to close" in result.detail
