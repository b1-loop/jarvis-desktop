from collections.abc import Callable

from jarvis.config import Settings
from jarvis.os_integration import StubPlatform
from jarvis.tools.notify import NotifyTool
from tests.tools.local_fakes import local_context


async def test_shows_a_notification_with_a_default_title(
    make_settings: Callable[..., Settings],
) -> None:
    platform = StubPlatform()

    result = await NotifyTool().call(
        {"message": "Wi-Fi password: hunter2"}, local_context(make_settings(), platform=platform)
    )

    assert result.ok
    assert result.spoken == "I've put it on screen."
    assert platform.notifications == [("Jarvis", "Wi-Fi password: hunter2")]


async def test_rejects_an_overlong_message(make_settings: Callable[..., Settings]) -> None:
    result = await NotifyTool().call(
        {"message": "x" * 1000}, local_context(make_settings(), platform=StubPlatform())
    )

    assert not result.ok
