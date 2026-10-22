from collections.abc import Callable

from jarvis.config import Settings
from jarvis.os_integration import StubPlatform
from jarvis.tools.open_app import OpenAppTool
from tests.tools.local_fakes import local_context


async def test_launches_the_target_and_reports_the_pid(
    make_settings: Callable[..., Settings],
) -> None:
    platform = StubPlatform()

    result = await OpenAppTool().call(
        {"target": "notepad", "arguments": "todo.txt"},
        local_context(make_settings(), platform=platform),
    )

    assert result.ok
    assert result.spoken == "Opening notepad."
    assert platform.calls == [("launch", ("notepad", "todo.txt"))]
    assert result.data == {"target": "notepad", "arguments": "todo.txt", "pid": 4000}


async def test_fails_clearly_without_a_platform(make_settings: Callable[..., Settings]) -> None:
    result = await OpenAppTool().call({"target": "notepad"}, local_context(make_settings()))

    assert not result.ok
    assert "operating-system integration is not available" in result.detail


async def test_rejects_an_empty_target(make_settings: Callable[..., Settings]) -> None:
    result = await OpenAppTool().call({"target": ""}, local_context(make_settings()))

    assert not result.ok
    assert "target" in result.detail


def test_needs_no_confirmation() -> None:
    assert OpenAppTool.requires_confirmation is False
