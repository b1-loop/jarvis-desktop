from collections.abc import Callable
from dataclasses import dataclass

import pytest

from jarvis.config import Settings
from jarvis.os_integration import StubPlatform, WindowNotFoundError
from jarvis.tools.window_action import WindowActionInput, WindowActionTool
from tests.tools.local_fakes import CODE, NOTEPAD, local_context


@dataclass
class VanishingPlatform(StubPlatform):
    """The window closes between lookup and action."""

    async def minimize_window(self, handle: int) -> None:
        raise WindowNotFoundError(f"handle {handle} is stale")


@pytest.fixture
def settings(make_settings: Callable[..., Settings]) -> Settings:
    return make_settings()


@pytest.mark.parametrize(
    ("action", "expected"),
    [("minimize", False), ("maximize", False), ("restore", False), ("close", True)],
)
def test_only_close_needs_confirmation(settings: Settings, action: str, expected: bool) -> None:
    params = WindowActionInput.model_validate({"action": action})

    assert WindowActionTool().needs_confirmation(params, local_context(settings)) is expected


def test_close_prompt_names_the_window() -> None:
    tool = WindowActionTool()

    named = tool.confirmation_prompt(WindowActionInput(action="close", title="Notepad"))
    current = tool.confirmation_prompt(WindowActionInput(action="close"))

    assert named == "Do you want me to close the Notepad window?"
    assert current == "Do you want me to close the current window?"


@pytest.mark.parametrize(
    ("action", "call", "spoken"),
    [
        ("minimize", "minimize_window", "Minimised"),
        ("maximize", "maximize_window", "Maximised"),
        ("restore", "restore_window", "Restored"),
    ],
)
async def test_acts_on_the_focused_window_by_default(
    settings: Settings, action: str, call: str, spoken: str
) -> None:
    platform = StubPlatform(windows=[NOTEPAD, CODE], active=CODE)

    result = await WindowActionTool().call(
        {"action": action}, local_context(settings, platform=platform)
    )

    assert result.ok
    assert result.spoken == f"{spoken} {CODE.title}."
    assert (call, (CODE.handle,)) in platform.calls


async def test_closes_a_window_by_title_without_focusing_it(settings: Settings) -> None:
    platform = StubPlatform(windows=[CODE, NOTEPAD], active=CODE)

    result = await WindowActionTool().call(
        {"action": "close", "title": "notepad"}, local_context(settings, platform=platform)
    )

    assert result.ok
    assert platform.windows == [CODE]
    assert platform.active == CODE
    assert not any(name.startswith("focus") for name, _ in platform.calls)


async def test_reports_a_missing_window(settings: Settings) -> None:
    context = local_context(settings, platform=StubPlatform(windows=[CODE]))

    by_title = await WindowActionTool().call({"action": "minimize", "title": "Excel"}, context)
    unfocused = await WindowActionTool().call({"action": "minimize"}, context)

    assert by_title.spoken == "I can't find a window called Excel."
    assert unfocused.spoken == "I can't find a focused window."


async def test_reports_a_window_that_vanished(settings: Settings) -> None:
    platform = VanishingPlatform(windows=[CODE], active=CODE)

    result = await WindowActionTool().call(
        {"action": "minimize"}, local_context(settings, platform=platform)
    )

    assert not result.ok
    assert result.spoken == "That window has just closed."


async def test_rejects_an_unknown_action(settings: Settings) -> None:
    result = await WindowActionTool().call({"action": "explode"}, local_context(settings))

    assert not result.ok
    assert "action" in result.detail
