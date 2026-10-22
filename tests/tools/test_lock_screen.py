from collections.abc import Callable

from jarvis.agent.confirmation import DECLINED_SPOKEN, ConfirmationPolicy
from jarvis.agent.tool import ToolContext
from jarvis.config import Settings
from jarvis.core.events import EventBus
from jarvis.os_integration import StubPlatform
from jarvis.tools.lock_screen import LockScreenInput, LockScreenTool
from tests.agent.fakes import FakeConfirmation
from tests.tools.local_fakes import local_context


async def test_locks_the_computer(make_settings: Callable[..., Settings]) -> None:
    platform = StubPlatform()

    result = await LockScreenTool().call({}, local_context(make_settings(), platform=platform))

    assert result.ok
    assert result.spoken == "Locking now."
    assert platform.calls == [("lock_screen", ())]


async def test_policy_asks_first_and_respects_a_no(make_settings: Callable[..., Settings]) -> None:
    platform = StubPlatform()
    confirmation = FakeConfirmation(answers=[False])
    context = ToolContext(
        settings=make_settings(),
        bus=EventBus(),
        ask_confirmation=confirmation,
        platform=platform,
    )

    result = await ConfirmationPolicy().execute_with_policy(LockScreenTool(), {}, context)

    assert result.spoken == DECLINED_SPOKEN
    assert confirmation.prompts == ["Do you want me to lock the computer?"]
    assert platform.calls == []
    assert LockScreenTool().confirmation_prompt(LockScreenInput()) == confirmation.prompts[0]
