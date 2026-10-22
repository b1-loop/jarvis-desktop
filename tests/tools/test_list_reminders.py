from collections.abc import AsyncIterator, Callable
from datetime import UTC, datetime
from pathlib import Path

import pytest

from jarvis.agent.tool import ToolContext
from jarvis.config import Settings
from jarvis.memory import MemoryManager
from jarvis.tools.list_reminders import ListRemindersTool
from tests.tools.local_fakes import local_context, open_memory


@pytest.fixture
async def memory(tmp_path: Path) -> AsyncIterator[MemoryManager]:
    async with open_memory(tmp_path) as manager:
        yield manager


@pytest.fixture
def context(make_settings: Callable[..., Settings], memory: MemoryManager) -> ToolContext:
    return local_context(make_settings(), memory=memory)


async def test_lists_pending_reminders_in_local_time(
    context: ToolContext, memory: MemoryManager
) -> None:
    await memory.add_reminder("Call mum", datetime(2026, 10, 21, 15, 0, tzinfo=UTC))
    await memory.add_reminder("Stand-up", datetime(2026, 10, 22, 6, 30, tzinfo=UTC))

    result = await ListRemindersTool().call({}, context)

    assert result.spoken == "Call mum on Wed 21 Oct at 18:00; Stand-up on Thu 22 Oct at 09:30."
    assert result.data is not None
    reminders = result.data["reminders"]
    assert isinstance(reminders, list)
    assert reminders[0] == {
        "id": 1,
        "text": "Call mum",
        "due_at": "2026-10-21T18:00:00+03:00",
        "fired": False,
    }


async def test_many_reminders_are_summarised(context: ToolContext, memory: MemoryManager) -> None:
    for hour in range(5):
        await memory.add_reminder(f"R{hour}", datetime(2026, 10, 22, hour, tzinfo=UTC))

    result = await ListRemindersTool().call({}, context)

    assert result.spoken.endswith("; and 2 more.")


async def test_no_reminders(context: ToolContext) -> None:
    result = await ListRemindersTool().call({"include_fired": True}, context)

    assert result.spoken == "You have no reminders."
