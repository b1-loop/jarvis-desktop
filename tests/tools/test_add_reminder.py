from collections.abc import AsyncIterator, Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from jarvis.agent.tool import ToolContext
from jarvis.config import Settings
from jarvis.memory import MemoryManager
from jarvis.tools.add_reminder import AddReminderTool
from tests.fakes import FakeClock
from tests.tools.local_fakes import local_context, open_memory

# FakeClock's "now" is 2026-10-21 15:42:07 at UTC+3.


@pytest.fixture
async def memory(tmp_path: Path) -> AsyncIterator[MemoryManager]:
    async with open_memory(tmp_path) as manager:
        yield manager


@pytest.fixture
def context(make_settings: Callable[..., Settings], memory: MemoryManager) -> ToolContext:
    return local_context(make_settings(), memory=memory, clock=FakeClock())


@pytest.mark.parametrize(
    ("due_at", "phrase"),
    [
        ("2026-10-21T18:00:00+03:00", "at 6:00 PM"),
        ("2026-10-21T18:00:00", "at 6:00 PM"),  # naive means local
        ("2026-10-21T13:30:00Z", "at 4:30 PM"),  # converted to local for speech
        ("2026-10-22T09:05:00+03:00", "tomorrow at 9:05 AM"),
        ("2026-10-30T12:00:00+03:00", "on Friday, 30 October at 12:00 PM"),
    ],
)
async def test_stores_future_reminders_and_phrases_the_time(
    context: ToolContext, memory: MemoryManager, due_at: str, phrase: str
) -> None:
    result = await AddReminderTool().call({"text": "Call mum", "due_at": due_at}, context)

    assert result.ok, result.detail
    assert result.spoken == f"I'll remind you to call mum {phrase}."
    (stored,) = await memory.list_reminders()
    assert stored.text == "Call mum"
    assert stored.due_at == datetime.fromisoformat(
        str(result.data["due_at"] if result.data else "")
    )


async def test_rejects_a_time_in_the_past(context: ToolContext, memory: MemoryManager) -> None:
    result = await AddReminderTool().call(
        {"text": "Call mum", "due_at": "2026-10-21T09:00:00+03:00"}, context
    )

    assert not result.ok
    assert result.spoken == "That time has already passed. When should I remind you?"
    assert await memory.list_reminders() == []


async def test_rejects_an_implausibly_distant_time(context: ToolContext) -> None:
    far = datetime(2026, 10, 21, tzinfo=UTC) + timedelta(days=6 * 366)

    result = await AddReminderTool().call({"text": "x", "due_at": far.isoformat()}, context)

    assert not result.ok
    assert "five years" in result.spoken


async def test_rejects_something_that_is_not_a_datetime(context: ToolContext) -> None:
    result = await AddReminderTool().call({"text": "x", "due_at": "at six"}, context)

    assert not result.ok
    assert "due_at" in result.detail
