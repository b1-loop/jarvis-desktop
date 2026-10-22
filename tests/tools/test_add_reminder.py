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
    assert "2026-10-21T15:42:07+03:00" in result.detail  # the model learns the real time
    assert "in_minutes" in result.detail  # and how to say "in five minutes" instead
    assert "year" not in result.detail
    assert await memory.list_reminders() == []


async def test_a_date_in_a_past_year_is_rejected_with_the_year_named(
    context: ToolContext, memory: MemoryManager
) -> None:
    # What qwen2.5:7b made of "påminn mig om fem minuter".
    result = await AddReminderTool().call(
        {"text": "Ring mamma", "due_at": "2023-10-24T13:47:00+02:00"}, context
    )

    assert not result.ok
    assert "The year 2023 is wrong." in result.detail
    assert await memory.list_reminders() == []


@pytest.mark.parametrize(
    ("minutes", "phrase"),
    [(5, "at 3:47 PM"), (120, "at 5:42 PM"), (9 * 60, "tomorrow at 12:42 AM")],
)
async def test_in_minutes_is_resolved_against_the_clock(
    context: ToolContext, memory: MemoryManager, minutes: int, phrase: str
) -> None:
    result = await AddReminderTool().call({"text": "Ring mamma", "in_minutes": minutes}, context)

    assert result.ok, result.detail
    assert result.spoken == f"I'll remind you to ring mamma {phrase}."
    (stored,) = await memory.list_reminders()
    assert stored.due_at == FakeClock().now() + timedelta(minutes=minutes)


@pytest.mark.parametrize(
    "arguments",
    [
        {"text": "x"},
        {"text": "x", "in_minutes": 5, "due_at": "2026-10-21T18:00:00+03:00"},
        {"text": "x", "in_minutes": 0},
        {"text": "x", "in_minutes": -5},
        {"text": "x", "in_minutes": 6 * 366 * 24 * 60},
    ],
)
async def test_exactly_one_plausible_time_is_required(
    context: ToolContext, memory: MemoryManager, arguments: dict[str, object]
) -> None:
    result = await AddReminderTool().call(arguments, context)

    assert not result.ok
    assert await memory.list_reminders() == []


def test_the_schema_offers_both_ways_and_requires_neither() -> None:
    schema = AddReminderTool.Input.model_json_schema()

    assert {"in_minutes", "due_at"} <= set(schema["properties"])
    assert schema["required"] == ["text"]


async def test_rejects_an_implausibly_distant_time(context: ToolContext) -> None:
    far = datetime(2026, 10, 21, tzinfo=UTC) + timedelta(days=6 * 366)

    result = await AddReminderTool().call({"text": "x", "due_at": far.isoformat()}, context)

    assert not result.ok
    assert "five years" in result.spoken


async def test_rejects_something_that_is_not_a_datetime(context: ToolContext) -> None:
    result = await AddReminderTool().call({"text": "x", "due_at": "at six"}, context)

    assert not result.ok
    assert "due_at" in result.detail
