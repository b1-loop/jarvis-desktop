from collections.abc import AsyncIterator, Callable
from pathlib import Path

import pytest

from jarvis.agent.tool import ToolContext
from jarvis.config import Settings
from jarvis.memory import MemoryManager
from jarvis.tools.complete_todo import CompleteTodoTool
from tests.tools.local_fakes import local_context, open_memory


@pytest.fixture
async def memory(tmp_path: Path) -> AsyncIterator[MemoryManager]:
    async with open_memory(tmp_path) as manager:
        for text in ("Buy milk", "Buy oat milk", "Renew passport"):
            await manager.add_todo(text)
        yield manager


@pytest.fixture
def context(make_settings: Callable[..., Settings], memory: MemoryManager) -> ToolContext:
    return local_context(make_settings(), memory=memory)


async def test_ticks_off_by_id(context: ToolContext, memory: MemoryManager) -> None:
    passport = (await memory.list_todos())[2]

    result = await CompleteTodoTool().call({"todo_id": passport.id}, context)

    assert result.spoken == "Ticked off 'Renew passport'."
    assert [todo.text for todo in await memory.list_todos()] == ["Buy milk", "Buy oat milk"]


async def test_ticks_off_by_a_unique_text_fragment(
    context: ToolContext, memory: MemoryManager
) -> None:
    result = await CompleteTodoTool().call({"text": "PASSPORT"}, context)

    assert result.ok
    assert len(await memory.list_todos()) == 2


async def test_an_ambiguous_fragment_asks_instead_of_guessing(
    context: ToolContext, memory: MemoryManager
) -> None:
    result = await CompleteTodoTool().call({"text": "milk"}, context)

    assert not result.ok
    assert result.spoken == "Several todos mention milk. Which one did you mean?"
    assert result.data is not None
    candidates = result.data["candidates"]
    assert isinstance(candidates, list)
    assert len(candidates) == 2
    assert len(await memory.list_todos()) == 3


async def test_unknown_id_or_text(context: ToolContext) -> None:
    by_id = await CompleteTodoTool().call({"todo_id": 999}, context)
    by_text = await CompleteTodoTool().call({"text": "dentist"}, context)

    assert by_id.spoken == "I can't find todo number 999."
    assert by_text.spoken == "I can't find an open todo about dentist."


@pytest.mark.parametrize("raw", [{}, {"todo_id": 1, "text": "milk"}, {"todo_id": 0}])
async def test_requires_exactly_one_valid_selector(
    context: ToolContext, raw: dict[str, object]
) -> None:
    result = await CompleteTodoTool().call(raw, context)

    assert not result.ok
