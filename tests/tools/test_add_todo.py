from collections.abc import AsyncIterator, Callable
from pathlib import Path

import pytest

from jarvis.config import Settings
from jarvis.memory import MemoryManager
from jarvis.tools.add_todo import AddTodoTool
from tests.tools.local_fakes import local_context, open_memory


@pytest.fixture
async def memory(tmp_path: Path) -> AsyncIterator[MemoryManager]:
    async with open_memory(tmp_path) as manager:
        yield manager


async def test_adds_an_open_todo_and_echoes_it(
    make_settings: Callable[..., Settings], memory: MemoryManager
) -> None:
    result = await AddTodoTool().call(
        {"text": "Buy milk"}, local_context(make_settings(), memory=memory)
    )

    assert result.spoken == "Added 'Buy milk' to your list."
    assert [(todo.text, todo.done) for todo in await memory.list_todos()] == [("Buy milk", False)]


async def test_rejects_empty_text(make_settings: Callable[..., Settings]) -> None:
    result = await AddTodoTool().call({"text": ""}, local_context(make_settings()))

    assert not result.ok
