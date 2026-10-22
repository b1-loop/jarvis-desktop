from collections.abc import AsyncIterator, Callable
from pathlib import Path

import pytest

from jarvis.config import Settings
from jarvis.memory import MemoryManager
from jarvis.tools.list_todos import ListTodosTool
from tests.tools.local_fakes import local_context, open_memory


@pytest.fixture
async def memory(tmp_path: Path) -> AsyncIterator[MemoryManager]:
    async with open_memory(tmp_path) as manager:
        yield manager


async def test_lists_open_items_and_optionally_done_ones(
    make_settings: Callable[..., Settings], memory: MemoryManager
) -> None:
    for text in ("Buy milk", "Renew passport", "Call mum"):
        await memory.add_todo(text)
    done = (await memory.list_todos())[0]
    await memory.complete_todo(done.id)
    context = local_context(make_settings(), memory=memory)

    open_only = await ListTodosTool().call({}, context)
    everything = await ListTodosTool().call({"include_done": True}, context)

    assert open_only.spoken == "You have 2 items: Renew passport; Call mum."
    assert f"[{done.id}] x Buy milk" in everything.detail
    assert everything.data is not None
    todos = everything.data["todos"]
    assert isinstance(todos, list)
    assert len(todos) == 3


async def test_long_lists_are_summarised(
    make_settings: Callable[..., Settings], memory: MemoryManager
) -> None:
    for number in range(7):
        await memory.add_todo(f"Task {number}")

    result = await ListTodosTool().call({}, local_context(make_settings(), memory=memory))

    assert result.spoken.endswith("Task 4; and 2 more.")


async def test_empty_and_single_lists(
    make_settings: Callable[..., Settings], memory: MemoryManager
) -> None:
    context = local_context(make_settings(), memory=memory)

    empty = await ListTodosTool().call({}, context)
    await memory.add_todo("Water plants")
    single = await ListTodosTool().call({}, context)

    assert empty.spoken == "Your todo list is empty."
    assert single.spoken == "You have one item: Water plants."
