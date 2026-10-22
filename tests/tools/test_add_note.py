from collections.abc import AsyncIterator, Callable
from pathlib import Path

import pytest

from jarvis.config import Settings
from jarvis.memory import MemoryManager
from jarvis.tools.add_note import AddNoteTool
from tests.tools.local_fakes import local_context, open_memory


@pytest.fixture
async def memory(tmp_path: Path) -> AsyncIterator[MemoryManager]:
    async with open_memory(tmp_path) as manager:
        yield manager


async def test_saves_the_note_with_an_optional_title(
    make_settings: Callable[..., Settings], memory: MemoryManager
) -> None:
    context = local_context(make_settings(), memory=memory)

    plain = await AddNoteTool().call({"content": "Parking code is 4711"}, context)
    titled = await AddNoteTool().call({"content": "Call at 9", "title": "Plumber"}, context)

    assert plain.spoken == "Noted."
    assert titled.detail.endswith("'Plumber'")
    found = await memory.search_notes("parking")
    assert [note.content for note in found] == ["Parking code is 4711"]


async def test_fails_clearly_without_memory(make_settings: Callable[..., Settings]) -> None:
    result = await AddNoteTool().call({"content": "x"}, local_context(make_settings()))

    assert not result.ok
    assert "memory store is not available" in result.detail
