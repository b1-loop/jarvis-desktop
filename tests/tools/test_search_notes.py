from collections.abc import AsyncIterator, Callable
from pathlib import Path

import pytest

from jarvis.agent.tool import ToolContext
from jarvis.config import Settings
from jarvis.memory import MemoryManager
from jarvis.tools.search_notes import SearchNotesTool
from tests.tools.local_fakes import local_context, open_memory


@pytest.fixture
async def memory(tmp_path: Path) -> AsyncIterator[MemoryManager]:
    async with open_memory(tmp_path) as manager:
        await manager.add_note("Parking code is 4711", "Parking")
        await manager.add_note("Parking at the station costs 3 euros")
        await manager.add_note("Buy a birthday present")
        yield manager


@pytest.fixture
def context(make_settings: Callable[..., Settings], memory: MemoryManager) -> ToolContext:
    return local_context(make_settings(), memory=memory)


async def test_returns_matching_notes_and_speaks_the_best(context: ToolContext) -> None:
    result = await SearchNotesTool().call({"query": "parking"}, context)

    assert result.ok
    assert result.spoken.startswith("Your note says: ")
    assert result.spoken.endswith("I found 2 notes in total.")
    assert result.data is not None
    notes = result.data["notes"]
    assert isinstance(notes, list)
    assert len(notes) == 2
    assert "Parking: Parking code is 4711" in result.detail


async def test_a_single_hit_is_read_out_alone(context: ToolContext) -> None:
    result = await SearchNotesTool().call({"query": "birthday"}, context)

    assert result.spoken == "Your note says: Buy a birthday present."


async def test_no_hits(context: ToolContext) -> None:
    result = await SearchNotesTool().call({"query": "dentist"}, context)

    assert result.ok
    assert result.spoken == "I don't have any notes about dentist."
