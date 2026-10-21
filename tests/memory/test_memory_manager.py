from collections.abc import AsyncIterator
from datetime import timedelta
from pathlib import Path

import pytest

from jarvis.memory.memory_manager import (
    MemoryManager,
    RecalledContext,
    truncating_summariser,
)
from jarvis.memory.models import Fact, Message, Role
from jarvis.memory.sqlite_store import SqliteStore
from jarvis.memory.vector_store import InMemoryVectorStore, VectorHit
from tests.memory.fakes import EPOCH, FakeClock, RecordingSummariser


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
async def store(tmp_path: Path, clock: FakeClock) -> AsyncIterator[SqliteStore]:
    async with SqliteStore(tmp_path / "memory.db", clock=clock) as opened:
        yield opened


@pytest.fixture
def vectors() -> InMemoryVectorStore:
    return InMemoryVectorStore()


@pytest.fixture
def summariser() -> RecordingSummariser:
    return RecordingSummariser()


@pytest.fixture
def manager(
    store: SqliteStore, vectors: InMemoryVectorStore, summariser: RecordingSummariser
) -> MemoryManager:
    return MemoryManager(store, vectors, recent_turns=2, semantic_hits=2, summariser=summariser)


def message(content: str, role: Role = Role.USER, message_id: int = 1) -> Message:
    return Message(message_id, 1, role, content, (), EPOCH)


# -- conversations --------------------------------------------------------------


async def test_record_turn_starts_a_conversation_and_embeds_the_exchange(
    manager: MemoryManager, store: SqliteStore, vectors: InMemoryVectorStore
) -> None:
    assert manager.conversation is None

    user, assistant = await manager.record_turn("open vs code", "done", [{"name": "open_app"}])

    assert manager.conversation is not None
    assert user.role is Role.USER
    assert assistant.tool_calls == ({"name": "open_app"},)
    assert len(await store.recent_messages(10)) == 2
    assert await vectors.count() == 1
    hit = (await vectors.query("vs code"))[0]
    assert hit.id == f"turn:{assistant.id}"
    assert hit.metadata["conversation_id"] == manager.conversation.id


async def test_end_conversation_uses_the_summariser_hook(
    manager: MemoryManager, store: SqliteStore, summariser: RecordingSummariser
) -> None:
    conversation = await manager.start_conversation()
    await manager.record_turn("hello", "hi")

    summary = await manager.end_conversation()

    assert summary == "we talked"
    assert [m.content for m in summariser.calls[0]] == ["hello", "hi"]
    assert (await store.get_conversation(conversation.id)).summary == "we talked"
    assert manager.conversation is None
    assert await manager.end_conversation() == ""


async def test_starting_a_conversation_ends_the_previous_one(
    manager: MemoryManager, store: SqliteStore
) -> None:
    first = await manager.start_conversation()
    await manager.record_turn("a", "b")

    second = await manager.start_conversation()

    assert second.id != first.id
    assert (await store.get_conversation(first.id)).summary == "we talked"


async def test_an_empty_conversation_is_not_summarised(
    manager: MemoryManager, summariser: RecordingSummariser
) -> None:
    await manager.start_conversation()

    assert await manager.end_conversation() == ""
    assert summariser.calls == []


async def test_truncating_summariser_keeps_the_start() -> None:
    long = message("x" * 1000)

    summary = await truncating_summariser([long])

    assert summary.startswith("user: xxx")
    assert len(summary) == 300
    assert summary.endswith("…")
    assert await truncating_summariser([message("short")]) == "user: short"


# -- recall ---------------------------------------------------------------------


async def test_recall_combines_profile_facts_recent_and_semantic(
    manager: MemoryManager, store: SqliteStore
) -> None:
    await manager.remember("profile.name", "Bozhidar")
    await manager.remember("weather.city", "Sofia", source="llm", confidence=0.8)
    await manager.record_turn("my thesis is about graph neural networks", "noted")
    await manager.record_turn("what is the weather", "sunny")
    await manager.record_turn("open spotify", "playing")
    await manager.record_turn("louder", "volume up")

    context = await manager.recall("tell me about my thesis topic")

    assert context.profile == "User profile:\n- Name: Bozhidar"
    assert [f.key for f in context.facts] == ["weather.city"]
    assert [m.content for m in context.recent] == ["open spotify", "playing", "louder", "volume up"]
    assert len(context.semantic) >= 1
    assert "graph neural networks" in context.semantic[0].text
    rendered = context.render()
    assert rendered.index("User profile") < rendered.index("Known facts")
    assert rendered.index("Related memories") < rendered.index("Recent conversation")


async def test_recall_does_not_repeat_recent_turns_as_semantic_hits(
    manager: MemoryManager,
) -> None:
    await manager.record_turn("the jarvis python project", "ok")

    context = await manager.recall("jarvis python project")

    assert len(context.recent) == 2
    assert context.semantic == ()


async def test_recall_includes_notes_as_semantic_hits(manager: MemoryManager) -> None:
    await manager.add_note("renew the passport before the trip to japan", title="travel")

    context = await manager.recall("japan trip passport")

    assert len(context.semantic) == 1
    assert context.semantic[0].metadata["kind"] == "note"
    assert context.semantic[0].text.startswith("travel\n")


async def test_recall_with_semantic_hits_disabled(
    store: SqliteStore, vectors: InMemoryVectorStore
) -> None:
    manager = MemoryManager(store, vectors, semantic_hits=0)
    await manager.add_note("something to find")

    assert (await manager.recall("find")).semantic == ()


async def test_recall_on_an_empty_memory_renders_nothing(manager: MemoryManager) -> None:
    context = await manager.recall("first words ever")

    assert context == RecalledContext("", (), (), ())
    assert context.render() == ""


async def test_recall_respects_the_character_budget(
    store: SqliteStore, vectors: InMemoryVectorStore
) -> None:
    manager = MemoryManager(store, vectors, recent_turns=5, context_budget_chars=120)
    await manager.remember("profile.name", "Bozhidar")
    for index in range(5):
        await manager.record_turn(f"long user message number {index} " * 3, "reply " * 5)

    context = await manager.recall("anything")

    assert len(context.render()) <= 120
    assert context.profile  # the profile is the last thing to go
    assert len(context.recent) < 10


def test_fit_drops_in_order_semantic_facts_recent_profile() -> None:
    context = RecalledContext(
        profile="User profile:\n- Name: B",
        facts=(Fact("k", "v", "user", 1.0, EPOCH, EPOCH),),
        recent=(message("one", message_id=1), message("two", message_id=2)),
        semantic=(VectorHit("h", "hit", {}, 0.1),),
    )

    without_semantic = context.fit(len(context.render()) - 1)
    assert without_semantic.semantic == ()
    assert without_semantic.facts

    only_profile = context.fit(len(context.profile))
    assert only_profile == RecalledContext(context.profile, (), (), ())

    assert context.fit(0).render() == ""


# -- facts, notes, todos, reminders -------------------------------------------


async def test_remember_forget_and_profile(manager: MemoryManager) -> None:
    fact = await manager.remember("profile.name", "Bozhidar")
    assert fact.value == "Bozhidar"
    assert (await manager.profile()).name == "Bozhidar"

    assert await manager.forget("profile.name")
    assert not await manager.forget("profile.name")
    assert (await manager.profile()).is_empty


async def test_notes_are_searchable_and_deletable_from_both_stores(
    manager: MemoryManager, vectors: InMemoryVectorStore
) -> None:
    note = await manager.add_note("buy oat milk", title="groceries")

    assert [n.id for n in await manager.search_notes("oat milk")] == [note.id]
    assert await vectors.count() == 1

    await manager.delete_note(note.id)

    assert await manager.search_notes("oat") == []
    assert await vectors.count() == 0


async def test_todos_pass_through(manager: MemoryManager) -> None:
    todo = await manager.add_todo("ship memory")

    assert [t.id for t in await manager.list_todos()] == [todo.id]
    assert (await manager.complete_todo(todo.id)).done
    assert await manager.list_todos() == []
    assert len(await manager.list_todos(include_done=True)) == 1


async def test_reminders_pass_through(manager: MemoryManager, clock: FakeClock) -> None:
    due = await manager.add_reminder("stretch", EPOCH + timedelta(minutes=30))

    assert [r.id for r in await manager.list_reminders()] == [due.id]
    assert await manager.pop_due_reminders() == []
    clock.advance(minutes=30)
    assert [r.id for r in await manager.pop_due_reminders()] == [due.id]
    assert await manager.list_reminders() == []
    assert len(await manager.list_reminders(include_fired=True)) == 1
