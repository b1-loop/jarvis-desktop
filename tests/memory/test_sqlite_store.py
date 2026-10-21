from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import pytest

from jarvis.memory.errors import MemoryStoreError, NotFoundError, StoreClosedError
from jarvis.memory.models import Role
from jarvis.memory.sqlite_store import MIGRATIONS, SqliteStore
from tests.memory.fakes import EPOCH, FakeClock


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture(params=["fts5", "like"])
async def store(
    request: pytest.FixtureRequest, tmp_path: Path, clock: FakeClock
) -> AsyncIterator[SqliteStore]:
    """Every test runs against both search back-ends."""
    use_fts = request.param == "fts5"
    async with SqliteStore(tmp_path / "memory.db", full_text_search=use_fts, clock=clock) as opened:
        yield opened


async def test_open_creates_parent_directories_and_migrates(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "deeper" / "memory.db"

    async with SqliteStore(path) as store:
        assert store.schema_version == len(MIGRATIONS)

    assert path.exists()


async def test_open_and_close_are_idempotent(tmp_path: Path) -> None:
    store = SqliteStore(tmp_path / "memory.db")
    await store.open()
    await store.open()
    await store.close()
    await store.close()


async def test_closed_store_refuses_queries(tmp_path: Path) -> None:
    store = SqliteStore(tmp_path / "memory.db")

    with pytest.raises(StoreClosedError):
        await store.list_todos()


async def test_requiring_fts5_fails_clearly_when_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def unavailable(_conn: object) -> bool:
        return False

    monkeypatch.setattr("jarvis.memory.sqlite_store.fts5_available", unavailable)

    with pytest.raises(MemoryStoreError, match="FTS5"):
        await SqliteStore(tmp_path / "memory.db", full_text_search=True).open()


async def test_autodetect_falls_back_to_like_without_fts5(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def unavailable(_conn: object) -> bool:
        return False

    monkeypatch.setattr("jarvis.memory.sqlite_store.fts5_available", unavailable)

    async with SqliteStore(tmp_path / "memory.db") as store:
        assert not store.full_text_search
        await store.add_note("fallback search still works")
        assert len(await store.search_notes("fallback")) == 1


async def test_wal_mode_is_enabled(tmp_path: Path) -> None:
    async with SqliteStore(tmp_path / "memory.db") as store:
        row = await store._fetch_one("PRAGMA journal_mode")

    assert row is not None
    assert row[0] == "wal"


# -- conversations and messages ---------------------------------------------


async def test_messages_round_trip_with_tool_calls(store: SqliteStore, clock: FakeClock) -> None:
    conversation = await store.start_conversation()
    calls = ({"name": "open_app", "input": {"app": "code"}},)

    await store.add_message(conversation.id, Role.USER, "open vs code")
    clock.advance(seconds=1)
    reply = await store.add_message(conversation.id, Role.ASSISTANT, "done", calls)

    messages = await store.recent_messages(10)
    assert [m.role for m in messages] == [Role.USER, Role.ASSISTANT]
    assert messages[1].tool_calls == calls
    assert messages[1].id == reply.id
    assert messages[1].created_at == EPOCH + timedelta(seconds=1)
    assert conversation.started_at == EPOCH


async def test_recent_messages_keeps_the_newest_in_chronological_order(
    store: SqliteStore,
) -> None:
    conversation = await store.start_conversation()
    for index in range(5):
        await store.add_message(conversation.id, Role.USER, f"message {index}")

    messages = await store.recent_messages(2)

    assert [m.content for m in messages] == ["message 3", "message 4"]


async def test_recent_messages_can_be_scoped_to_one_conversation(store: SqliteStore) -> None:
    first = await store.start_conversation()
    second = await store.start_conversation()
    await store.add_message(first.id, Role.USER, "in first")
    await store.add_message(second.id, Role.USER, "in second")

    assert [m.content for m in await store.recent_messages(10, conversation_id=first.id)] == [
        "in first"
    ]
    assert len(await store.recent_messages(10)) == 2


async def test_summary_is_stored_and_conversations_listed_newest_first(
    store: SqliteStore,
) -> None:
    first = await store.start_conversation()
    second = await store.start_conversation()

    await store.set_summary(first.id, "talked about the weather")

    assert (await store.get_conversation(first.id)).summary == "talked about the weather"
    assert [c.id for c in await store.recent_conversations(10)] == [second.id, first.id]


async def test_missing_conversation_raises_not_found(store: SqliteStore) -> None:
    with pytest.raises(NotFoundError, match="conversations"):
        await store.get_conversation(999)
    with pytest.raises(NotFoundError):
        await store.set_summary(999, "nothing")


async def test_search_messages_requires_every_word_and_ignores_punctuation(
    store: SqliteStore,
) -> None:
    conversation = await store.start_conversation()
    await store.add_message(conversation.id, Role.USER, "What is the capital of France?")
    await store.add_message(conversation.id, Role.ASSISTANT, "Paris is the capital of France.")
    await store.add_message(conversation.id, Role.USER, "And of Spain?")

    hits = await store.search_messages('capital "France" -the')

    assert len(hits) == 2
    assert all("France" in hit.content for hit in hits)
    assert await store.search_messages("capital spain") == []
    assert await store.search_messages("   ?!  ") == []


async def test_search_is_case_insensitive_and_literal_for_underscores(
    store: SqliteStore,
) -> None:
    conversation = await store.start_conversation()
    await store.add_message(conversation.id, Role.USER, "Run build_native please")
    await store.add_message(conversation.id, Role.USER, "Run buildXnative please")

    hits = await store.search_messages("BUILD_NATIVE")

    assert [hit.content for hit in hits] == ["Run build_native please"]


# -- notes ----------------------------------------------------------------------


async def test_notes_are_listed_newest_first_and_searchable_by_title(store: SqliteStore) -> None:
    await store.add_note("buy milk and eggs", title="groceries")
    await store.add_note("call the dentist", title="health")

    assert [n.title for n in await store.list_notes()] == ["health", "groceries"]
    assert [n.title for n in await store.search_notes("groceries")] == ["groceries"]
    assert [n.title for n in await store.search_notes("dentist")] == ["health"]
    assert await store.search_notes("") == []


async def test_deleted_notes_disappear_from_search(store: SqliteStore) -> None:
    note = await store.add_note("temporary thought")

    await store.delete_note(note.id)

    assert await store.search_notes("temporary") == []
    with pytest.raises(NotFoundError):
        await store.delete_note(note.id)


# -- reminders ------------------------------------------------------------------


async def test_reminders_are_popped_once_in_due_order(store: SqliteStore, clock: FakeClock) -> None:
    later = await store.add_reminder("standup", EPOCH + timedelta(minutes=10))
    soon = await store.add_reminder("coffee", EPOCH + timedelta(minutes=5))
    await store.add_reminder("tomorrow", EPOCH + timedelta(days=1))

    assert await store.pop_due_reminders() == []
    clock.advance(minutes=10)
    popped = await store.pop_due_reminders()

    assert [r.id for r in popped] == [soon.id, later.id]
    assert all(r.fired for r in popped)
    assert await store.pop_due_reminders() == []
    assert [r.text for r in await store.list_reminders()] == ["tomorrow"]
    assert len(await store.list_reminders(include_fired=True)) == 3


async def test_reminder_due_times_are_normalised_to_utc(store: SqliteStore) -> None:
    plus_two = timezone(timedelta(hours=2))
    naive = datetime(2026, 1, 1, 16, 0)  # naive on purpose: the store takes it as UTC

    aware = await store.add_reminder("aware", datetime(2026, 1, 1, 16, 0, tzinfo=plus_two))
    assumed = await store.add_reminder("naive", naive)

    assert aware.due_at == datetime(2026, 1, 1, 14, 0, tzinfo=UTC)
    assert assumed.due_at == datetime(2026, 1, 1, 16, 0, tzinfo=UTC)
    popped = await store.pop_due_reminders(datetime(2026, 1, 1, 15, 0, tzinfo=UTC))
    assert [r.text for r in popped] == ["aware"]


async def test_delete_reminder(store: SqliteStore) -> None:
    reminder = await store.add_reminder("gone", EPOCH)

    await store.delete_reminder(reminder.id)

    assert await store.list_reminders() == []
    with pytest.raises(NotFoundError):
        await store.delete_reminder(reminder.id)


# -- todos ----------------------------------------------------------------------


async def test_todos_can_be_completed(store: SqliteStore) -> None:
    first = await store.add_todo("write tests")
    second = await store.add_todo("open pull request")

    done = await store.complete_todo(first.id)

    assert done.done
    assert [t.id for t in await store.list_todos()] == [second.id]
    assert [t.id for t in await store.list_todos(include_done=True)] == [first.id, second.id]


async def test_completing_a_missing_todo_raises(store: SqliteStore) -> None:
    with pytest.raises(NotFoundError, match="todos"):
        await store.complete_todo(42)


# -- facts ----------------------------------------------------------------------


async def test_facts_upsert_by_key(store: SqliteStore, clock: FakeClock) -> None:
    first = await store.upsert_fact("profile.name", "Bozhidar")
    clock.advance(hours=1)
    second = await store.upsert_fact("profile.name", "Bobby", source="llm", confidence=0.6)

    assert second.created_at == first.created_at
    assert second.updated_at == first.updated_at + timedelta(hours=1)
    assert second.value == "Bobby"
    assert second.source == "llm"
    assert second.confidence == 0.6
    assert len(await store.list_facts()) == 1


async def test_facts_are_listed_by_prefix_literally(store: SqliteStore) -> None:
    await store.upsert_fact("profile.name", "B")
    await store.upsert_fact("profile_name", "underscore is not a wildcard")
    await store.upsert_fact("project.jarvis", "voice assistant")

    assert [f.key for f in await store.list_facts("profile.")] == ["profile.name"]
    assert [f.key for f in await store.list_facts()] == [
        "profile.name",
        "profile_name",
        "project.jarvis",
    ]


async def test_forgetting_facts(store: SqliteStore) -> None:
    await store.upsert_fact("temp", "x")

    assert await store.delete_fact("temp")
    assert not await store.delete_fact("temp")
    assert await store.get_fact("temp") is None


async def test_driver_errors_become_memory_store_errors(store: SqliteStore) -> None:
    with pytest.raises(MemoryStoreError, match="query failed"):
        await store._execute("SELECT * FROM no_such_table")
