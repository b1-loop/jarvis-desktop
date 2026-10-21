from pathlib import Path

import aiosqlite
import pytest

from jarvis.memory.errors import MemoryStoreError
from jarvis.memory.sqlite_store import MIGRATIONS, SqliteStore, apply_migrations


async def table_names(path: Path) -> set[str]:
    async with aiosqlite.connect(path) as conn:
        cursor = await conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        return {str(row[0]) for row in await cursor.fetchall()}


async def test_migrating_an_empty_database_creates_every_table(tmp_path: Path) -> None:
    path = tmp_path / "memory.db"

    async with SqliteStore(path, full_text_search=False) as store:
        assert store.schema_version == len(MIGRATIONS)

    assert {"conversations", "messages", "notes", "reminders", "todos", "facts"} <= (
        await table_names(path)
    )


async def test_an_older_database_is_upgraded_in_place(tmp_path: Path) -> None:
    path = tmp_path / "memory.db"
    async with aiosqlite.connect(path, isolation_level=None) as conn:
        assert await apply_migrations(conn, up_to=1) == 1
        await conn.execute("INSERT INTO todos (text, created_at) VALUES ('old', '2026-01-01')")
    assert "facts" not in await table_names(path)

    async with SqliteStore(path, full_text_search=False) as store:
        assert store.schema_version == len(MIGRATIONS)
        todos = await store.list_todos()
        await store.upsert_fact("after", "upgrade")

    assert [t.text for t in todos] == ["old"]
    assert "facts" in await table_names(path)


async def test_migrations_are_recorded_once_and_reopening_is_a_no_op(tmp_path: Path) -> None:
    path = tmp_path / "memory.db"
    async with SqliteStore(path, full_text_search=False):
        pass
    async with SqliteStore(path, full_text_search=False):
        pass

    async with aiosqlite.connect(path) as conn:
        cursor = await conn.execute("SELECT version FROM schema_version ORDER BY version")
        versions = [row[0] for row in await cursor.fetchall()]

    assert versions == list(range(1, len(MIGRATIONS) + 1))


async def test_a_database_from_the_future_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "memory.db"
    async with aiosqlite.connect(path) as conn:
        await conn.execute(
            "CREATE TABLE schema_version (version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL)"
        )
        await conn.execute("INSERT INTO schema_version VALUES (99, 'later')")
        await conn.commit()

    with pytest.raises(MemoryStoreError, match="newer than this build"):
        await SqliteStore(path).open()


async def test_a_failing_migration_rolls_back_and_leaves_the_version_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def broken(conn: aiosqlite.Connection) -> None:
        await conn.execute("CREATE TABLE half_done (x)")
        raise RuntimeError("disk on fire")

    monkeypatch.setattr("jarvis.memory.sqlite_store.MIGRATIONS", (MIGRATIONS[0], broken))
    path = tmp_path / "memory.db"

    with pytest.raises(RuntimeError, match="disk on fire"):
        await SqliteStore(path).open()

    assert "half_done" not in await table_names(path)
    async with aiosqlite.connect(path) as conn:
        cursor = await conn.execute("SELECT MAX(version) FROM schema_version")
        row = await cursor.fetchone()
    assert row is not None
    assert row[0] == 1


async def test_fts_index_is_built_for_rows_that_predate_it(tmp_path: Path) -> None:
    path = tmp_path / "memory.db"
    async with SqliteStore(path, full_text_search=False) as store:
        await store.add_note("written before the index existed")

    async with SqliteStore(path, full_text_search=True) as store:
        assert store.full_text_search
        assert len(await store.search_notes("index existed")) == 1
