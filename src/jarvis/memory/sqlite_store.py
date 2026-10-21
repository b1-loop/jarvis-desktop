"""Relational memory: conversations, notes, reminders, todos and facts in SQLite.

One file under ``Settings.data_dir`` holds everything the assistant knows
that has a shape: who said what and when, what is due, what is done, what
is true. SQLite is used through ``aiosqlite`` so no query ever blocks the
event loop, in WAL mode so the proactive scheduler can read reminders while
the agent writes a turn.

The schema is versioned: ``MIGRATIONS`` is an ordered tuple of functions and
``schema_version`` records which have run, so a database created by an
older build is upgraded in place on open. Full-text search over messages
and notes uses FTS5 when the SQLite build has it and falls back to ``LIKE``
otherwise; see :func:`SqliteStore.full_text_search`.
"""

from __future__ import annotations

import asyncio
import json
import re
import sqlite3
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import aiosqlite

from jarvis.memory.errors import MemoryStoreError, NotFoundError, StoreClosedError
from jarvis.memory.models import (
    Conversation,
    Fact,
    JsonObject,
    Message,
    Note,
    Reminder,
    Role,
    Todo,
    utc_now,
)
from jarvis.utils.logging import get_logger

log = get_logger(__name__)

Migration = Callable[[aiosqlite.Connection], Awaitable[None]]
"""One schema step. Runs inside a transaction; must be idempotent-safe to retry."""

Params = tuple[object, ...]
Row = aiosqlite.Row

_WORD: Final = re.compile(r"\w+")
_LIKE_ESCAPE: Final = "\\"


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_V1_HISTORY: Final[tuple[str, ...]] = (
    """
    CREATE TABLE conversations (
        id         INTEGER PRIMARY KEY,
        started_at TEXT NOT NULL,
        summary    TEXT NOT NULL DEFAULT ''
    )
    """,
    """
    CREATE TABLE messages (
        id              INTEGER PRIMARY KEY,
        conversation_id INTEGER NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
        role            TEXT NOT NULL,
        content         TEXT NOT NULL,
        tool_calls      TEXT NOT NULL DEFAULT '[]',
        created_at      TEXT NOT NULL
    )
    """,
    "CREATE INDEX messages_by_conversation ON messages(conversation_id, id)",
    """
    CREATE TABLE notes (
        id         INTEGER PRIMARY KEY,
        title      TEXT NOT NULL DEFAULT '',
        content    TEXT NOT NULL,
        created_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE reminders (
        id         INTEGER PRIMARY KEY,
        text       TEXT NOT NULL,
        due_at     TEXT NOT NULL,
        fired      INTEGER NOT NULL DEFAULT 0,
        created_at TEXT NOT NULL
    )
    """,
    "CREATE INDEX reminders_due ON reminders(fired, due_at)",
    """
    CREATE TABLE todos (
        id         INTEGER PRIMARY KEY,
        text       TEXT NOT NULL,
        done       INTEGER NOT NULL DEFAULT 0,
        created_at TEXT NOT NULL
    )
    """,
)

_V2_FACTS: Final[tuple[str, ...]] = (
    """
    CREATE TABLE facts (
        key        TEXT PRIMARY KEY,
        value      TEXT NOT NULL,
        source     TEXT NOT NULL,
        confidence REAL NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    """,
)


async def _run_all(conn: aiosqlite.Connection, statements: Iterable[str]) -> None:
    for statement in statements:
        await conn.execute(statement)


async def _migration_1_history(conn: aiosqlite.Connection) -> None:
    """Conversation history, notes, reminders and todos."""
    await _run_all(conn, _V1_HISTORY)


async def _migration_2_facts(conn: aiosqlite.Connection) -> None:
    """Key/value facts, the backing store of the user profile."""
    await _run_all(conn, _V2_FACTS)


MIGRATIONS: Final[tuple[Migration, ...]] = (_migration_1_history, _migration_2_facts)
"""Schema steps in order; index + 1 is the version they produce. Append only."""


async def apply_migrations(conn: aiosqlite.Connection, *, up_to: int | None = None) -> int:
    """Bring ``conn`` to schema version ``up_to`` (latest by default).

    Each pending step runs in its own transaction and is recorded in
    ``schema_version`` only when it succeeds, so a crash mid-migration leaves
    the database at the previous version rather than half-way.

    Args:
        conn: An open connection in autocommit mode.
        up_to: Stop after this version. Tests use it to build "old" databases.

    Returns:
        The schema version after migrating.

    Raises:
        MemoryStoreError: If the database is newer than this build knows.
    """
    target = len(MIGRATIONS) if up_to is None else up_to
    await conn.execute(
        "CREATE TABLE IF NOT EXISTS schema_version "
        "(version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL)"
    )
    current = await _current_version(conn)
    if current > len(MIGRATIONS):
        raise MemoryStoreError(
            f"database schema version {current} is newer than this build ({len(MIGRATIONS)})"
        )
    for version in range(current + 1, target + 1):
        await conn.execute("BEGIN IMMEDIATE")
        try:
            await MIGRATIONS[version - 1](conn)
            await conn.execute(
                "INSERT INTO schema_version (version, applied_at) VALUES (?, ?)",
                (version, _to_db(utc_now())),
            )
            await conn.execute("COMMIT")
        except BaseException:
            await conn.execute("ROLLBACK")
            raise
        log.info("memory.sqlite.migrated", version=version)
    return max(current, target)


async def _current_version(conn: aiosqlite.Connection) -> int:
    cursor = await conn.execute("SELECT COALESCE(MAX(version), 0) FROM schema_version")
    row = await cursor.fetchone()
    return int(row[0]) if row is not None else 0


# ---------------------------------------------------------------------------
# Full-text search (optional)
# ---------------------------------------------------------------------------

_FTS_SETUP: Final[tuple[str, ...]] = (
    """
    CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts
        USING fts5(content, content='messages', content_rowid='id')
    """,
    """
    CREATE TRIGGER IF NOT EXISTS messages_fts_insert AFTER INSERT ON messages BEGIN
        INSERT INTO messages_fts(rowid, content) VALUES (new.id, new.content);
    END
    """,
    """
    CREATE TRIGGER IF NOT EXISTS messages_fts_delete AFTER DELETE ON messages BEGIN
        INSERT INTO messages_fts(messages_fts, rowid, content)
            VALUES ('delete', old.id, old.content);
    END
    """,
    """
    CREATE VIRTUAL TABLE IF NOT EXISTS notes_fts
        USING fts5(title, content, content='notes', content_rowid='id')
    """,
    """
    CREATE TRIGGER IF NOT EXISTS notes_fts_insert AFTER INSERT ON notes BEGIN
        INSERT INTO notes_fts(rowid, title, content) VALUES (new.id, new.title, new.content);
    END
    """,
    """
    CREATE TRIGGER IF NOT EXISTS notes_fts_delete AFTER DELETE ON notes BEGIN
        INSERT INTO notes_fts(notes_fts, rowid, title, content)
            VALUES ('delete', old.id, old.title, old.content);
    END
    """,
)


async def fts5_available(conn: aiosqlite.Connection) -> bool:
    """Probe the SQLite build for the FTS5 module by creating a temp table."""
    try:
        await conn.execute("CREATE VIRTUAL TABLE temp.fts5_probe USING fts5(x)")
    except sqlite3.OperationalError as error:
        log.info("memory.sqlite.fts5_unavailable", reason=str(error))
        return False
    await conn.execute("DROP TABLE temp.fts5_probe")
    return True


async def _enable_fts(conn: aiosqlite.Connection) -> None:
    """Create the FTS5 shadow tables and keep them in sync via triggers.

    The index is rebuilt from the content tables the first time it is
    created, so a database that pre-dates FTS support (or was written by a
    build without it) becomes searchable on the next open.
    """
    cursor = await conn.execute(
        "SELECT COUNT(*) FROM sqlite_master WHERE type = 'table' AND name = 'messages_fts'"
    )
    row = await cursor.fetchone()
    first_time = row is not None and int(row[0]) == 0
    await _run_all(conn, _FTS_SETUP)
    if first_time:
        await conn.execute("INSERT INTO messages_fts(messages_fts) VALUES ('rebuild')")
        await conn.execute("INSERT INTO notes_fts(notes_fts) VALUES ('rebuild')")


def _match_expression(query: str) -> str:
    """Turn free text into a safe FTS5 query: every word quoted, all required."""
    return " ".join(f'"{word}"' for word in _WORD.findall(query))


def _escape_like(text: str) -> str:
    """Escape ``LIKE`` wildcards so user text matches literally."""
    return (
        text.replace(_LIKE_ESCAPE, _LIKE_ESCAPE * 2)
        .replace("%", _LIKE_ESCAPE + "%")
        .replace("_", _LIKE_ESCAPE + "_")
    )


def _like_clauses(column_expr: str, query: str) -> tuple[str, Params]:
    """Build an AND of ``LIKE`` tests, one per word, for builds without FTS5."""
    words = _WORD.findall(query.lower())
    clause = " AND ".join(f"{column_expr} LIKE ? ESCAPE '{_LIKE_ESCAPE}'" for _ in words)
    params = tuple(f"%{_escape_like(word)}%" for word in words)
    return clause, params


# ---------------------------------------------------------------------------
# Row mapping
# ---------------------------------------------------------------------------


def _to_db(moment: datetime) -> str:
    """ISO-8601 in UTC with fixed precision so string order is time order."""
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    return moment.astimezone(UTC).isoformat(timespec="microseconds")


def _from_db(text: str) -> datetime:
    return datetime.fromisoformat(text)


def _conversation(row: Row) -> Conversation:
    return Conversation(
        id=row["id"], started_at=_from_db(row["started_at"]), summary=row["summary"]
    )


def _message(row: Row) -> Message:
    return Message(
        id=row["id"],
        conversation_id=row["conversation_id"],
        role=Role(row["role"]),
        content=row["content"],
        tool_calls=tuple(json.loads(row["tool_calls"])),
        created_at=_from_db(row["created_at"]),
    )


def _note(row: Row) -> Note:
    return Note(
        id=row["id"],
        title=row["title"],
        content=row["content"],
        created_at=_from_db(row["created_at"]),
    )


def _reminder(row: Row) -> Reminder:
    return Reminder(
        id=row["id"],
        text=row["text"],
        due_at=_from_db(row["due_at"]),
        fired=bool(row["fired"]),
        created_at=_from_db(row["created_at"]),
    )


def _todo(row: Row) -> Todo:
    return Todo(
        id=row["id"],
        text=row["text"],
        done=bool(row["done"]),
        created_at=_from_db(row["created_at"]),
    )


def _fact(row: Row) -> Fact:
    return Fact(
        key=row["key"],
        value=row["value"],
        source=row["source"],
        confidence=row["confidence"],
        created_at=_from_db(row["created_at"]),
        updated_at=_from_db(row["updated_at"]),
    )


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


class SqliteStore:
    """Async facade over one SQLite database file.

    Use as an async context manager or call ``open``/``close`` explicitly.
    Every public method raises :class:`MemoryStoreError` (or a subclass) on
    failure; driver exceptions never escape.

    Args:
        path: Database file. Parent directories are created on open.
        full_text_search: ``None`` detects FTS5 and uses it when present,
            ``True`` requires it, ``False`` forces the ``LIKE`` fallback.
        clock: Source of "now", injectable for deterministic tests.
    """

    def __init__(
        self,
        path: Path,
        *,
        full_text_search: bool | None = None,
        clock: Callable[[], datetime] = utc_now,
    ) -> None:
        self._path = path
        self._fts_requested = full_text_search
        self._clock = clock
        self._conn: aiosqlite.Connection | None = None
        self._fts = False
        self._version = 0

    async def __aenter__(self) -> SqliteStore:
        await self.open()
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.close()

    @property
    def schema_version(self) -> int:
        """Version the database was at after the last ``open``."""
        return self._version

    @property
    def full_text_search(self) -> bool:
        """True when searches use FTS5; False when they use ``LIKE``."""
        return self._fts

    async def open(self) -> None:
        """Connect, switch on WAL, migrate and set up search. Idempotent."""
        if self._conn is not None:
            return
        await asyncio.to_thread(self._path.parent.mkdir, parents=True, exist_ok=True)
        try:
            conn = await aiosqlite.connect(self._path, isolation_level=None)
        except sqlite3.Error as error:
            raise MemoryStoreError(f"cannot open memory database {self._path}: {error}") from error
        try:
            await self._prepare(conn)
        except BaseException:
            # aiosqlite runs each connection on a non-daemon thread; leaking
            # it on a failed open would keep the whole process alive at exit.
            await conn.close()
            raise
        self._conn = conn
        log.info(
            "memory.sqlite.opened",
            path=str(self._path),
            schema_version=self._version,
            fts5=self._fts,
        )

    async def _prepare(self, conn: aiosqlite.Connection) -> None:
        """Pragmas, migrations and search setup on a freshly opened connection."""
        try:
            conn.row_factory = aiosqlite.Row
            await conn.execute("PRAGMA journal_mode=WAL")
            await conn.execute("PRAGMA foreign_keys=ON")
            self._version = await apply_migrations(conn)
            self._fts = await self._configure_search(conn)
        except sqlite3.Error as error:
            raise MemoryStoreError(
                f"cannot prepare memory database {self._path}: {error}"
            ) from error

    async def _configure_search(self, conn: aiosqlite.Connection) -> bool:
        if self._fts_requested is False:
            return False
        available = await fts5_available(conn)
        if not available:
            if self._fts_requested is True:
                raise MemoryStoreError("FTS5 was required but this SQLite build lacks it")
            return False
        await _enable_fts(conn)
        return True

    async def close(self) -> None:
        """Close the connection. Idempotent."""
        if self._conn is None:
            return
        await self._conn.close()
        self._conn = None
        log.info("memory.sqlite.closed", path=str(self._path))

    # -- low-level helpers --------------------------------------------------

    def _connection(self) -> aiosqlite.Connection:
        if self._conn is None:
            raise StoreClosedError(f"SqliteStore({self._path}) is not open")
        return self._conn

    async def _execute(self, sql: str, params: Params = ()) -> aiosqlite.Cursor:
        try:
            return await self._connection().execute(sql, params)
        except sqlite3.Error as error:
            raise MemoryStoreError(f"query failed: {error}") from error

    async def _fetch_all(self, sql: str, params: Params = ()) -> list[Row]:
        cursor = await self._execute(sql, params)
        return list(await cursor.fetchall())

    async def _fetch_one(self, sql: str, params: Params = ()) -> Row | None:
        cursor = await self._execute(sql, params)
        return await cursor.fetchone()

    async def _insert(self, sql: str, params: Params) -> int:
        cursor = await self._execute(sql, params)
        if cursor.lastrowid is None:
            raise MemoryStoreError("insert returned no row id")
        return cursor.lastrowid

    @asynccontextmanager
    async def _transaction(self) -> AsyncIterator[None]:
        await self._execute("BEGIN IMMEDIATE")
        try:
            yield
        except BaseException:
            await self._execute("ROLLBACK")
            raise
        await self._execute("COMMIT")

    def _now(self) -> str:
        return _to_db(self._clock())

    # -- conversations and messages ------------------------------------------

    async def start_conversation(self) -> Conversation:
        """Create a new conversation starting now."""
        started = self._now()
        row_id = await self._insert("INSERT INTO conversations (started_at) VALUES (?)", (started,))
        return Conversation(id=row_id, started_at=_from_db(started))

    async def get_conversation(self, conversation_id: int) -> Conversation:
        """Fetch one conversation.

        Raises:
            NotFoundError: If no such conversation exists.
        """
        row = await self._fetch_one("SELECT * FROM conversations WHERE id = ?", (conversation_id,))
        if row is None:
            raise NotFoundError("conversations", conversation_id)
        return _conversation(row)

    async def set_summary(self, conversation_id: int, summary: str) -> None:
        """Record the "what was discussed" text for a finished conversation."""
        cursor = await self._execute(
            "UPDATE conversations SET summary = ? WHERE id = ?", (summary, conversation_id)
        )
        if cursor.rowcount == 0:
            raise NotFoundError("conversations", conversation_id)

    async def recent_conversations(self, limit: int) -> list[Conversation]:
        """The most recent conversations, newest first."""
        rows = await self._fetch_all(
            "SELECT * FROM conversations ORDER BY id DESC LIMIT ?", (limit,)
        )
        return [_conversation(row) for row in rows]

    async def add_message(
        self,
        conversation_id: int,
        role: Role,
        content: str,
        tool_calls: Iterable[JsonObject] = (),
    ) -> Message:
        """Append a message to a conversation."""
        created = self._now()
        calls = tuple(tool_calls)
        row_id = await self._insert(
            "INSERT INTO messages (conversation_id, role, content, tool_calls, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (conversation_id, role.value, content, json.dumps(list(calls)), created),
        )
        return Message(
            id=row_id,
            conversation_id=conversation_id,
            role=role,
            content=content,
            tool_calls=calls,
            created_at=_from_db(created),
        )

    async def recent_messages(
        self, limit: int, conversation_id: int | None = None
    ) -> list[Message]:
        """The last ``limit`` messages in chronological order.

        Args:
            limit: Maximum number of messages.
            conversation_id: Restrict to one conversation; ``None`` spans all.
        """
        if conversation_id is None:
            rows = await self._fetch_all(
                "SELECT * FROM messages ORDER BY id DESC LIMIT ?", (limit,)
            )
        else:
            rows = await self._fetch_all(
                "SELECT * FROM messages WHERE conversation_id = ? ORDER BY id DESC LIMIT ?",
                (conversation_id, limit),
            )
        return [_message(row) for row in reversed(rows)]

    async def search_messages(self, query: str, limit: int = 10) -> list[Message]:
        """Messages containing every word of ``query``, best match first."""
        if not _WORD.search(query):
            return []
        if self._fts:
            rows = await self._fetch_all(
                "SELECT m.* FROM messages_fts f JOIN messages m ON m.id = f.rowid "
                "WHERE messages_fts MATCH ? ORDER BY f.rank LIMIT ?",
                (_match_expression(query), limit),
            )
        else:
            clause, params = _like_clauses("lower(content)", query)
            rows = await self._fetch_all(
                f"SELECT * FROM messages WHERE {clause} ORDER BY id DESC LIMIT ?",  # noqa: S608
                (*params, limit),
            )
        return [_message(row) for row in rows]

    # -- notes ------------------------------------------------------------------

    async def add_note(self, content: str, title: str = "") -> Note:
        """Store a note."""
        created = self._now()
        row_id = await self._insert(
            "INSERT INTO notes (title, content, created_at) VALUES (?, ?, ?)",
            (title, content, created),
        )
        return Note(id=row_id, title=title, content=content, created_at=_from_db(created))

    async def list_notes(self, limit: int = 50) -> list[Note]:
        """Notes, newest first."""
        rows = await self._fetch_all("SELECT * FROM notes ORDER BY id DESC LIMIT ?", (limit,))
        return [_note(row) for row in rows]

    async def search_notes(self, query: str, limit: int = 10) -> list[Note]:
        """Notes whose title or body contains every word of ``query``."""
        if not _WORD.search(query):
            return []
        if self._fts:
            rows = await self._fetch_all(
                "SELECT n.* FROM notes_fts f JOIN notes n ON n.id = f.rowid "
                "WHERE notes_fts MATCH ? ORDER BY f.rank LIMIT ?",
                (_match_expression(query), limit),
            )
        else:
            clause, params = _like_clauses("lower(title || ' ' || content)", query)
            rows = await self._fetch_all(
                f"SELECT * FROM notes WHERE {clause} ORDER BY id DESC LIMIT ?",  # noqa: S608
                (*params, limit),
            )
        return [_note(row) for row in rows]

    async def delete_note(self, note_id: int) -> None:
        """Remove a note.

        Raises:
            NotFoundError: If no such note exists.
        """
        cursor = await self._execute("DELETE FROM notes WHERE id = ?", (note_id,))
        if cursor.rowcount == 0:
            raise NotFoundError("notes", note_id)

    # -- reminders ------------------------------------------------------------

    async def add_reminder(self, text: str, due_at: datetime) -> Reminder:
        """Schedule a reminder. Naive ``due_at`` values are taken as UTC."""
        created = self._now()
        due = _to_db(due_at)
        row_id = await self._insert(
            "INSERT INTO reminders (text, due_at, created_at) VALUES (?, ?, ?)",
            (text, due, created),
        )
        return Reminder(
            id=row_id, text=text, due_at=_from_db(due), fired=False, created_at=_from_db(created)
        )

    async def list_reminders(self, *, include_fired: bool = False) -> list[Reminder]:
        """Reminders ordered by due time, pending ones only unless asked."""
        sql = "SELECT * FROM reminders"
        if not include_fired:
            sql += " WHERE fired = 0"
        rows = await self._fetch_all(sql + " ORDER BY due_at, id")
        return [_reminder(row) for row in rows]

    async def pop_due_reminders(self, now: datetime | None = None) -> list[Reminder]:
        """Return reminders due at or before ``now`` and mark them fired.

        Selecting and marking happen in one transaction so two pollers can
        never deliver the same reminder twice.
        """
        cutoff = _to_db(now) if now is not None else self._now()
        async with self._transaction():
            rows = await self._fetch_all(
                "SELECT * FROM reminders WHERE fired = 0 AND due_at <= ? ORDER BY due_at, id",
                (cutoff,),
            )
            await self._execute(
                "UPDATE reminders SET fired = 1 WHERE fired = 0 AND due_at <= ?", (cutoff,)
            )
        due = [_reminder(row) for row in rows]
        return [
            Reminder(
                id=item.id,
                text=item.text,
                due_at=item.due_at,
                fired=True,
                created_at=item.created_at,
            )
            for item in due
        ]

    async def delete_reminder(self, reminder_id: int) -> None:
        """Cancel a reminder.

        Raises:
            NotFoundError: If no such reminder exists.
        """
        cursor = await self._execute("DELETE FROM reminders WHERE id = ?", (reminder_id,))
        if cursor.rowcount == 0:
            raise NotFoundError("reminders", reminder_id)

    # -- todos ------------------------------------------------------------------

    async def add_todo(self, text: str) -> Todo:
        """Add an open todo."""
        created = self._now()
        row_id = await self._insert(
            "INSERT INTO todos (text, created_at) VALUES (?, ?)", (text, created)
        )
        return Todo(id=row_id, text=text, done=False, created_at=_from_db(created))

    async def list_todos(self, *, include_done: bool = False) -> list[Todo]:
        """Todos in creation order, open ones only unless asked."""
        sql = "SELECT * FROM todos"
        if not include_done:
            sql += " WHERE done = 0"
        rows = await self._fetch_all(sql + " ORDER BY id")
        return [_todo(row) for row in rows]

    async def complete_todo(self, todo_id: int) -> Todo:
        """Tick a todo off and return its new state.

        Raises:
            NotFoundError: If no such todo exists.
        """
        await self._execute("UPDATE todos SET done = 1 WHERE id = ?", (todo_id,))
        row = await self._fetch_one("SELECT * FROM todos WHERE id = ?", (todo_id,))
        if row is None:
            raise NotFoundError("todos", todo_id)
        return _todo(row)

    # -- facts ------------------------------------------------------------------

    async def upsert_fact(
        self, key: str, value: str, *, source: str = "user", confidence: float = 1.0
    ) -> Fact:
        """Insert a fact or update the value, source and confidence of an existing key."""
        now = self._now()
        await self._execute(
            "INSERT INTO facts (key, value, source, confidence, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value, source = excluded.source, "
            "confidence = excluded.confidence, updated_at = excluded.updated_at",
            (key, value, source, confidence, now, now),
        )
        fact = await self.get_fact(key)
        if fact is None:
            raise MemoryStoreError(f"fact {key!r} vanished after upsert")
        return fact

    async def get_fact(self, key: str) -> Fact | None:
        """One fact by key, or ``None``."""
        row = await self._fetch_one("SELECT * FROM facts WHERE key = ?", (key,))
        return _fact(row) if row is not None else None

    async def list_facts(self, prefix: str = "") -> list[Fact]:
        """Facts whose key starts with ``prefix`` (all facts by default), by key."""
        rows = await self._fetch_all(
            "SELECT * FROM facts WHERE key LIKE ? ESCAPE '\\' ORDER BY key",
            (_escape_like(prefix) + "%",),
        )
        return [_fact(row) for row in rows]

    async def delete_fact(self, key: str) -> bool:
        """Forget a fact. Returns whether anything was deleted."""
        cursor = await self._execute("DELETE FROM facts WHERE key = ?", (key,))
        return cursor.rowcount > 0


# Design notes:
# - Autocommit connection plus explicit BEGIN/COMMIT: Python's implicit
#   transaction rules differ between DDL and DML and changed across versions;
#   spelling transactions out makes migrations and pop_due_reminders
#   obviously atomic.
# - FTS5 lives outside the versioned migrations because its availability is
#   a property of the SQLite build, not of the schema; a database can move
#   between builds and keep working either way. The shadow tables are
#   external-content so they add an index, not a copy of the text.
# - Search queries are built by quoting every word rather than passing user
#   text to the FTS5 parser, so a spoken "what's -that" cannot become a
#   syntax error or an unintended NOT.
# - Timestamps are ISO-8601 UTC text with microsecond precision: SQLite has
#   no datetime type, and fixed-width text sorts chronologically, which is
#   what the due-reminder query relies on.
# - Rows are small frozen dataclasses mapped by hand; an ORM would hide the
#   SQL that a reviewer wants to see and add a dependency for six tables.
