"""Errors raised by the memory package.

Kept in one small module so ``sqlite_store``, ``vector_store`` and
``memory_manager`` can share them without importing each other.
"""

from __future__ import annotations

from jarvis.utils.errors import JarvisError


class MemoryStoreError(JarvisError):
    """A memory backend could not complete an operation.

    Wraps driver-level failures (SQLite, Chroma) so the agent can degrade to
    "I could not reach my memory" instead of leaking a driver traceback into
    a spoken answer.
    """


class StoreClosedError(MemoryStoreError):
    """A store was used before ``open()`` or after ``close()``."""


class NotFoundError(MemoryStoreError):
    """The requested row does not exist.

    Attributes:
        table: Which table was searched.
        key: The identifier that was looked up.
    """

    def __init__(self, table: str, key: object) -> None:
        self.table = table
        self.key = key
        super().__init__(f"no {table} row with id {key!r}")


# Design notes:
# - NotFoundError is deliberately not a KeyError subclass: a missing todo is
#   something the agent should explain to the user, not a programming error.
