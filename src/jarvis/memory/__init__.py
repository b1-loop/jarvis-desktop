"""Persistent memory: SQLite for facts and history, ChromaDB for semantics.

``MemoryManager`` is the only entry point the agent uses; the stores
behind it are swappable and individually testable. See ADR-0008.
"""

from jarvis.memory.memory_manager import MemoryManager, RecalledContext
from jarvis.memory.sqlite_store import SqliteStore
from jarvis.memory.user_profile import UserProfile
from jarvis.memory.vector_store import ChromaVectorStore, InMemoryVectorStore, VectorStore

__all__ = [
    "ChromaVectorStore",
    "InMemoryVectorStore",
    "MemoryManager",
    "RecalledContext",
    "SqliteStore",
    "UserProfile",
    "VectorStore",
]
