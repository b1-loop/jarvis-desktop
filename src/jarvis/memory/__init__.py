"""Persistent memory: SQLite for facts and history, ChromaDB for semantics.

``MemoryManager`` is the only entry point the agent uses; the stores
behind it are swappable and individually testable.
"""
