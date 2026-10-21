"""Wires the pipeline stages together and runs the main loop.

Receives already-constructed stage instances (dependency injection) and
owns the asyncio tasks that move audio and text between them. Arrives in
step 6 (text-only) and step 7 (full voice).
"""
