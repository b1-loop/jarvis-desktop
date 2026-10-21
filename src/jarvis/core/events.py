"""Typed event bus connecting pipeline stages.

Stages publish events (state transitions, transcripts, tool calls) and
subscribers such as the overlay, tray icon and logger react without the
publisher knowing they exist. Arrives in step 6 (agent core).
"""
