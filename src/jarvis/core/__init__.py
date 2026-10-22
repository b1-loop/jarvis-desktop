"""Orchestration core: event bus, state machine, orchestrator, interruption.

This package owns the lifecycle of a conversation turn and knows every
stage only through its interface. The orchestrator never imports a
concrete STT, TTS or LLM implementation; they are built by
``jarvis.core.providers`` (the provider half of the composition root) and
injected by ``jarvis.__main__``.
"""
