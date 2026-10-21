"""Orchestration core: event bus, state machine, orchestrator, interruption.

This package owns the lifecycle of a conversation turn and knows every
stage only through its interface. Nothing here imports a concrete STT,
TTS or LLM implementation; those are injected by ``jarvis.__main__``.
"""
