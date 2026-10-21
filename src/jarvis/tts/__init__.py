"""Text-to-speech providers behind a common ``TTS`` interface.

Providers stream audio chunks as they are synthesised so playback starts
before the full sentence exists; the orchestrator relies on that for
perceived latency.
"""
