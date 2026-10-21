"""Python wrappers around the native audio engine (``jarvis_audio``).

Exposes capture, VAD, wake word and playback as asyncio-friendly objects.
The C++ side owns the real-time threads; this layer only moves frames
between those threads and the event loop.
"""
