"""Jarvis: a voice-controlled personal AI assistant for the desktop.

Python is the conductor, C++ is the muscle. Orchestration, agent logic and
tools live here; the real-time audio, OS hooks, local inference and overlay
live in ``native/`` and are reached through :mod:`jarvis._native`.

See ``README.md`` for the overview and ``ARCHITECTURE.md`` for the design.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("jarvis")
except PackageNotFoundError:  # running from a checkout that was never installed
    __version__ = "0.0.0+unknown"

__all__ = ["__version__"]
