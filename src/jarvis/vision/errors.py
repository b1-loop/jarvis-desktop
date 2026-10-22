"""Errors raised by screen understanding.

Two classes because the tools phrase them differently: an unavailable
feature ("vision is off", "this model cannot see images") is a setup fact
the user can change, whereas a failed attempt (capture refused, model
error, image too large) is a one-off the user may simply retry.
"""

from __future__ import annotations

from jarvis.utils.errors import JarvisError


class VisionError(JarvisError):
    """A screen question could not be answered this time."""


class VisionUnavailableError(VisionError):
    """Screen understanding cannot work in the current configuration.

    Attributes:
        reason: Short, speakable cause including the remedy where there is
            one, e.g. "screen understanding is turned off".
    """

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(f"screen understanding is unavailable: {reason}")


# Design notes:
# - A module of its own, like os_integration/errors.py, so the vision
#   hierarchy can grow without touching the shared utils/errors.py.
