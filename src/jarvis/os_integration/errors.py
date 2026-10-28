"""Errors raised by the OS integration layer.

Tools catch :class:`PlatformError` to report "the computer refused" in one
clause; the subclasses let them phrase the refusal precisely ("that window
is gone", "this machine has no brightness control") without parsing the
message. The Windows backend maps the native ``jarvis_os`` exceptions onto
these so that no tool ever imports a pybind11 type.
"""

from __future__ import annotations

from jarvis.utils.errors import JarvisError


class PlatformError(JarvisError):
    """An operating-system operation failed."""


class WindowNotFoundError(PlatformError):
    """A window handle is stale or no window matched a title search."""


class ProcessNotFoundError(PlatformError):
    """A process id is gone or no running process matched a name."""


class UnsupportedFeatureError(PlatformError):
    """The capability exists in the API but not on this machine.

    Typical cause: brightness control on a desktop with external monitors.
    """


class ClipboardBusyError(PlatformError):
    """Another application kept the clipboard open past the retry window.

    Transient, unlike a plain :class:`PlatformError`: asking again in a
    moment usually works, so a tool can say so instead of "it failed".
    """


class InvalidHotkeyError(PlatformError):
    """A hotkey specification such as ``"ctrl+alt+j"`` could not be parsed."""


# Design notes:
# - A separate module instead of jarvis.utils.errors so the OS layer can
#   evolve its hierarchy without touching the shared file that other steps
#   edit concurrently; everything still roots at JarvisError.
