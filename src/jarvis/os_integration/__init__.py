"""Python facade over the native OS layer (``jarvis_os``).

Defines the ``Platform`` interface that tools depend on, so a tool never
touches Win32 directly and a macOS or Linux backend can be dropped in
later without changing a single tool. ``WindowsPlatform`` is the real
backend; ``StubPlatform`` records calls for text-only mode and tests.
"""

from jarvis.os_integration.base import HotkeyEvent, Platform, WindowInfo
from jarvis.os_integration.errors import (
    InvalidHotkeyError,
    PlatformError,
    ProcessNotFoundError,
    UnsupportedFeatureError,
    WindowNotFoundError,
)
from jarvis.os_integration.stub import StubPlatform
from jarvis.os_integration.windows import WindowsPlatform, create_windows_platform

__all__ = [
    "HotkeyEvent",
    "InvalidHotkeyError",
    "Platform",
    "PlatformError",
    "ProcessNotFoundError",
    "StubPlatform",
    "UnsupportedFeatureError",
    "WindowInfo",
    "WindowNotFoundError",
    "WindowsPlatform",
    "create_windows_platform",
]
