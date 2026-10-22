"""The default output device, read from Core Audio through ``jarvis_os``.

A read-only query outside :class:`~jarvis.os_integration.base.Platform`:
it is needed once, by the voice front end's composition root, before the
platform object exists, and no tool should ever call it.
"""

from __future__ import annotations

from typing import Protocol, cast

from jarvis import _native
from jarvis.os_integration.base import OutputEndpoint
from jarvis.utils.errors import NativeUnavailableError
from jarvis.utils.logging import get_logger

log = get_logger(__name__)


class NativeAudioEndpoint(Protocol):
    """Shape of ``jarvis_os.AudioEndpoint``."""

    @property
    def name(self) -> str:
        """Friendly name of the device."""

    @property
    def form_factor(self) -> str:
        """Core Audio form factor in snake_case."""


def default_output_endpoint() -> OutputEndpoint | None:
    """The default output device, or None when the native module or the device is missing."""
    try:
        module = _native.load("jarvis_os")
    except NativeUnavailableError as error:
        log.info("audio_endpoint.native_unavailable", error=str(error))
        return None
    probe = getattr(module, "default_output_endpoint", None)
    if probe is None:
        log.info("audio_endpoint.native_too_old")  # a build from before this query existed
        return None
    native = cast(NativeAudioEndpoint | None, probe())
    if native is None:
        return None
    return OutputEndpoint(name=native.name, form_factor=native.form_factor)


# Design notes:
# - Every failure is None, which the caller treats as "probably speakers":
#   the safe choice, because it keeps Jarvis from interrupting itself.
# - The getattr guard keeps a stale .pyd from an older build working; the
#   decision then falls back to the device names.
