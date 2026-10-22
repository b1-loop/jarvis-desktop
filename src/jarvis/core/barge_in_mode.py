"""Which barge-in mode this session uses: voice on headphones, the wake word on speakers.

Talking over Jarvis (``voice``) only works when the microphone cannot hear
Jarvis itself. On headphones it cannot; on laptop speakers it does, and the
VAD then "hears the user" in Jarvis's own voice and cuts the answer off. The
``auto`` setting therefore looks at the default output device once at
start-up: Core Audio's endpoint form factor (headphones or headset) when the
native OS module can read it, otherwise the device names, and picks
``voice`` for headphones and ``wake_word`` for everything else.
"""

from __future__ import annotations

from collections.abc import Callable
from enum import StrEnum
from typing import Final

from jarvis.os_integration.base import OutputEndpoint
from jarvis.utils.logging import get_logger

log = get_logger(__name__)

AUTO_MODE: Final = "auto"

HEADPHONE_FORM_FACTORS: Final = frozenset({"headphones", "headset"})
"""Core Audio form factors (``jarvis_os.AudioEndpoint.form_factor``) worn on the head."""

HEADPHONE_NAME_HINTS: Final = ("headphone", "headset", "hörlur")
"""Substrings of device names that mean headphones (English and Swedish)."""


class BargeInMode(StrEnum):
    """What may interrupt Jarvis while it is speaking.

    VOICE: a speech onset (talking over Jarvis), the wake word, the hotkey or typing.
    WAKE_WORD: only the wake word, the hotkey or typing; plain speech and echo are ignored.
    OFF: only the hotkey or typing.
    """

    VOICE = "voice"
    WAKE_WORD = "wake_word"
    OFF = "off"


EndpointProbe = Callable[[], OutputEndpoint | None]
"""Reads the default output endpoint; None when the OS layer cannot tell."""

NameProbe = Callable[[], str | None]
"""Another device name to consult (the default microphone's), or None."""


def looks_like_headphones(endpoint: OutputEndpoint | None, microphone: str | None) -> bool:
    """Headphones by form factor, or by name when the form factor is unknown.

    The microphone name counts too: a headset's microphone is called
    "Headset Microphone" even when its output side reports a generic form.
    """
    if endpoint is not None and endpoint.form_factor in HEADPHONE_FORM_FACTORS:
        return True
    if endpoint is not None and endpoint.form_factor not in {"", "unknown"}:
        return False  # the driver said what it is (speakers, HDMI, line out)
    names = [endpoint.name if endpoint else None, microphone]
    return any(_has_headphone_hint(name) for name in names if name)


def resolve_barge_in_mode(
    configured: str, *, endpoint: EndpointProbe, microphone: NameProbe
) -> BargeInMode:
    """The effective mode for this session; ``auto`` decides from the devices.

    Logs the decision once, with the evidence, so a surprising choice can
    be explained from ``jarvis.log``.
    """
    if configured != AUTO_MODE:
        mode = BargeInMode(configured)
        log.info("barge_in.mode", mode=mode.value, source="configured")
        return mode
    device = endpoint()
    mic = microphone()
    headphones = looks_like_headphones(device, mic)
    mode = BargeInMode.VOICE if headphones else BargeInMode.WAKE_WORD
    log.info(
        "barge_in.mode",
        mode=mode.value,
        source="auto",
        output=device.name if device else None,
        form_factor=device.form_factor if device else None,
        microphone=mic,
    )
    return mode


def _has_headphone_hint(name: str) -> bool:
    lowered = name.casefold()
    return any(hint in lowered for hint in HEADPHONE_NAME_HINTS)


# Design notes:
# - Decided once per session, not per answer: switching devices mid-session
#   is rare, and a mode that changed between two answers would be harder to
#   understand than one that needs a restart (the log line says why).
# - Unknown means speakers. The cost of guessing wrong that way is that the
#   user has to say the wake word to interrupt; the cost of guessing wrong
#   the other way is Jarvis interrupting itself, which is the bug this fixes.
# - A declared form factor beats a name: "Speakers (Bluetooth Headset)" is
#   rare but real, and the driver knows better than a substring match.
