"""Exception hierarchy for Jarvis.

Every error raised deliberately by Jarvis derives from :class:`JarvisError`,
so callers can catch "anything Jarvis knows how to explain" in one clause
while programming errors (``TypeError``, ``KeyError`` ...) still surface
loudly. Subclasses carry structured context so logs and spoken error messages
can be specific without string parsing.
"""


class JarvisError(Exception):
    """Base class for all errors raised by Jarvis.

    Subclasses should add attributes for the context that matters (module
    name, tool name, stage) rather than encoding it only in the message.
    """


class ConfigError(JarvisError):
    """Configuration is missing or inconsistent for the requested mode.

    Raised at startup, never mid-conversation, so the user gets one clear
    message instead of a failure deep in the pipeline.
    """


class NativeUnavailableError(JarvisError):
    """A compiled extension module could not be loaded.

    Attributes:
        module: Name of the pybind11 module, e.g. ``"jarvis_audio"``.
        reason: Human-readable cause, typically the underlying ``ImportError``.
    """

    def __init__(self, module: str, reason: str) -> None:
        self.module = module
        self.reason = reason
        super().__init__(
            f"native module '{module}' is unavailable: {reason}. "
            "Run 'uv run python scripts/build_native.py' or use --text-only."
        )


# Design notes:
# - The hierarchy is intentionally shallow at this stage. Stage-, tool- and
#   integration-specific errors are added in the step that introduces them,
#   next to the code that raises them, so every class has a real raiser.
# - NativeUnavailableError spells out the remedy because the most common cause
#   is a fresh checkout without a native build, not a real bug.
