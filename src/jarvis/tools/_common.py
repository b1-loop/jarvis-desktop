"""Small helpers shared by the local tool plugins.

Private (underscore) module, so the registry never scans it for tools. It
holds only what several plugins would otherwise copy: fetching an optional
service from the context with a clear error, and trimming text for speech.
Tools still never import each other.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from jarvis.agent.tool import ToolError

if TYPE_CHECKING:
    from jarvis.agent.tool import MemoryFacade, ToolContext
    from jarvis.os_integration.base import Platform

ELLIPSIS: Final = "…"


class ServiceUnavailableError(ToolError):
    """The tool needs a service that this run was started without.

    Typical cause: text-only mode without an OS backend, or memory disabled.
    The message names the service so the LLM can tell the user why.

    Attributes:
        service: Human-readable name of the missing service.
    """

    def __init__(self, tool_name: str, service: str) -> None:
        self.service = service
        super().__init__(tool_name, f"the {service} is not available in this session")


def require_platform(context: ToolContext, tool_name: str) -> Platform:
    """The OS facade, or :class:`ServiceUnavailableError` when there is none."""
    if context.platform is None:
        raise ServiceUnavailableError(tool_name, "operating-system integration")
    return context.platform


def require_memory(context: ToolContext, tool_name: str) -> MemoryFacade:
    """The memory facade, or :class:`ServiceUnavailableError` when there is none."""
    if context.memory is None:
        raise ServiceUnavailableError(tool_name, "memory store")
    return context.memory


def truncate(text: str, limit: int) -> str:
    """Cut ``text`` to at most ``limit`` characters, marking the cut."""
    if len(text) <= limit:
        return text
    return text[: max(limit - len(ELLIPSIS), 0)] + ELLIPSIS


# Design notes:
# - ServiceUnavailableError derives from ToolError (a JarvisError), so
#   Tool.execute turns it into a failed result carrying this message instead
#   of a crash: a missing service is an expected condition, not a bug.
