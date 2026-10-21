"""Auto-discovering tool registry.

Every module in ``jarvis.tools`` that defines a concrete ``Tool`` subclass
is a plugin; dropping a file in adds a capability. The registry walks the
package with ``pkgutil``, imports each module and instantiates the tools it
finds. A plugin that fails to import, declares an invalid definition or
reuses a name is logged, recorded for the dashboard and skipped, so one
broken tool never prevents Jarvis from starting.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
from collections.abc import Iterator
from dataclasses import dataclass
from types import ModuleType
from typing import Any, Final

from jarvis.agent.tool import Tool, ToolCategory, ToolError
from jarvis.utils.logging import get_logger

log = get_logger(__name__)

DEFAULT_PACKAGE: Final = "jarvis.tools"

AnyTool = Tool[Any]
"""A tool of any input type, as the registry stores them."""


class UnknownToolError(ToolError):
    """The LLM (or a caller) asked for a tool name that is not registered."""

    def __init__(self, tool_name: str) -> None:
        super().__init__(tool_name, "no such tool is registered")


class DuplicateToolError(ToolError):
    """Two plugins declared the same ``name``."""

    def __init__(self, tool_name: str, module: str) -> None:
        self.module = module
        super().__init__(tool_name, f"already registered; duplicate in {module}")


@dataclass(frozen=True, slots=True)
class DiscoveryFailure:
    """A plugin that could not be loaded, kept for the dashboard.

    Attributes:
        module: Fully qualified module name.
        error: ``repr`` of the exception.
    """

    module: str
    error: str


class ToolRegistry:
    """Name-indexed collection of tool instances with discovery failures."""

    def __init__(self) -> None:
        self._tools: dict[str, AnyTool] = {}
        self._failures: list[DiscoveryFailure] = []

    @classmethod
    def discover(cls, package_name: str = DEFAULT_PACKAGE) -> ToolRegistry:
        """Build a registry from every plugin module in ``package_name``."""
        registry = cls()
        registry.load_package(package_name)
        return registry

    @property
    def failures(self) -> tuple[DiscoveryFailure, ...]:
        """Plugins that were skipped, in discovery order."""
        return tuple(self._failures)

    @property
    def names(self) -> tuple[str, ...]:
        """Registered tool names, sorted."""
        return tuple(sorted(self._tools))

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def __iter__(self) -> Iterator[AnyTool]:
        return iter(self.tools())

    def load_package(self, package_name: str) -> None:
        """Import every public module directly inside ``package_name``.

        Raises:
            ModuleNotFoundError: If the package itself does not exist; that
                is a deployment error, not a plugin error.
        """
        package = importlib.import_module(package_name)
        for info in pkgutil.iter_modules(package.__path__):
            if info.name.startswith("_"):
                continue
            self._load_module(f"{package_name}.{info.name}")

    def register(self, tool: AnyTool) -> None:
        """Add ``tool``; mainly for tests and for tools built by hand.

        Raises:
            DuplicateToolError: If a tool with the same name exists.
        """
        if tool.name in self._tools:
            raise DuplicateToolError(tool.name, type(tool).__module__)
        self._tools[tool.name] = tool
        log.info("tools.registered", tool=tool.name, category=tool.category.value)

    def get(self, name: str) -> AnyTool:
        """Look a tool up by name.

        Raises:
            UnknownToolError: If no tool has that name.
        """
        try:
            return self._tools[name]
        except KeyError:
            raise UnknownToolError(name) from None

    def tools(self, category: ToolCategory | None = None) -> list[AnyTool]:
        """All tools, or those in ``category``, sorted by name."""
        return [
            tool
            for name, tool in sorted(self._tools.items())
            if category is None or tool.category is category
        ]

    def schemas(self, category: ToolCategory | None = None) -> list[dict[str, Any]]:
        """LLM tool definitions for ``tools(category)``."""
        return [tool.to_llm_schema() for tool in self.tools(category)]

    def _load_module(self, module_name: str) -> None:
        try:
            module = importlib.import_module(module_name)
        except Exception as error:
            self._record_failure(module_name, error)
            return
        for tool_class in _tool_classes(module):
            self._register_discovered(module_name, tool_class)

    def _register_discovered(self, module_name: str, tool_class: type[AnyTool]) -> None:
        try:
            self.register(tool_class())
        except Exception as error:
            self._record_failure(module_name, error)

    def _record_failure(self, module_name: str, error: Exception) -> None:
        log.error("tools.discovery.failed", module=module_name, error=repr(error), exc_info=error)
        self._failures.append(DiscoveryFailure(module=module_name, error=repr(error)))


def _tool_classes(module: ModuleType) -> list[type[AnyTool]]:
    """Concrete ``Tool`` subclasses defined (not merely imported) in ``module``."""
    return [
        obj
        for _, obj in inspect.getmembers(module, inspect.isclass)
        if issubclass(obj, Tool)
        and obj.__module__ == module.__name__
        and not inspect.isabstract(obj)
    ]


# Design notes:
# - pkgutil + importlib rather than entry points or a hand-written list:
#   the plugin contract is "a file in tools/ with a Tool subclass", which a
#   contributor can satisfy without touching any other file (ADR-0010).
# - Only classes whose ``__module__`` is the scanned module count, so a
#   plugin that imports another tool's class for reuse does not register it
#   twice.
# - Failures are recorded, not raised: the assistant should start and tell
#   the user "the X tool failed to load" rather than refuse to run at all.
#   The dashboard reads ``failures``.
