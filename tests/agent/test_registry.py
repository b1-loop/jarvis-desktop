import sys
from pathlib import Path
from textwrap import dedent

import pytest

from jarvis.agent.registry import (
    DuplicateToolError,
    ToolRegistry,
    UnknownToolError,
)
from jarvis.agent.tool import ToolCategory
from jarvis.utils.errors import JarvisError
from tests.agent.fakes import DestructiveTool, EchoTool, SendTool, tool_names

PLUGIN_SOURCES = {
    "alpha_tool": """
        from jarvis.agent.tool import Tool, ToolCategory, ToolContext, ToolInput, ToolResult

        class AlphaInput(ToolInput):
            x: int = 0

        class AlphaTool(Tool[AlphaInput]):
            name = "alpha"
            description = "First plugin."
            Input = AlphaInput
            category = ToolCategory.WEB

            async def run(self, params: AlphaInput, context: ToolContext) -> ToolResult:
                return ToolResult.success("alpha")
        """,
    "beta_tool": """
        from jarvis.agent.tool import Tool, ToolContext, ToolInput, ToolResult
        from plugins_pkg.alpha_tool import AlphaTool  # imported, must not register twice

        class BetaInput(ToolInput):
            pass

        class BetaTool(Tool[BetaInput]):
            name = "beta"
            description = "Second plugin."
            Input = BetaInput

            async def run(self, params: BetaInput, context: ToolContext) -> ToolResult:
                return ToolResult.success("beta")
        """,
    "broken_import": """
        import module_that_does_not_exist
        """,
    "bad_definition": """
        from jarvis.agent.tool import Tool, ToolContext, ToolInput, ToolResult

        class Nameless(Tool[ToolInput]):
            description = "forgot the name"
            Input = ToolInput

            async def run(self, params: ToolInput, context: ToolContext) -> ToolResult:
                return ToolResult.success("x")
        """,
    "duplicate_alpha": """
        from jarvis.agent.tool import Tool, ToolContext, ToolInput, ToolResult

        class AnotherAlpha(Tool[ToolInput]):
            name = "alpha"
            description = "Same name as alpha_tool."
            Input = ToolInput

            async def run(self, params: ToolInput, context: ToolContext) -> ToolResult:
                return ToolResult.success("x")
        """,
    "needs_arguments": """
        from jarvis.agent.tool import Tool, ToolContext, ToolInput, ToolResult

        class NeedsArgs(Tool[ToolInput]):
            name = "needs_args"
            description = "Cannot be built without arguments."
            Input = ToolInput

            def __init__(self, required: int) -> None:
                self.required = required

            async def run(self, params: ToolInput, context: ToolContext) -> ToolResult:
                return ToolResult.success("x")
        """,
    "_private": """
        raise RuntimeError("private modules must never be imported")
        """,
    "helpers": """
        def not_a_tool() -> None:
            pass
        """,
}


@pytest.fixture
def plugin_package(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    """A throwaway plugin package on sys.path with one module per source above."""
    package = tmp_path / "plugins_pkg"
    package.mkdir()
    (package / "__init__.py").write_text('"""Test plugins."""\n', encoding="utf-8")
    for module, source in PLUGIN_SOURCES.items():
        (package / f"{module}.py").write_text(dedent(source), encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    for name in [name for name in sys.modules if name.startswith("plugins_pkg")]:
        monkeypatch.delitem(sys.modules, name)
    return "plugins_pkg"


def test_discovery_registers_good_plugins_and_records_the_rest(plugin_package: str) -> None:
    registry = ToolRegistry.discover(plugin_package)

    assert registry.names == ("alpha", "beta")
    assert len(registry) == 2
    assert "alpha" in registry
    failed = {failure.module.removeprefix("plugins_pkg.") for failure in registry.failures}
    assert failed == {"broken_import", "bad_definition", "duplicate_alpha", "needs_arguments"}


def test_discovery_failures_carry_the_error(plugin_package: str) -> None:
    registry = ToolRegistry.discover(plugin_package)
    by_module = {f.module.removeprefix("plugins_pkg."): f.error for f in registry.failures}

    assert "ModuleNotFoundError" in by_module["broken_import"]
    assert "missing class attributes: name" in by_module["bad_definition"]
    assert "already registered" in by_module["duplicate_alpha"]
    assert "TypeError" in by_module["needs_arguments"]


def test_discovering_a_missing_package_is_a_deployment_error() -> None:
    with pytest.raises(ModuleNotFoundError):
        ToolRegistry.discover("jarvis.no_such_package")


def test_the_real_tools_package_loads_without_failures() -> None:
    registry = ToolRegistry.discover()

    assert registry.failures == ()


def test_register_get_and_iteration() -> None:
    registry = ToolRegistry()
    echo, destroy = EchoTool(), DestructiveTool()
    registry.register(destroy)
    registry.register(echo)

    assert registry.get("echo") is echo
    assert tool_names(list(registry)) == ["destroy", "echo"]
    assert registry.names == ("destroy", "echo")


def test_register_rejects_duplicate_names() -> None:
    registry = ToolRegistry()
    registry.register(EchoTool())

    with pytest.raises(DuplicateToolError) as info:
        registry.register(EchoTool())

    assert isinstance(info.value, JarvisError)
    assert info.value.tool_name == "echo"
    assert info.value.module == "tests.agent.fakes"


def test_get_unknown_tool() -> None:
    with pytest.raises(UnknownToolError, match="nope"):
        ToolRegistry().get("nope")


def test_filter_by_category_and_schemas() -> None:
    registry = ToolRegistry()
    for tool in (EchoTool(), DestructiveTool(), SendTool()):
        registry.register(tool)

    assert tool_names(registry.tools(ToolCategory.FILES)) == ["destroy"]
    assert tool_names(registry.tools()) == ["destroy", "echo", "send"]
    assert [s["name"] for s in registry.schemas()] == ["destroy", "echo", "send"]
    assert [s["name"] for s in registry.schemas(ToolCategory.COMMUNICATION)] == ["send"]
    assert registry.schemas(ToolCategory.MEDIA) == []
