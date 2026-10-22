"""Registry-level guarantees for the computer, file, dev, notes and system tools."""

import json
from typing import Any

import pytest

from jarvis.agent.registry import ToolRegistry
from jarvis.agent.tool import ToolCategory, ToolContext
from jarvis.config import Settings
from tests.tools.local_fakes import local_context

ALWAYS_CONFIRM = frozenset(
    {"close_app", "lock_screen", "type_text", "git_push", "start_dev_server"}
)
"""Tools that change something hard to undo or act outside the user's explicit target."""

CONFIRM_PER_CALL = {"window_action": {"action": "close"}}
"""Tools whose risk depends on the arguments: these arguments must confirm."""

NEVER_CONFIRM = frozenset(
    {
        "open_app",
        "list_windows",
        "switch_window",
        "get_volume",
        "set_volume",
        "mute",
        "set_brightness",
        "take_screenshot",
        "clipboard_get",
        "clipboard_set",
        "notify",
        "find_files",
        "recent_files",
        "open_file",
        "summarize_document",
        "git_status",
        "git_log",
        "git_diff_summary",
        "run_tests",
        "open_in_vscode",
        "explain_stack_trace",
        "add_note",
        "search_notes",
        "add_todo",
        "list_todos",
        "complete_todo",
        "add_reminder",
        "list_reminders",
        "battery_status",
    }
)

LOCAL_TOOLS = ALWAYS_CONFIRM | CONFIRM_PER_CALL.keys() | NEVER_CONFIRM


@pytest.fixture(scope="module")
def registry() -> ToolRegistry:
    return ToolRegistry.discover()


@pytest.fixture
def context(make_settings: Any) -> ToolContext:
    settings: Settings = make_settings()
    return local_context(settings)


def test_every_local_tool_is_discovered_without_failures(registry: ToolRegistry) -> None:
    assert set(registry.names) >= LOCAL_TOOLS
    assert registry.failures == ()


def test_names_are_unique_across_the_sets() -> None:
    sizes = len(ALWAYS_CONFIRM) + len(CONFIRM_PER_CALL) + len(NEVER_CONFIRM)

    assert sizes == len(LOCAL_TOOLS)


@pytest.mark.parametrize("name", sorted(LOCAL_TOOLS))
def test_schema_is_valid_json_schema_with_described_fields(
    registry: ToolRegistry, name: str
) -> None:
    jsonschema = pytest.importorskip("jsonschema")
    schema = registry.get(name).to_llm_schema()

    jsonschema.Draft202012Validator.check_schema(schema["input_schema"])
    assert json.loads(json.dumps(schema)) == schema
    assert len(schema["description"]) >= 40
    assert schema["input_schema"]["type"] == "object"
    for field_name, field in schema["input_schema"].get("properties", {}).items():
        assert field.get("description"), f"{name}.{field_name} has no description"


@pytest.mark.parametrize("name", sorted(ALWAYS_CONFIRM))
def test_state_changing_tools_require_confirmation(registry: ToolRegistry, name: str) -> None:
    assert registry.get(name).requires_confirmation


@pytest.mark.parametrize(("name", "raw"), sorted(CONFIRM_PER_CALL.items()))
def test_argument_dependent_tools_confirm_the_risky_call(
    registry: ToolRegistry, context: ToolContext, name: str, raw: dict[str, object]
) -> None:
    tool = registry.get(name)

    assert tool.needs_confirmation(tool.parse(raw), context)


@pytest.mark.parametrize("name", sorted(NEVER_CONFIRM))
def test_read_only_or_reversible_tools_do_not_nag(registry: ToolRegistry, name: str) -> None:
    assert not registry.get(name).requires_confirmation


def test_categories_group_the_tools(registry: ToolRegistry) -> None:
    categories = {registry.get(name).category for name in LOCAL_TOOLS}

    assert categories == {
        ToolCategory.SYSTEM,
        ToolCategory.FILES,
        ToolCategory.DEVELOPER,
        ToolCategory.PRODUCTIVITY,
    }
