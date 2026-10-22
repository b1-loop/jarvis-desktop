"""Cross-cutting checks for every integration tool.

Each tool has its own test file; this one asserts the properties that must
hold for all of them at once: they are discovered, their schemas are valid,
every tool that sends or creates something confirms, and none of them
crashes when the composition root wired no integrations.
"""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from jarvis.agent.registry import ToolRegistry
from jarvis.agent.tool import Tool
from jarvis.config import Settings
from tests.tools.integration_helpers import integration_context

INTEGRATION_TOOLS = (
    "list_unread_email",
    "summarize_inbox",
    "search_email",
    "draft_email_reply",
    "send_email",
    "todays_agenda",
    "find_free_slot",
    "create_event",
    "upcoming_events",
    "web_search",
    "read_article",
    "summarize_youtube",
    "read_messages",
    "send_message",
    "music_control",
    "play_music",
    "queue_music",
    "now_playing",
    "list_devices",
    "set_device",
    "draft_document",
)

ALWAYS_CONFIRM = {"send_email", "create_event", "send_message"}
"""Tools that send to other people or create in an external account."""

CONFIRM_PER_CALL = {"set_device", "draft_document"}
"""Tools whose risk depends on the arguments (sensitive device, overwrite)."""

MINIMAL_ARGUMENTS: dict[str, dict[str, Any]] = {
    "search_email": {"query": "invoice"},
    "draft_email_reply": {"message_id": "gmail:1", "body": "Thanks!"},
    "send_email": {
        "draft_id": "gmail:d1",
        "to": ["a@b.co"],
        "subject": "Re: x",
        "first_line": "Hi",
    },
    "create_event": {"title": "Sync", "start": "2026-10-22T10:00"},
    "web_search": {"query": "weather"},
    "read_article": {"url": "https://example.org/post"},
    "summarize_youtube": {"video": "dQw4w9WgXcQ"},
    "send_message": {"channel": "general", "text": "hi"},
    "music_control": {"action": "pause"},
    "play_music": {"query": "jazz"},
    "queue_music": {"query": "jazz"},
    "set_device": {"device_id": "light.kitchen", "action": "on"},
    "draft_document": {"title": "x", "content": "y", "source_email_id": "gmail:1"},
}


@pytest.fixture(scope="module")
def registry() -> ToolRegistry:
    return ToolRegistry.discover()


def test_every_integration_tool_is_discovered_once(registry: ToolRegistry) -> None:
    assert set(INTEGRATION_TOOLS) <= set(registry.names)
    assert len(set(INTEGRATION_TOOLS)) == len(INTEGRATION_TOOLS)
    assert registry.failures == ()


@pytest.mark.parametrize("name", INTEGRATION_TOOLS)
def test_schema_is_a_described_json_object(registry: ToolRegistry, name: str) -> None:
    schema = registry.get(name).to_llm_schema()

    json.dumps(schema)  # must be JSON-serialisable as sent to the LLM
    assert schema["name"] == name
    assert len(schema["description"]) > 60  # says when to use it and what it returns
    input_schema = schema["input_schema"]
    assert input_schema["type"] == "object"
    assert input_schema.get("additionalProperties") is False
    for field_name, field in input_schema.get("properties", {}).items():
        assert field.get("description"), f"{name}.{field_name} has no description"


@pytest.mark.parametrize("name", INTEGRATION_TOOLS)
def test_schema_satisfies_the_json_schema_meta_schema(registry: ToolRegistry, name: str) -> None:
    validator = pytest.importorskip("jsonschema").Draft202012Validator

    validator.check_schema(registry.get(name).to_llm_schema()["input_schema"])


@pytest.mark.parametrize("name", INTEGRATION_TOOLS)
def test_sending_and_creating_tools_confirm(registry: ToolRegistry, name: str) -> None:
    tool = registry.get(name)
    overrides_decision = type(tool).needs_confirmation is not Tool.needs_confirmation

    if name in ALWAYS_CONFIRM:
        assert tool.requires_confirmation
    elif name in CONFIRM_PER_CALL:
        assert overrides_decision
    else:
        assert not tool.requires_confirmation


@pytest.mark.parametrize("name", INTEGRATION_TOOLS)
async def test_degrades_cleanly_without_integrations(
    registry: ToolRegistry, name: str, make_settings: Callable[..., Settings], tmp_path: Path
) -> None:
    context = integration_context(make_settings(documents_output_dir=tmp_path), None)

    result = await registry.get(name).call(MINIMAL_ARGUMENTS.get(name, {}), context)

    assert not result.ok
    assert result.spoken.startswith("That integration isn't configured")
    assert result.data is not None
    assert result.data["configured"] is False
    assert result.data["env_vars"]
