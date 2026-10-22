"""End to end in text mode: scripted typing, a fake LLM, everything else real.

Real tool registry (plugin discovery), real ``StubPlatform``, real SQLite
memory under ``tmp_path``, real orchestrator, agent, chaining and
confirmation policy. Only the model and the keyboard are scripted.
"""

from collections.abc import Callable
from pathlib import Path

import pytest

from jarvis.config import Settings
from jarvis.core.bootstrap import MEMORY_DB_FILENAME, Overrides, build_application, build_registry
from jarvis.core.events import Event, ToolCallFinished
from jarvis.llm.base import SystemMessage, ToolResultMessage
from jarvis.memory import SqliteStore
from jarvis.memory.models import Role
from jarvis.memory.vector_store import InMemoryVectorStore
from jarvis.os_integration.stub import StubPlatform
from tests.agent.llm_fakes import ScriptedLLM, call, text_round, tool_round
from tests.core.fakes import RecordingOutput, ScriptedInput

pytestmark = pytest.mark.integration


async def test_text_mode_answers_with_tools_and_remembers_the_turn(
    make_settings: Callable[..., Settings], tmp_path: Path
) -> None:
    llm = ScriptedLLM(
        rounds=[
            tool_round(call("t1", "current_time")),
            text_round("It is a fine evening, sir."),
            text_round("You asked me for the time earlier."),
        ]
    )
    output = RecordingOutput()
    overrides = Overrides(
        source=ScriptedInput(["what time is it?", "what did I ask?", "exit"]),
        sink=output,
        llm=llm,
        platform=StubPlatform(),
        vectors=InMemoryVectorStore(),
        registry=build_registry(),
    )
    application = await build_application(make_settings(data_dir=tmp_path), overrides)
    finished: list[Event] = []
    application.agent.context.bus.subscribe_callback(ToolCallFinished, finished.append)

    await application.run()

    assert output.said == [
        "It is a fine evening, sir.",
        "You asked me for the time earlier.",
        "Goodbye.",
    ]
    assert [e.name for e in finished if isinstance(e, ToolCallFinished)] == ["current_time"]
    # The first turn was answered with a tool, so its stale data stays out of
    # the second prompt (ADR-0029); the turn itself is still stored below.
    second_system = llm.requests[2][0]
    assert isinstance(second_system, SystemMessage)
    assert "It is a fine evening" not in second_system.text
    async with SqliteStore(tmp_path / MEMORY_DB_FILENAME) as store:
        messages = await store.recent_messages(10)
    assert [(m.role, m.content) for m in messages][:2] == [
        (Role.USER, "what time is it?"),
        (Role.ASSISTANT, "It is a fine evening, sir."),
    ]
    assert messages[1].tool_calls[0]["name"] == "current_time"


async def test_destructive_shell_command_is_confirmed_in_the_terminal(
    make_settings: Callable[..., Settings], tmp_path: Path
) -> None:
    command = "git push --force"
    llm = ScriptedLLM(
        rounds=[
            tool_round(call("t1", "shell_command", command=command)),
            text_round("Understood, I did not push."),
        ]
    )
    source = ScriptedInput(["force push my branch", "no", "exit"])
    settings = make_settings(data_dir=tmp_path, shell_working_root=tmp_path)
    overrides = Overrides(
        source=source,
        sink=RecordingOutput(),
        llm=llm,
        platform=StubPlatform(),
        vectors=InMemoryVectorStore(),
        registry=build_registry(),
    )
    application = await build_application(settings, overrides)

    await application.run()

    assert any(prompt.endswith("[y/n] ") for prompt in source.prompts)
    declined = llm.requests[1][-1]
    assert isinstance(declined, ToolResultMessage)
    assert declined.results[0].is_error
    assert "declined" in declined.results[0].content
