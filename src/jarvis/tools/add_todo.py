"""Tool: add an item to the todo list.

"Add buy milk to my todo list", "I need to renew the passport". Todos are
open until complete_todo ticks them off.
"""

from __future__ import annotations

from typing import Final

from pydantic import Field

from jarvis.agent.tool import Tool, ToolCategory, ToolContext, ToolInput, ToolResult
from jarvis.tools._common import require_memory

MAX_TODO_CHARS: Final = 500


class AddTodoInput(ToolInput):
    """Arguments for :class:`AddTodoTool`."""

    text: str = Field(
        min_length=1,
        max_length=MAX_TODO_CHARS,
        description="The task, short and imperative, e.g. 'Buy milk'.",
    )


class AddTodoTool(Tool[AddTodoInput]):
    """Appends an open todo."""

    name = "add_todo"
    description = (
        "Add a task to the user's todo list. Use for 'add X to my list', 'I need to X'. "
        "For something at a specific time use add_reminder instead."
    )
    Input = AddTodoInput
    category = ToolCategory.PRODUCTIVITY
    keywords = ("task", "shopping", "lägg", "attgöra", "göralista", "uppgift", "inköpslista")

    async def run(self, params: AddTodoInput, context: ToolContext) -> ToolResult:
        """Store the todo and echo it back so a mishearing is caught."""
        memory = require_memory(context, self.name)
        todo = await memory.add_todo(params.text)
        return ToolResult.success(
            f"Added '{todo.text}' to your list.", data={"id": todo.id, "text": todo.text}
        )
