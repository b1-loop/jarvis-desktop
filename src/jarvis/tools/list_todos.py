"""Tool: read out the todo list.

"What's on my list?". Open items by default; finished ones on request. The
ids in ``data`` and ``detail`` are what complete_todo takes.
"""

from __future__ import annotations

from typing import Final

from pydantic import Field, JsonValue

from jarvis.agent.tool import Tool, ToolCategory, ToolContext, ToolInput, ToolResult
from jarvis.tools._common import require_memory

_SPOKEN_ITEMS: Final = 5


class ListTodosInput(ToolInput):
    """Arguments for :class:`ListTodosTool`."""

    include_done: bool = Field(
        default=False, description="Also list items that were already ticked off."
    )


class ListTodosTool(Tool[ListTodosInput]):
    """Lists todos with their ids."""

    name = "list_todos"
    description = (
        "List the user's todo items (open ones unless include_done) with their ids, oldest "
        "first. Use the id with complete_todo."
    )
    Input = ListTodosInput
    category = ToolCategory.PRODUCTIVITY

    async def run(self, params: ListTodosInput, context: ToolContext) -> ToolResult:
        """Fetch and read out the first few."""
        memory = require_memory(context, self.name)
        todos = await memory.list_todos(include_done=params.include_done)
        if not todos:
            return ToolResult.success("Your todo list is empty.", data={"todos": []})
        rows: list[JsonValue] = [{"id": t.id, "text": t.text, "done": t.done} for t in todos]
        names = "; ".join(todo.text for todo in todos[:_SPOKEN_ITEMS])
        more = len(todos) - min(len(todos), _SPOKEN_ITEMS)
        count = "one item" if len(todos) == 1 else f"{len(todos)} items"
        spoken = f"You have {count}: {names}" + (f"; and {more} more." if more else ".")
        return ToolResult.success(
            spoken,
            detail="\n".join(f"[{t.id}] {'x' if t.done else ' '} {t.text}" for t in todos),
            data={"todos": rows},
        )
