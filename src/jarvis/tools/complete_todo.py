"""Tool: tick a todo off, by id or by matching its text.

"Tick off buy milk", "I've done the passport thing". People name todos, not
ids, so the tool also accepts a text fragment and resolves it against the
open items, refusing to guess when the fragment is ambiguous.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, Self

from pydantic import Field, JsonValue, model_validator

from jarvis.agent.tool import Tool, ToolCategory, ToolContext, ToolInput, ToolResult
from jarvis.memory.errors import NotFoundError
from jarvis.tools._common import require_memory

if TYPE_CHECKING:
    from jarvis.agent.tool import MemoryFacade

MAX_MATCH_CHARS: Final = 200


class CompleteTodoInput(ToolInput):
    """Arguments for :class:`CompleteTodoTool`; exactly one of the two fields."""

    todo_id: int | None = Field(
        default=None, ge=1, description="Id of the todo, as list_todos reports it."
    )
    text: str | None = Field(
        default=None,
        min_length=1,
        max_length=MAX_MATCH_CHARS,
        description="Part of the todo's text, e.g. 'milk'; used when the id is unknown.",
    )

    @model_validator(mode="after")
    def _exactly_one(self) -> Self:
        if (self.todo_id is None) == (self.text is None):
            raise ValueError("give exactly one of 'todo_id' or 'text'")
        return self


class CompleteTodoTool(Tool[CompleteTodoInput]):
    """Marks one open todo as done."""

    name = "complete_todo"
    description = (
        "Mark a todo item as done, identified by its id or by a fragment of its text. "
        "If the fragment matches several open items, nothing is ticked off and the "
        "candidates are returned so you can ask which one."
    )
    Input = CompleteTodoInput
    category = ToolCategory.PRODUCTIVITY

    async def run(self, params: CompleteTodoInput, context: ToolContext) -> ToolResult:
        """Resolve the item, then complete it."""
        memory = require_memory(context, self.name)
        todo_id = params.todo_id
        if todo_id is None:
            resolved = await _resolve_by_text(memory, params.text or "")
            if isinstance(resolved, ToolResult):
                return resolved
            todo_id = resolved
        try:
            todo = await memory.complete_todo(todo_id)
        except NotFoundError as error:
            return ToolResult.failure(f"I can't find todo number {todo_id}.", detail=str(error))
        return ToolResult.success(
            f"Ticked off '{todo.text}'.", data={"id": todo.id, "text": todo.text}
        )


async def _resolve_by_text(memory: MemoryFacade, fragment: str) -> int | ToolResult:
    """Id of the single open todo containing ``fragment``, or a result explaining why not."""
    needle = fragment.lower()
    matches = [todo for todo in await memory.list_todos() if needle in todo.text.lower()]
    if len(matches) == 1:
        return matches[0].id
    if not matches:
        return ToolResult.failure(f"I can't find an open todo about {fragment}.")
    candidates: list[JsonValue] = [{"id": t.id, "text": t.text} for t in matches]
    return ToolResult.failure(
        f"Several todos mention {fragment}. Which one did you mean?",
        detail="\n".join(f"[{t.id}] {t.text}" for t in matches),
        data={"candidates": candidates},
    )


# Design notes:
# - An ambiguous fragment is a failed result with the candidates, never a
#   guess: ticking off the wrong item silently is worse than one question.
# - jarvis.memory.errors is the memory layer's public error module (no
#   storage code), so importing it keeps the tool independent of SQLite.
