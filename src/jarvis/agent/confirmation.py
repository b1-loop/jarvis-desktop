"""Central confirmation policy for side-effecting tools.

The decision "must the user approve this call?" is made in one place, not
inside each tool, so a forgotten check in a plugin cannot delete a file or
send an email unasked. The policy combines the tool's own declaration
(``requires_confirmation`` and the per-call ``needs_confirmation`` hook)
with a configurable list of categories that always confirm, and a
settings-level trusted mode that switches confirmation off explicitly.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Final

from pydantic import BaseModel

from jarvis.agent.tool import Tool, ToolCategory, ToolContext, ToolInputError, ToolResult
from jarvis.config import Settings
from jarvis.core.events import ConfirmationRequested, ConfirmationResolved
from jarvis.utils.logging import get_logger

log = get_logger(__name__)

DECLINED_SPOKEN: Final = "Okay, I won't do that."
"""What Jarvis says when the user turns a confirmation down."""


@dataclass(frozen=True, slots=True)
class NotNeeded:
    """The call may proceed without asking.

    Attributes:
        reason: Which rule decided, for logs.
    """

    reason: str


@dataclass(frozen=True, slots=True)
class Required:
    """The user must approve first.

    Attributes:
        prompt: The yes/no question to put to the user.
        reason: Which rule decided, for logs and the dashboard.
    """

    prompt: str
    reason: str


Decision = NotNeeded | Required


class ConfirmationPolicy:
    """Decides whether a tool call needs the user's approval and obtains it."""

    def __init__(
        self,
        *,
        confirm_destructive: bool = True,
        always_confirm_categories: Iterable[ToolCategory] = (),
    ) -> None:
        self._confirm_destructive = confirm_destructive
        self._always_confirm = frozenset(always_confirm_categories)
        if not confirm_destructive:
            log.warning("confirmation.trusted_mode", detail="no tool will ask before acting")

    @classmethod
    def from_settings(
        cls, settings: Settings, *, always_confirm_categories: Iterable[ToolCategory] = ()
    ) -> ConfirmationPolicy:
        """Build the policy from ``JARVIS_CONFIRM_DESTRUCTIVE`` and the given categories."""
        return cls(
            confirm_destructive=settings.confirm_destructive,
            always_confirm_categories=always_confirm_categories,
        )

    @property
    def trusted_mode(self) -> bool:
        """True when confirmation has been switched off in settings."""
        return not self._confirm_destructive

    def decide(self, tool: Tool[Any], params: BaseModel, context: ToolContext) -> Decision:
        """Classify one call. ``params`` must already be validated."""
        if self.trusted_mode:
            return NotNeeded("trusted mode is on")
        if tool.needs_confirmation(params, context):
            return Required(tool.confirmation_prompt(params), "the tool requires confirmation")
        if tool.category in self._always_confirm:
            return Required(
                tool.confirmation_prompt(params),
                f"the {tool.category.value} category always confirms",
            )
        return NotNeeded("no rule requires confirmation")

    async def execute_with_policy(
        self, tool: Tool[Any], raw_params: Mapping[str, object], context: ToolContext
    ) -> ToolResult:
        """Validate, ask the user if the policy says so, then run the tool.

        Returns:
            The tool's result, or an ``ok=False`` result when the arguments
            are invalid or the user declined.
        """
        try:
            params = tool.parse(raw_params)
        except ToolInputError as error:
            log.warning("tool.invalid_input", tool=tool.name, error=str(error))
            return tool.invalid_input_result(error)
        decision = self.decide(tool, params, context)
        log.debug("confirmation.decided", tool=tool.name, reason=decision.reason)
        if isinstance(decision, Required) and not await self._ask(tool, decision, context):
            return ToolResult.failure(
                DECLINED_SPOKEN, detail=f"The user declined to run {tool.name}."
            )
        return await tool.execute(params, context)

    async def _ask(self, tool: Tool[Any], decision: Required, context: ToolContext) -> bool:
        context.bus.publish(ConfirmationRequested(tool_name=tool.name, prompt=decision.prompt))
        approved = await context.ask_confirmation(decision.prompt)
        context.bus.publish(ConfirmationResolved(tool_name=tool.name, approved=approved))
        log.info("confirmation.resolved", tool=tool.name, approved=approved, reason=decision.reason)
        return approved


# Design notes:
# - ``decide`` takes the context because the shell tool's verdict depends on
#   the configured allowlist, which lives in settings; a tool cannot be
#   expected to know it at import time.
# - Trusted mode is a single boolean that defaults to on-guard and is only
#   read from settings, never toggled at runtime, so a spoken "stop asking"
#   can never be misheard into disabling it.
# - Declining returns a normal failed ToolResult instead of raising: the
#   LLM sees "the user declined" as a tool outcome and can wrap up the turn
#   gracefully instead of the agent loop handling a special exception.
