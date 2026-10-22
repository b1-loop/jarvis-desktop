"""Central assistant state machine: IDLE, LISTENING, THINKING, SPEAKING.

Every transition is validated against an explicit table and published on
the event bus as a ``StateChanged`` event, so the overlay, tray icon, logs
and orchestrator can never disagree about what Jarvis is doing. The table is
data, not code, so a reviewer can check the allowed edges at a glance.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from enum import StrEnum
from typing import Final

from jarvis.core.events import EventBus, StateChanged
from jarvis.utils.errors import JarvisError
from jarvis.utils.logging import get_logger

log = get_logger(__name__)


class AssistantState(StrEnum):
    """What Jarvis is doing right now, as the user would describe it."""

    IDLE = "idle"
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"


TRANSITIONS: Final[Mapping[AssistantState, frozenset[AssistantState]]] = {
    AssistantState.IDLE: frozenset({AssistantState.LISTENING}),
    AssistantState.LISTENING: frozenset({AssistantState.THINKING, AssistantState.IDLE}),
    AssistantState.THINKING: frozenset({AssistantState.SPEAKING, AssistantState.IDLE}),
    AssistantState.SPEAKING: frozenset({AssistantState.IDLE, AssistantState.LISTENING}),
}
"""Allowed edges. ``SPEAKING -> LISTENING`` is barge-in; ``LISTENING -> IDLE`` is silence."""


class InvalidTransitionError(JarvisError):
    """A transition that the table does not allow was requested.

    Attributes:
        current: State the machine was in.
        requested: State that was asked for.
        reason: The caller's stated reason, kept for the log line.
    """

    def __init__(self, current: AssistantState, requested: AssistantState, reason: str) -> None:
        self.current = current
        self.requested = requested
        self.reason = reason
        super().__init__(
            f"cannot transition from {current.value} to {requested.value} (reason: {reason})"
        )


EnterHook = Callable[[StateChanged], None]
"""Called synchronously right after the machine enters the hooked state."""


class StateMachine:
    """Owns the current ``AssistantState`` and enforces the transition table.

    Hooks registered with ``on_enter`` run synchronously after the event is
    published; they are for the orchestrator's own bookkeeping. Anything
    that merely observes state (UI, logs) should subscribe to
    ``StateChanged`` on the bus instead.
    """

    def __init__(self, bus: EventBus, *, initial: AssistantState = AssistantState.IDLE) -> None:
        self._bus = bus
        self._state = initial
        self._hooks: dict[AssistantState, list[EnterHook]] = {state: [] for state in AssistantState}

    @property
    def state(self) -> AssistantState:
        """The current state."""
        return self._state

    def can_transition(self, to: AssistantState) -> bool:
        """Whether the table allows moving from the current state to ``to``."""
        return to in TRANSITIONS[self._state]

    def transition(self, to: AssistantState, reason: str) -> StateChanged:
        """Move to ``to``, publish ``StateChanged`` and run the enter hooks.

        Args:
            to: Target state.
            reason: Short cause for logs and the dashboard, e.g. ``"barge-in"``.

        Returns:
            The published event, convenient for tests and callers that log.

        Raises:
            InvalidTransitionError: If the table has no edge to ``to``.
        """
        if not self.can_transition(to):
            raise InvalidTransitionError(self._state, to, reason)
        event = StateChanged(old=self._state, new=to, reason=reason)
        self._state = to
        log.info("state.changed", old=event.old.value, new=to.value, reason=reason)
        self._bus.publish(event)
        for hook in list(self._hooks[to]):
            hook(event)
        return event

    def on_enter(self, state: AssistantState, hook: EnterHook) -> Callable[[], None]:
        """Register ``hook`` to run whenever ``state`` is entered.

        Returns:
            A function that removes the hook again.
        """
        self._hooks[state].append(hook)

        def remove() -> None:
            if hook in self._hooks[state]:
                self._hooks[state].remove(hook)

        return remove


# Design notes:
# - No self-transitions: IDLE -> IDLE is an error, not a no-op, because a
#   caller asking for it almost always has a bookkeeping bug worth surfacing.
# - The event is published before the hooks run so observers see the new
#   state even if a hook raises; hook exceptions propagate to the caller
#   because hooks are the orchestrator's own code, not third-party listeners.
# - ``transition`` is synchronous. Publishing never blocks (see events.py),
#   so an ``async`` signature would only make the call sites noisier.
