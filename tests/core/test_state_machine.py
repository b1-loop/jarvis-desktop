import asyncio
from itertools import product

import pytest

from jarvis.core.events import EventBus, StateChanged
from jarvis.core.state_machine import (
    TRANSITIONS,
    AssistantState,
    InvalidTransitionError,
    StateMachine,
)
from jarvis.utils.errors import JarvisError

IDLE, LISTENING, THINKING, SPEAKING = AssistantState

VALID_EDGES = [
    (IDLE, LISTENING),
    (LISTENING, THINKING),
    (LISTENING, IDLE),
    (THINKING, SPEAKING),
    (THINKING, IDLE),
    (SPEAKING, IDLE),
    (SPEAKING, LISTENING),
]
INVALID_EDGES = [edge for edge in product(AssistantState, repeat=2) if edge not in VALID_EDGES]


@pytest.fixture
def bus() -> EventBus:
    return EventBus()


def test_transition_table_matches_the_documented_edges() -> None:
    edges = {(src, dst) for src, targets in TRANSITIONS.items() for dst in targets}

    assert edges == set(VALID_EDGES)


def test_starts_idle(bus: EventBus) -> None:
    assert StateMachine(bus).state is IDLE


@pytest.mark.parametrize(("source", "target"), VALID_EDGES)
async def test_valid_transition_changes_state_and_publishes(
    bus: EventBus, source: AssistantState, target: AssistantState
) -> None:
    machine = StateMachine(bus, initial=source)
    subscription = bus.subscribe(StateChanged)

    event = machine.transition(target, "test")

    assert machine.state is target
    published = await asyncio.wait_for(anext(subscription), 1.0)
    assert published == event
    assert (published.old, published.new, published.reason) == (source, target, "test")


@pytest.mark.parametrize(("source", "target"), INVALID_EDGES)
async def test_invalid_transition_raises_and_changes_nothing(
    bus: EventBus, source: AssistantState, target: AssistantState
) -> None:
    machine = StateMachine(bus, initial=source)
    subscription = bus.subscribe(StateChanged)

    assert not machine.can_transition(target)
    with pytest.raises(InvalidTransitionError) as info:
        machine.transition(target, "bad")

    assert machine.state is source
    assert info.value.current is source
    assert info.value.requested is target
    assert info.value.reason == "bad"
    assert source.value in str(info.value) and target.value in str(info.value)
    assert isinstance(info.value, JarvisError)
    assert bus.published == 0
    subscription.close()


def test_full_conversation_turn_including_barge_in(bus: EventBus) -> None:
    machine = StateMachine(bus)

    machine.transition(LISTENING, "wake word")
    machine.transition(THINKING, "speech ended")
    machine.transition(SPEAKING, "answer ready")
    machine.transition(LISTENING, "barge-in")
    machine.transition(IDLE, "silence")

    assert machine.state is IDLE


def test_on_enter_hooks_run_after_the_state_is_set(bus: EventBus) -> None:
    machine = StateMachine(bus)
    seen: list[tuple[AssistantState, str]] = []

    def hook(event: StateChanged) -> None:
        seen.append((machine.state, event.reason))

    machine.on_enter(LISTENING, hook)
    machine.transition(LISTENING, "wake word")
    machine.transition(THINKING, "speech ended")

    assert seen == [(LISTENING, "wake word")]


def test_on_enter_hook_can_be_removed(bus: EventBus) -> None:
    machine = StateMachine(bus)
    calls: list[StateChanged] = []
    remove = machine.on_enter(LISTENING, calls.append)

    remove()
    remove()  # idempotent
    machine.transition(LISTENING, "wake word")

    assert calls == []


def test_hook_exceptions_propagate_but_state_and_event_already_moved(bus: EventBus) -> None:
    machine = StateMachine(bus)
    seen: list[StateChanged] = []
    bus.subscribe_callback(StateChanged, seen.append)

    def broken(event: StateChanged) -> None:
        raise RuntimeError("hook failed")

    machine.on_enter(LISTENING, broken)
    with pytest.raises(RuntimeError, match="hook failed"):
        machine.transition(LISTENING, "wake word")

    assert machine.state is LISTENING
    assert len(seen) == 1
