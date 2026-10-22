import asyncio
from dataclasses import dataclass, field

import pytest

from jarvis.audio.engine import NativePlayback
from jarvis.core.events import EventBus
from jarvis.core.interruption import BargeInController, BargeInNotReadyError
from jarvis.core.state_machine import AssistantState, StateMachine
from tests.audio.fakes import FakeNativePlayback

GUARD = 0.02


@dataclass
class RecordingTarget:
    targets: list[NativePlayback | None] = field(default_factory=list)

    def enable_barge_in(self, playback: NativePlayback | None) -> None:
        self.targets.append(playback)

    @property
    def current(self) -> NativePlayback | None:
        return self.targets[-1] if self.targets else None


@dataclass
class FakeTurns:
    running: bool = True
    cancels: int = 0

    def cancel_current_turn(self) -> bool:
        self.cancels += 1
        return self.running


@dataclass
class FakeCapture:
    started: int = 0

    def begin_barge_in_capture(self) -> None:
        self.started += 1


def speaking_machine() -> StateMachine:
    state = StateMachine(EventBus())
    state.transition(AssistantState.LISTENING, "test")
    state.transition(AssistantState.THINKING, "test")
    state.transition(AssistantState.SPEAKING, "test")
    return state


@pytest.fixture
def playback() -> FakeNativePlayback:
    return FakeNativePlayback()


@pytest.fixture
def target() -> RecordingTarget:
    return RecordingTarget()


def controller(
    target: RecordingTarget, playback: FakeNativePlayback, state: StateMachine
) -> BargeInController:
    return BargeInController(
        engine=target, playback=playback, state=state, echo_guard_seconds=GUARD
    )


async def test_onset_during_the_echo_guard_cannot_stop_playback(
    target: RecordingTarget, playback: FakeNativePlayback
) -> None:
    barge_in = controller(target, playback, speaking_machine())

    barge_in.playback_started()

    # Inside the guard window the engine has no target: Jarvis's own voice
    # leaking into the microphone cannot stop it.
    assert target.current is None
    assert not barge_in.armed
    await asyncio.sleep(GUARD * 3)
    assert target.current is playback
    assert barge_in.armed


async def test_finished_playback_disarms(
    target: RecordingTarget, playback: FakeNativePlayback
) -> None:
    barge_in = controller(target, playback, speaking_machine())
    barge_in.playback_started()
    await asyncio.sleep(GUARD * 3)

    barge_in.playback_finished()

    assert target.current is None
    assert not barge_in.armed


async def test_finishing_inside_the_guard_never_arms(
    target: RecordingTarget, playback: FakeNativePlayback
) -> None:
    barge_in = controller(target, playback, speaking_machine())
    barge_in.playback_started()
    barge_in.playback_finished()
    await asyncio.sleep(GUARD * 3)

    assert target.targets == []


async def test_interruption_captures_cancels_and_listens(
    target: RecordingTarget, playback: FakeNativePlayback
) -> None:
    state = speaking_machine()
    turns, capture = FakeTurns(), FakeCapture()
    barge_in = controller(target, playback, state)
    barge_in.attach(turns=turns, capture=capture)
    barge_in.playback_started()
    await asyncio.sleep(GUARD * 3)

    barge_in.playback_interrupted()

    assert capture.started == 1
    assert turns.cancels == 1
    assert state.state is AssistantState.LISTENING
    assert barge_in.interruptions == 1
    assert target.current is None


async def test_interruption_outside_speaking_is_ignored(
    target: RecordingTarget, playback: FakeNativePlayback
) -> None:
    state = StateMachine(EventBus())
    turns, capture = FakeTurns(), FakeCapture()
    barge_in = controller(target, playback, state)
    barge_in.attach(turns=turns, capture=capture)

    barge_in.playback_interrupted()

    assert state.state is AssistantState.IDLE
    assert turns.cancels == capture.started == 0


async def test_interruption_before_attach_is_an_error(
    target: RecordingTarget, playback: FakeNativePlayback
) -> None:
    barge_in = controller(target, playback, speaking_machine())

    with pytest.raises(BargeInNotReadyError):
        barge_in.playback_interrupted()


async def test_suspended_keeps_barge_in_off_and_disarms_at_once(
    target: RecordingTarget, playback: FakeNativePlayback
) -> None:
    barge_in = controller(target, playback, speaking_machine())
    barge_in.playback_started()
    await asyncio.sleep(GUARD * 3)
    assert barge_in.armed

    with barge_in.suspended():
        assert not barge_in.armed
        barge_in.playback_started()
        await asyncio.sleep(GUARD * 3)
        assert not barge_in.armed

    barge_in.playback_started()
    await asyncio.sleep(GUARD * 3)
    assert barge_in.armed
